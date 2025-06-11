import os
import json
from datasets import Dataset
from typing import Optional
import pickle as pkl
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import vars
from utils import load_jsonlines


def prepare_sft_text(args, dataset: list, tokenizer):

    new_dataset = []
    has_show_example = False
    for datum in dataset:
        q = datum["question"]
        a = str(datum["answer"])
        t = f"{q}{a}" if a[0] == " " else f"{q} {a}"
        t += tokenizer.eos_token
        if not has_show_example:
            print(f"Example: -> {t}")
            has_show_example = True
        datum[args.dataset_text_field] = t
        new_dataset.append(datum)
    return new_dataset

@dataclass
class CustomConfig:
    sft_stage: str

parser = HfArgumentParser((SFTConfig, CustomConfig))
(args, custom_cfg) = parser.parse_args_into_dataclasses()

if custom_cfg.sft_stage == "qa":
    model_name_or_path = "Qwen/Qwen2.5-1.5B"
else:
    assert custom_cfg.sft_stage == "qa-additional"
    os.makedirs(f"{vars.PROJ_DIR}/models", exist_ok=True)
    model_name_or_path = f"{vars.PROJ_DIR}/models/Qwen2.5-1.5B-eos-sft"


model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_eos_token=True, use_fast=False)
tokenizer.padding_side = "right"
original_vocab_size = len(tokenizer)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# tokenizer.add_special_tokens({'additional_special_tokens': ['<sft_token_1>']}, replace_additional_special_tokens=False)
model.resize_token_embeddings(len(tokenizer))

tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

assert tokenizer.eos_token != tokenizer.pad_token
assert tokenizer.eos_token_id != tokenizer.pad_token_id

if custom_cfg.sft_stage == "qa":
    train_dataset = prepare_sft_text(args, load_jsonlines(f"{vars.DATA_DIR}/qa-sft/train.jsonl"), tokenizer)
    valid_dataset = prepare_sft_text(args, load_jsonlines(f"{vars.DATA_DIR}/qa-sft/valid.jsonl"), tokenizer)
else:
    assert custom_cfg.sft_stage == "qa-additional"
    train_dataset = prepare_sft_text(
        args, load_jsonlines(f"{vars.DATA_DIR}/qa-additional-sft/light_weight_sft_content_curated_v1_sample=10.jsonl"), tokenizer
    )
    eval_dataset = None


train_dataset = Dataset.from_list(train_dataset)

response_template = "?"  # tokenizer.additional_special_tokens[0] # "?" alternative "Ġ?"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,  # type: ignore
    eval_dataset=valid_dataset,  # type: ignore
    args=args,
    data_collator=collator,
)

trainer.train()

trainer.model.config.pad_token_id = None
trainer.model.resize_token_embeddings(original_vocab_size)
trainer.model.save_pretrained(save_directory=args.output_dir)

trainer.accelerator.wait_for_everyone()

