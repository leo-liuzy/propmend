import os
import json
from datasets import Dataset
from typing import Optional, List
import pickle as pkl
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, GenerationConfig, Trainer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from transformers import DataCollatorForLanguageModeling
import vars, io
import torch
import gc
import utils
from utils import StrEnum, load_jsonlines
import pandas as pd

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def add_eos(tokenizer_output, eos_token_id, ignore=False):
    if ignore:
        return tokenizer_output
    return {
        k: torch.concat(
            [
                v,
                torch.full(
                    (v.shape[0], 1),  # shape of the constant tensor
                    (
                        1
                        if k == "attention_mask"
                        else eos_token_id  # this is to teach the model to end after outputing the answer.
                    ),
                ),
            ],
            dim=-1,
        )
        for k, v in tokenizer_output.items()
    }


def generate(
    context: str,
    answer: str,
    config,
    model,
    tokenizer,
    generation_config,
):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.add_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]

    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    logging.info(
        "Input for generation: "
        + "["
        + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"]))
        + "]"
    )
    logging.info("Label for generation: " + "[" + answer + "]")

    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    generated_texts = [t.replace(ctx_decoded, "") for t in generated_texts]
    # predicted_answer = generated_texts[0]
    model_response_content = []
    for g_i, generated_text in enumerate(generated_texts):
        predicted_answer = generated_text.strip()
        model_response_content.append(
            {
                "question": context,
                "answer": answer.strip(),
                "predicted_answer_idx": g_i,
                "predicted_answer": predicted_answer,
            }
        )
    model_response = pd.DataFrame(model_response_content)


    return model_response


def generate_multi_answers(
    context: str,
    answers: List[str],
    config,
    model,
    tokenizer,
    generation_config,
):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.add_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]

    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    print(
        "Input for generation:",
        "[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) + "]",
    )
    print("Label for generation:", "[" + str(answers) + "]")
    print("--------------------")

    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    generated_texts = [t.replace(ctx_decoded, "") for t in generated_texts]
    predicted_answer = generated_texts[0]
    if hasattr(config, "add_icl") and config.add_icl:
        # if using ICL, extract by the first new line
        if "\n" in predicted_answer:
            predicted_answer = predicted_answer[: predicted_answer.find("\n")]

    model_response = pd.DataFrame(
        [
            {
                "question": context,
                "answer": answers,
                "predicted_answer_idx": 0,
                "predicted_answer": predicted_answer.strip(),
            }
        ]
    )
    return score_df(model_response)


def get_edit_labels(labels, tokenizer):
    return labels.masked_fill(labels == tokenizer.pad_token_id, -100)


def prepare_clm_text(args, custom_cfg, instance, tokenizer):
    def tokenize(element):
        outputs = tokenizer(
            element[args.dataset_text_field],
            truncation=True,
            max_length=args.max_seq_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length <= args.max_seq_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    # assert len(tokenizer.additional_special_tokens) == 1
    dataset = [instance[custom_cfg.text_data]]

    new_dataset = []
    for datum in dataset:
        new_dataset.append({args.dataset_text_field: datum + tokenizer.eos_token})
    dataset = Dataset.from_list(new_dataset)

    tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=[args.dataset_text_field])
    return tokenized_datasets


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class InputFormat(StrEnum):
    two_single_hop = "two-1hop"
    first_single_hop = "first-1hop"
    second_single_hop = "second-1hop"
    seen_hop = "seen-hop"


@dataclass
class CustomConfig:
    example_idx: int
    tunable_params: str
    device: Optional[str] = "cuda:0"
    add_eos_accuracy: Optional[bool] = True
    add_bos: Optional[bool] = True
    base_model_name: str
    save_dir_suffix: Optional[str] = None
    text_data: Optional[str] = "text"
    test_data: Optional[str] = "4K_test_id"


parser = HfArgumentParser((SFTConfig, CustomConfig))
(args, custom_cfg) = parser.parse_args_into_dataclasses()
model_name_or_path = f"{os.getcwd()}/models/{custom_cfg.base_model_name}"

logging.info(f"CustomConfig: {custom_cfg}")

exp_save_dir = f"{os.getcwd()}/controlled_ripple_exp_output/{custom_cfg.base_model_name}_clm-baseline_lr={args.learning_rate}_epoch={args.num_train_epochs}{'_' + custom_cfg.save_dir_suffix if custom_cfg.save_dir_suffix is not None else ''}_tunable-params={custom_cfg.tunable_params}"

os.makedirs(exp_save_dir, exist_ok=True)


if custom_cfg.test_data == "4K_test_id":
    individual_result_save_dir = f"{exp_save_dir}/individual_results_{custom_cfg.text_data}_id"
    cpt_dev_dataset = load_jsonlines(
        f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_data_id.jsonl"
    )
elif custom_cfg.test_data == "4K_test_ood":
    individual_result_save_dir = f"{exp_save_dir}/individual_results_{custom_cfg.text_data}_ood"
    cpt_dev_dataset = load_jsonlines(
        f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_data_ood.jsonl"
    )
elif custom_cfg.test_data == "4K_test_ood-entity":
    individual_result_save_dir = f"{exp_save_dir}/individual_results_{custom_cfg.text_data}_ood-entity"
    cpt_dev_dataset = load_jsonlines(
        f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_data_ood-entity.jsonl"
    )

elif custom_cfg.test_data == "4K_test_ood-relation":
    individual_result_save_dir = f"{exp_save_dir}/individual_results_{custom_cfg.text_data}_ood-relation"
    cpt_dev_dataset = load_jsonlines(
        f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_data_ood-relation.jsonl"
    )
elif custom_cfg.test_data == "profiling":
    individual_result_save_dir = f"{exp_save_dir}/individual_results_{custom_cfg.text_data}_profiling"
    cpt_dev_dataset = load_jsonlines(
        f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_data_id.jsonl"
    )
    cpt_dev_dataset = cpt_dev_dataset[:50]
else:
    raise NotImplementedError(f"test_data: {custom_cfg.test_data}")

os.makedirs(individual_result_save_dir, exist_ok=True)


instance = cpt_dev_dataset[custom_cfg.example_idx]

logging.info(f"Example: {instance}")

fpath = (
    f"{individual_result_save_dir}/{custom_cfg.example_idx}_eval_results"
    + ("_e+s" if custom_cfg.spec_question else "_e")
    + ".xlsx"
)
if os.path.exists(fpath):
    logging.info("=" * 20 + "Already evaluated" + "=" * 20)
    exit(0)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False, device_map=custom_cfg.device)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    add_eos_token=True,
    use_fast=False,
)
tokenizer.padding_side = "right"
original_vocab_size = len(tokenizer)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

assert tokenizer.eos_token != tokenizer.pad_token
assert tokenizer.eos_token_id != tokenizer.pad_token_id

if custom_cfg.tunable_params != "all":
    # assert custom_cfg.tunable_params in custom_cfg.base_model_name
    if custom_cfg.tunable_params == "top3-mlp":
        params = [
            "model.layers.13.mlp.gate_proj.weight",
            "model.layers.13.mlp.up_proj.weight",
            "model.layers.13.mlp.down_proj.weight",
            "model.layers.14.mlp.gate_proj.weight",
            "model.layers.14.mlp.up_proj.weight",
            "model.layers.14.mlp.down_proj.weight",
            "model.layers.15.mlp.gate_proj.weight",
            "model.layers.15.mlp.up_proj.weight",
            "model.layers.15.mlp.down_proj.weight",
        ]
    elif custom_cfg.tunable_params == "midupper3-mlp":
        params = [
            "model.layers.10.mlp.gate_proj.weight",
            "model.layers.10.mlp.up_proj.weight",
            "model.layers.10.mlp.down_proj.weight",
            "model.layers.11.mlp.gate_proj.weight",
            "model.layers.11.mlp.up_proj.weight",
            "model.layers.11.mlp.down_proj.weight",
            "model.layers.12.mlp.gate_proj.weight",
            "model.layers.12.mlp.up_proj.weight",
            "model.layers.12.mlp.down_proj.weight",
        ]
    else:
        raise ValueError(f"Unknown tunable_params: {custom_cfg.tunable_params}")

    for n, param in model.named_parameters():
        if any(p in n for p in params):
            param.requires_grad = True
        else:
            param.requires_grad = False

print_trainable_parameters(model)

generation_config = GenerationConfig(
    do_sample=False,  # Greedy
    top_k=None,
    top_p=None,
    temperature=None,
    max_new_tokens=20,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

train_dataset = prepare_clm_text(args, custom_cfg, instance, tokenizer)

logging.info(f"Setting per_device_train_batch_size == {len(train_dataset)}")
args.per_device_train_batch_size = len(train_dataset)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


trainer = Trainer(
    model,
    train_dataset=train_dataset,  # type: ignore
    # eval_dataset=valid_dataset, # type: ignore
    args=args,
    data_collator=data_collator,
)

trainer.train()
trainer.accelerator.wait_for_everyone()


model = trainer.model
# clear internal pointer in trainer/accelerator
trainer.accelerator.free_memory(trainer.model, trainer.optimizer, trainer.lr_scheduler)
del trainer.model, trainer.optimizer, trainer.lr_scheduler
del trainer
# clear cache to make spaces in GPU and CPU
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

eos_token_id = tokenizer.eos_token_id

question_types = [
    ("efficacy", instance["questions"]),
]

logging.info("Start evaluating model: Generation, Accuracy")

all_result_df = []
for question_type, questions in question_types:
    logging.info(f"Question type: {question_type}")

    for question_key in ["alias_question", "unalias_question"]:
        for q_i, question in tqdm(enumerate(questions), total=len(questions)):
            test_queries_a_str = str(question["answer"])
            test_queries_q_str = question[question_key]

            post_result_df = generate(
                test_queries_q_str, test_queries_a_str, custom_cfg, model, tokenizer, generation_config
            )

            post_result_df.insert(0, "question_key", question_key)
            post_result_df.insert(0, "stage", "post-edit")
            if "efficacy" in question_type:
                post_result_df.insert(0, "question_tag", f"{question_type}_{question['question_template']}")
            else:
                post_result_df.insert(0, "question_tag", f"{question_type}_{q_i}")
            post_result_df.insert(0, "question_type", question_type)
            post_result_df.insert(0, "id", str(custom_cfg.example_idx))

            all_result_df.append(post_result_df)
all_result_df = pd.concat(all_result_df)

logging.info(f"Saving individual results to {individual_result_save_dir}")

all_result_df.to_excel(
    fpath,
    index=False,
)
