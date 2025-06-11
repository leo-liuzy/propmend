import copy
import importlib
import logging
import random

import os
import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

import gc
from trainer import EditTrainer
import vars
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

from copy import deepcopy
import models
import utils
from utils import EditLoss, StrEnum, get_eval_result, add_eos, load_jsonlines


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

logging.basicConfig(format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


icl_prompt = "\n".join(
    [
        "Q: When did the simpsons first air on television?",
        "A: December 17, 1989",
        "Q: Who has more super bowl wins afc or nfc?",
        "A: NFC",
        "Q: Is the federal court the same as the supreme court?",
        "A: No",
    ]
)


def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM) and not isinstance(model, transformers.Qwen2ForCausalLM):
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


@hydra.main(config_path="config", config_name="config")
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)
    add_padding(tokenizer, model)

    from data_classes.zsre import ZsreDataset

    train_set = ZsreDataset(
        tokenizer, f"{vars.DATA_DIR}/zsre/structured_zeroshot-train-new_annotated_final.jsonl", config
    )
    val_set = ZsreDataset(tokenizer, f"{vars.DATA_DIR}/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)
    tokenizer = val_set.tok

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

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

    trainer = EditTrainer(alg, config, train_set, val_set)
    print("Task: ", config.task)

    assert hasattr(config, "spec_question")
    assert hasattr(config, "test_data")
    if config.test_data == "4K_test_id":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_structure_data_id_entity152_rel31.jsonl"
        )
        config.val_steps = 500
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "4K_test_ood":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_structure_data_ood_entity37_rel7.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "4K_test_ood-entity":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_structure_data_ood-entity_entity37_rel31.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "4K_test_ood-relation":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_structure_data_ood-relation_entity152_rel7.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "profile":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_structure_data_id_entity152_rel31.jsonl"
        )
        config.val_steps = 50
        edit_dev_dataset = edit_dev_dataset[:config.val_steps]
        assert len(edit_dev_dataset) == config.val_steps
    else:
        raise NotImplementedError("Only all_propagation is supported for dattest_datae_data")

    all_results = []
    edit_model_infos = []
    assert config.val_steps <= len(edit_dev_dataset)
    assert config.eval_only

    eos_token_id = tokenizer.eos_token_id

    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        datum = edit_dev_dataset[i]

        prefixs = [f["prefix"] for f in datum["facts"]]
        assert config.edit_loss == EditLoss.sft, f"edit_loss `{config.edit_loss}` is not supported"
        targets = [" " + f["target"].strip() + tokenizer.eos_token for f in datum["facts"]]
        sentences = [p + t for p, t in zip(prefixs, targets)]

        tokenizer.padding_side = "left"
        sentences_toks = add_eos(
            tokenizer(sentences, padding=True, return_tensors="pt"), eos_token_id, ignore=True
        )
        tokenizer.padding_side = "right"
        targets_toks = add_eos(
            tokenizer(targets, padding=True,  return_tensors="pt", add_special_tokens=False),
            eos_token_id,
            ignore=True, # since I already manually added eos in the targets
        )
        tokenizer.padding_side = "left"
        edit_inner = {
            "input_ids": sentences_toks["input_ids"],
            "attention_mask": sentences_toks["attention_mask"],
            "labels": val_set.get_edit_labels(targets_toks["input_ids"]),
        }

        print("Input for EDIT: ")
        print("[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(edit_inner["input_ids"])) + "]")
        print("Label for EDIT: ")
        print("[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(targets_toks["input_ids"])) + "]")
        print()

        edit_inner = utils.dict_to(edit_inner, config.device)

        question_types = [
            ("efficacy", datum["questions"]),
        ]
        
        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)

        for question_type, questions in question_types:
            logging.info(f"Question type: {question_type}")
            for question_key in ["alias_question", "unalias_question"]:
                for q_i, question in enumerate(questions):

                    post_result_df = get_eval_result(
                        question=question[question_key],
                        answer=str(question["answer"]),
                        model=edited_model.model,
                        tokenizer=tokenizer,
                        config=config,
                        generation_config=generation_config,
                    )
                    post_result_df.insert(0, "question_type", question_type)
                    post_result_df.insert(0, "question_key", question_key)
                    post_result_df.insert(0, "stage", "post-edit")
                    post_result_df.insert(
                        0, "edit_input", "\n\n".join(f"[[{tokenizer.decode(s)}]]" for s in sentences_toks["input_ids"])
                    )

                    post_result_df.insert(0, "id", str(i))
                    all_results.append(post_result_df)

        del edited_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results = pd.concat(all_results)

    if config.generation.save_dir:
        save_dir = config.generation.save_dir
        if os.path.abspath(config.generation.save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{config.generation.save_dir}"

        LOG.info(f"Saving to dir: {save_dir}")

        os.makedirs(save_dir, exist_ok=True)
        fpath = (
            f"{save_dir}/mend_eval_loss={config.edit_loss}_input={config.edit_input}_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl"
            + ("_e+s" if config.spec_question else "_e")
            + f"_{config.test_data}-question"
            + ".xlsx"
        )
        LOG.info(f"Saving to dir: {fpath}")

        all_results.to_excel(fpath, index=False)


if __name__ == "__main__":
    run()
