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
    if not isinstance(model, transformers.LlamaForCausalLM):
        #     model.model.embed_tokens.weight[-1] = model.model.embed_tokens.weight.mean(0)
        # else:
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

    if config.test_data == "all":
        edit_dev_dataset = load_jsonlines(f"{vars.DATA_DIR}/ripple_edit/test_aug.jsonl")
        config.val_steps = 500
    else:
        raise NotImplementedError("Only all_propagation is supported for test_data")

    all_results = []

    assert config.val_steps <= len(edit_dev_dataset)
    assert config.eval_only
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id

    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        datum = edit_dev_dataset[i]

        sentences = [datum["edit"]["prompt"]]

        assert config.edit_loss == EditLoss.sft, f"edit_loss `{config.edit_loss}` is not supported"

        if config.test_data == "all":
            targets = [" " + datum["edit"]["target"].strip()]
        else:
            targets = [" " + datum["edit"]["object"].strip()]
        sentences = [datum["edit"]["context"] + targets[0]]

        sentences_toks = add_eos(
            tokenizer(sentences, padding=True, return_tensors="pt"), eos_token_id, ignore=not config.add_eos
        )
        targets_toks = add_eos(
            tokenizer(targets, padding=True, return_tensors="pt", add_special_tokens=False),
            eos_token_id,
            ignore=not config.add_eos,
        )

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

        all_datum_result_df = []
        outerloop_queries = []
        # collect propagation queries
        for k in ["Logical_Generalization", "Compositionality_I", "Compositionality_II", "Subject_Aliasing"]:
            for instance in datum[k]:
                for q in instance["test_queries"]:
                    if (
                        len(q["answers"]) > 0
                        and len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0
                    ):
                        q["question_type"] = k
                        outerloop_queries.append(q)

        assert len(outerloop_queries) > 0

        locality_queries = []
        for k in ["Relation_Specificity", "Forgetfulness"]:
            for instance in datum[k]:
                for q in instance["test_queries"]:
                    if (
                        len(q["answers"]) > 0
                        and len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0
                    ):
                        q["question_type"] = k
                        locality_queries.append(q)
        assert len(locality_queries) > 0

        question_types = [
            ("efficacy", outerloop_queries),
            ("specificity", locality_queries),
        ]

        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)

        for question_type, questions in question_types:
            logging.info(f"Question type: {question_type}")

            for q_i, question in enumerate(questions):
                answer_candidates = [a["value"] for a in question["answers"]]
                answer = answer_candidates[0]

                post_result_df = get_eval_result(
                    question=question["prompt"],
                    answer=answer,
                    model=edited_model.model,
                    tokenizer=tokenizer,
                    config=config,
                    generation_config=generation_config,
                )
                post_result_df.insert(0, "stage", "post-edit")
                post_result_df.insert(
                    0, "edit_input", "\n\n".join(f"[[{tokenizer.decode(s)}]]" for s in sentences_toks["input_ids"])
                )
                post_result_df.insert(0, "relation", f"{question['relation']}")
                post_result_df.insert(0, "question_tag", f"{question_type}_{question['question_type']}")
                post_result_df.insert(0, "question_type", question_type)
                post_result_df.insert(0, "id", str(i))

                all_datum_result_df.append(post_result_df)

        del edited_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_datum_result_df = pd.concat(all_datum_result_df)
        all_results.append(all_datum_result_df)

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
