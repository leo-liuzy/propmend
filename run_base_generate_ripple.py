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

from trainer import EditTrainer
import vars

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

import models
import utils
from utils import EditLoss, EditInput, load_jsonlines
from typing import Iterable

OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

logging.basicConfig(format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)

icl_prompt = "\n".join(
    [
        "Q: When did the simpsons first air on television?",
        "A: 1989",
        "Q: When was Jesus born?",
        "A: 6 to 4 BC",
        "Q: What year did the United State declare independence?",
        "A: 1776",
    ]
)

def add_padding(tokenizer, model):
    import transformers

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM) and not isinstance(model, transformers.Qwen2ForCausalLM):
        model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


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
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.gen_w_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]

    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    print(
        "Input for generation:",
        "[" + "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) + "]",
    )
    print("Label for generation:", "[" + answer + "]")
    print("--------------------")

    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
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
                "answer": answer.strip(),
                "predicted_answer_idx": 0,
                "predicted_answer": predicted_answer.strip(),
            }
        ]
    )
    return model_response


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
    model = model.to(config.device)

    from data_classes.zsre import ZsreDataset

    val_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)

    generation_config = GenerationConfig(
        do_sample=False,  # Greedy
        top_k=None,
        top_p=None,
        temperature=None,
        max_new_tokens=30,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert hasattr(config, "test_data")
    if config.test_data == "all":
        val_data = load_jsonlines(f"{vars.DATA_DIR}/ripple_edits/test.jsonl")
        config.val_steps = 500
    else:
        raise ValueError(f"Unknown test_data: {config.test_data}")

    all_results = []
    assert config.val_steps <= len(val_data)
    assert config.eval_only

    assert hasattr(config, "ice")  # and config.ice

    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id

    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        datum = val_data[i]

        sentences = [datum["edit"]["prompt"].strip()]

        outerloop_queries = []
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

        for question_type, test_queries in question_types:
            for q_i, test_query in enumerate(test_queries):
                answer_candidates = [a["value"] for a in test_query["answers"]]
                answer = answer_candidates[0]

                if config.ice:
                    test_queries_q_str = (
                        f"Imagine that {sentences[0][0].lower() + sentences[0][1:]} {test_query['prompt'].strip()}"
                    )
                else:
                    test_queries_q_str = f"{test_query['prompt'].strip()}"
                test_queries_a_str = answer.strip()
                test_queries_str = (
                    test_queries_q_str + (" " if test_queries_a_str[0] != " " else "") + test_queries_a_str
                )

                acc_toks = add_eos(
                    tokenizer(test_queries_str, padding=True, return_tensors="pt", add_special_tokens=True),
                    eos_token_id,
                    ignore=not config.add_eos,
                )
                acc_toks = utils.dict_to(acc_toks, config.device)
                sft_labels = val_set.get_edit_labels(
                    add_eos(
                        tokenizer(
                            [(" " if test_queries_a_str[0] != " " else "") + test_queries_a_str],
                            padding=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        ),
                        eos_token_id,
                        ignore=not config.add_eos,
                    )["input_ids"]
                ).to(config.device)

                clm_labels = val_set.get_edit_labels(acc_toks["input_ids"]).to(config.device)

                print("Input for [Q][A] Accuracy: ")
                print("[" + tokenizer.decode(acc_toks["input_ids"][0]) + "]")
                print("SFT label:", "[" + tokenizer.decode(sft_labels[0]) + "]")
                print("CLM label(before ShiftLeft):", "[" + tokenizer.decode(clm_labels[0]) + "]")
                print()

                if config.do_generation:
                    pre_result_df = generate(
                        test_queries_q_str,
                        test_queries_a_str,
                        config,
                        model,
                        tokenizer,
                        generation_config,
                    )
                else:
                    pre_result_df = pd.DataFrame([{"predicted_answer_idx": 0}])
                assert len(pre_result_df) == 1

                pre_result_df.insert(0, "input", "\n\n".join(f"[[{s}]]" for s in [sentences[0]]))
                pre_result_df.insert(1, "stage", "pre-edit")
                pre_result_df.insert(0, "relation", f"{test_query['relation']}")
                pre_result_df.insert(0, "question_tag", f"{question_type}_{test_query['question_type']}")
                pre_result_df.insert(0, "question_type", question_type)
                pre_result_df.insert(0, "id", str(i))

                all_results.append(pre_result_df)

    all_results = pd.concat(all_results)

    if config.generation.save_dir:
        save_dir = config.generation.save_dir
        if os.path.abspath(config.generation.save_dir) != config.generation.save_dir:
            # using relative path
            save_dir = f"{base_dir}/{config.generation.save_dir}"
        save_dir = os.path.join(save_dir, config.test_data)

        os.makedirs(save_dir, exist_ok=True)
        fpath = f"{save_dir}/base_n={config.val_steps}_prompt={config.generation.prompt}_{'w' if config.do_generation else 'wo'}-gen_{'w' if hasattr(config, 'add_icl') and config.add_icl else 'wo'}-icl_ice={config.ice}.xlsx"
        all_results.to_excel(
            fpath,
            index=False,
        )
        LOG.info(f"Saving to dir: {fpath}")


if __name__ == "__main__":
    run()
