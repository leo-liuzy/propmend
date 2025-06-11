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


from typing import List, Dict

from copy import deepcopy
import models
import utils
from utils import EditLoss, StrEnum, add_eos, load_jsonlines


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


def generate_multi_answers(
    context: str,
    answers: List[str],
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
    return model_response


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
        tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-train-new_annotated_final.jsonl", config
    )
    val_set = ZsreDataset(tokenizer, f"{base_dir}/data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config)
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
    # import pdb

    # pdb.set_trace()
    if config.test_data == "4K_test_id":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_text_data_id_entity152_rel31.jsonl"
        )
        config.val_steps = 500
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "4K_test_ood":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_text_data_ood_entity37_rel7.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "4K_test_ood-relation":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_text_data_ood-relation_entity152_rel7.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "4K_test_ood-entity":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_text_data_ood-entity_entity37_rel31.jsonl"
        )
        config.val_steps = 350
        assert len(edit_dev_dataset) == config.val_steps
    elif config.test_data == "profile":
        edit_dev_dataset = load_jsonlines(
            f"{vars.DATA_DIR}/4K_controlled_ripple_edit/test_text_data_id_entity152_rel31.jsonl"
        )
        config.val_steps = 50
        edit_dev_dataset = edit_dev_dataset[:config.val_steps]
        assert len(edit_dev_dataset) == config.val_steps
    else:
        raise NotImplementedError(f"test_data `{config.test_data}` is not supported")

    all_results = []
    edit_model_infos = []

    assert config.val_steps <= len(edit_dev_dataset)
    assert config.eval_only
    if hasattr(config, "add_icl") and config.add_icl:
        eos_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
    else:
        eos_token_id = tokenizer.eos_token_id

    for i in tqdm(range(config.val_steps), desc=f"Running eval on {config.task}"):
        datum = edit_dev_dataset[i]

        sentences = [datum["text"]]

        assert config.edit_loss == EditLoss.clm, f"edit_loss `{config.edit_loss}` is not supported"
        sentences_toks = targets_toks = add_eos(
            tokenizer(sentences, padding=True, return_tensors="pt", add_special_tokens=True),
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

        # edit the model with MEND
        edited_model, model_info = trainer.model.edit(edit_inner)
        model_info["input"] = sentences[0]
        model_info["target"] = tokenizer.decode(targets_toks["input_ids"][0])
        edit_model_infos.append(model_info)

        question_types = [
            ("efficacy", datum["questions"]),
        ]

        for question_type, questions in question_types:
            logging.info(f"Question type: {question_type}")
            for question_key in ["alias_question", "unalias_question"]:
                for q_i, question in enumerate(questions):
                    post_result_df = generate_multi_answers(
                        context=question[question_key],
                        answers=str(question["answer"]),
                        config=config,
                        model=edited_model.model,
                        tokenizer=tokenizer,
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
            + f"_{config.test_data}-question"
            + ".xlsx"
        )
        LOG.info(f"Saving to dir: {fpath}")
        all_results.to_excel(fpath, index=False)



if __name__ == "__main__":
    run()
