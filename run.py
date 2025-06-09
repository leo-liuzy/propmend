import copy
import random
import importlib
import logging

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils
from utils import StrEnum

import vars
from trainer import EditTrainer
import models
import transformers


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if not isinstance(model, transformers.LlamaForCausalLM) and not isinstance(model, transformers.Qwen2ForCausalLM):
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

    if config.task == "qa" or config.task == "zsre":
        from data_classes.zsre import ZsreDataset

        add_padding(tokenizer, model)

        train_set = ZsreDataset(
            tokenizer,
            f"{vars.DATA_DIR}/zsre/structured_zeroshot-train-new_annotated_final.jsonl",
            config,
            size=getattr(config, "train_size", None),
        )
        val_set = ZsreDataset(
            tokenizer, f"{vars.DATA_DIR}/zsre/structured_zeroshot-dev-new_annotated_final.jsonl", config
        )
    elif config.task == "qa" or config.task == "ripple_edits":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits import RippleEditsDataset

        assert hasattr(config, "train_set_size"), "ripple_edits config must be provided"
        train_set = RippleEditsDataset(
            tokenizer,
            f"{vars.DATA_DIR}/ripple_edits/train.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsDataset(
            tokenizer,
            f"{vars.DATA_DIR}/ripple_edits/valid.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "ripple_edits_mend":
        add_padding(tokenizer, model)
        from data_classes.ripple_edits_mend import RippleEditsMENDDataset

        train_set = RippleEditsMENDDataset(
            tokenizer,
            f"{vars.DATA_DIR}/ripple_edits/train_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = RippleEditsMENDDataset(
            tokenizer,
            f"{vars.DATA_DIR}/ripple_edits/valid_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
    elif config.task == "qa" or config.task == "syn_story":
        add_padding(tokenizer, model)
        from data_classes.syn_story import SynStoryDataset

        assert hasattr(config, "train_set_size"), "train_set_size must be provided"
        assert hasattr(config, "train_prefix"), "train_prefix must be provided"
        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/train_text_data_id_entity152_rel31.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/valid_text_data_id_entity152_rel31.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")
    elif config.task == "qa" or config.task == "syn_story_mend":
        add_padding(tokenizer, model)
        from data_classes.syn_story_mend import SynStoryMENDDataset

        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStoryMENDDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/train_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStoryMENDDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/valid_mend.jsonl",
            config,
            max_length=tokenizer.model_max_length,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")
    elif config.task == "qa" or config.task == "syn_story_ablate_propagation":
        add_padding(tokenizer, model)
        from data_classes.syn_story import SynStoryDataset

        assert hasattr(config, "train_set_size"), "train_set_size must be provided"
        assert hasattr(config, "train_prefix"), "train_prefix must be provided"
        
        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/train_text_data_id_entity152_rel31_paraphrase-only.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStoryDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/valid_text_data_id_entity152_rel31_paraphrase-only.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")
    elif config.task == "qa" or config.task == "syn_story_ablate_cpt":
        add_padding(tokenizer, model)
        from data_classes.syn_story_ablate_cpt import SynStorySFTDataset

        assert hasattr(config, "train_set_size"), "train_set_size must be provided"
        assert hasattr(config, "train_prefix"), "train_prefix must be provided"
        
        config.dataset += f"-{config.train_prefix}train"
        train_set = SynStorySFTDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/train_structure_data_id_entity152_rel31.jsonl",
            config,
            size=config.train_set_size,
            max_length=tokenizer.model_max_length,
        )
        val_set = SynStorySFTDataset(
            tokenizer,
            f"{vars.DATA_DIR}/{config.train_prefix}_story_propagation/valid_structure_data_id_entity152_rel31.jsonl",
            config,
            max_length=tokenizer.model_max_length,
            is_eval=True,
        )
        LOG.info(f"train_set size: {len(train_set)}")
        LOG.info(f"val_set size: {len(val_set)}")
        LOG.info(f"model_max_length: {tokenizer.model_max_length}")

    else:
        raise ValueError(f"Unrecognized task {config.task}")
    # train_set[0]
    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(config.ft.locality.batch_size + 1)
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(train_set.edit_generator(config.ft.locality.batch_size + 1))["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    trainer = EditTrainer(alg, config, train_set, val_set)
    trainer.run()


if __name__ == "__main__":
    run()
