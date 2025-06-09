import jsonlines
from torch.utils.data import Dataset
import random
from utils import EditBatchSampler, dict_to
import torch
from transformers import BartTokenizerFast, BartTokenizer
import logging
import typing
import json
import io
from utils import StrEnum, load_jsonlines
import numpy as np
from copy import deepcopy


LOG = logging.getLogger(__name__)
    

class SynStoryMENDDataset(Dataset):
    """

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        # size: typing.Optional[int] = None,
        max_length=32,
    ):
        super().__init__()
        self.tok = tokenizer
        self.data = load_jsonlines(data_path)
        self.config = config
        
        # if size is not None:
            # self.data = self.data[:size]
        self.show_first_example = False
        
        assert self.config.data.rephrase, "propogation question must be used."
        self.max_length = max_length
        if self.config.data.zsre_nq: # ! Leo: original if-condition: `and "train" not in data_path`
            self.use_nq = True
            LOG.info("** Using natural questions for zsre base samples **")
            from data_classes.nq import NQDataset
            self.nq = NQDataset(self.config.data.nq_path + ("/train.json" if "train" in data_path else "/validation.json"),tokenizer, config)
        else:
            self.use_nq = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, seed=None):
        
        
        assert all(e in self.data[item] for e in ["context", "paraphrase", "completion"])
        
        output = {
            "src": str(self.data[item]["context"]).strip(),
            "trg": str(self.data[item]["completion"]).strip(),
            "rephrase": str(self.data[item]["paraphrase"]).strip(),
        }
        return output

    def collate_fn(self, batch):
        src = [b["src"].strip() for b in batch]
        trg = [" " + b["trg"].strip() for b in batch]
        src = [src_ + trg_ for src_, trg_ in zip(src, trg)]
        
        rephrase = [b["rephrase"].strip() for b in batch]
        rephrase = [rephrase_ + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
        all_input_from_batchs = {
                "src": src,
                "trg": trg,
                "rephrase": rephrase,
            }
        batches = {
            f"{k1}_{k2}": 
                torch.concat(
                    [
                        v2, 
                        torch.full(
                            (v2.shape[0], 1), # shape of the constant tensor
                            (
                                1 
                                if k2 == "attention_mask" else
                                self.tok.eos_token_id # this is to teach the model to end after outputing the answer.
                            )
                        )
                    ], dim=-1)
            for k1, v1 in all_input_from_batchs.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                add_special_tokens="trg" not in k1, # make the SFT label free of BOS
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = all_input_from_batchs
        return batches

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tok.pad_token_id).any():
            raise ValueError("Left-padding not supported")

    def get_edit_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(n, memorize_mode=self.config.single_batch, loc_disjoint=not self.use_nq, seed=self.config.seed)

        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            assert len(edit_idxs) == 1
            # idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in edit_idxs])

            # ne = self.config.data.n_edits
            edit_inner = {}
            edit_inner["input_ids"] = toks["src_input_ids"]
            edit_inner["attention_mask"] = toks["src_attention_mask"]
            edit_inner["labels"] = self.get_edit_labels(toks["trg_input_ids"])
                
            assert edit_inner["labels"].size(1) < edit_inner["input_ids"].size(1)


            # in this case, rephrase means using propogation questions for L_e
            edit_outer = {}
            edit_outer["input_ids"] = toks["rephrase_input_ids"]
            edit_outer["attention_mask"] = toks["rephrase_attention_mask"]
            edit_outer["labels"] = self.get_edit_labels(toks["trg_input_ids"])
            
            loc = {}
            assert self.use_nq
            batch = [self.nq[idx] for idx in loc_idxs]
            nq_questions = [b[0] for b in batch]
            nq_answers = [b[1] for b in batch]
            nq_answers = [("" if answer[0] == " " else " ") + answer for answer in nq_answers]
            nq_questions = [q + a for (q, a) in zip(nq_questions, nq_answers) ]

            loc = dict(self.tok(nq_questions, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True))
            trg_toks = dict(self.tok(nq_answers, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True, add_special_tokens=False))
            loc["labels"] = self.get_edit_labels(trg_toks["input_ids"])

            if not self.show_first_example:
                LOG.info("Edit_inner:")
                LOG.info("Input: " +  "\n@@\n".join(self.tok.batch_decode(edit_inner["input_ids"])))
                LOG.info("Input: " +  "\n@@\n".join([str(x) for x in edit_inner["input_ids"]]))
                LOG.info("Label:" +  "\n@@\n".join(self.tok.batch_decode(torch.where(edit_inner["labels"] == -100, self.tok.pad_token_id, edit_inner["labels"]))))
                LOG.info("Label: " +  "\n@@\n".join([str(x) for x in torch.where(edit_inner["labels"] == -100, self.tok.pad_token_id, edit_inner["labels"])]))
                LOG.info("\n\n")
                
                LOG.info("Edit_outer:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(edit_outer["input_ids"])))
                LOG.info("Input: " +  "\n@@\n".join([str(x) for x in edit_outer["input_ids"]]))
                LOG.info("Label: " +  "\n@@\n".join(self.tok.batch_decode(torch.where(edit_outer["labels"] == -100, self.tok.pad_token_id, edit_outer["labels"]))))
                LOG.info("Label: " +  "\n@@\n".join([str(x) for x in torch.where(edit_outer["labels"] == -100, self.tok.pad_token_id, edit_outer["labels"])]))
                LOG.info("\n\n")
                
                LOG.info("loc:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(loc["input_ids"])))
                LOG.info("Input: " +  "\n@@\n".join([str(x) for x in loc["input_ids"]]))
                LOG.info("Label: " +  "\n@@\n".join(self.tok.batch_decode(torch.where(loc["labels"] == -100, self.tok.pad_token_id, loc["labels"]))))
                LOG.info("Label: " +  "\n@@\n".join([str(x) for x in torch.where(loc["labels"] == -100, self.tok.pad_token_id, loc["labels"])]))
                # exit(0)
                self.show_first_example = True

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": None,
                "raw": toks["raw"]
            }

            yield dict_to(batch, self.config.device)
