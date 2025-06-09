import jsonlines
from torch.utils.data import Dataset
import random
from utils import EditBatchSampler, dict_to
import torch
from transformers import BartTokenizerFast, BartTokenizer
import logging
import typing
import json

LOG = logging.getLogger(__name__)


class ZsreDataset(Dataset):
    """
    ! Leo: adding support for running zsre with Decoder-only model

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        size: typing.Optional[int] = None,
        max_length=32,
    ):
        super().__init__()
        self.tok = tokenizer
        self.data = []
        self.config = config

        self.show_first_example = False
        
        def extract(d):
            ex = {k: d[k] for k in ["input", "prediction", "alternatives", "filtered_rephrases", "output"]}
            if ex["input"] in ex["filtered_rephrases"]:
                ex["filtered_rephrases"].remove(ex["input"])
            return ex

        with jsonlines.open(data_path) as f:
            for d in f:
                extracted = extract(d)
                if len(extracted["alternatives"]) > 0 and len(extracted["filtered_rephrases"]) > 0:
                    self.data.append(extracted)
        
        if size is not None:
            self.data = self.data[:size]
            
        self.max_length = max_length
        if self.config.data.zsre_nq: # ! Leo: original if-condition: `and "train" not in data_path`
            self.use_nq = True
            LOG.info("** Using natural questions for zsre base samples **")
            from data_classes.nq import NQDataset
            self.nq = NQDataset(self.config.data.nq_path + ("/train.json" if "train" in data_path else "/validation.json"),
                                tokenizer, config)
        else:
            self.use_nq = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, seed=None):
        new_label = random.choice(self.data[item]["alternatives"])
        rephrase = random.choice(self.data[item]["filtered_rephrases"])
        output = {
            "src": self.data[item]["input"],
            "pred": self.data[item]["prediction"],
            "rephrase": rephrase,
            "alt": new_label,
            "answers": [x["answer"] for x in self.data[item]["output"]],
            "cond": "{} >> {} || {}".format(
                self.data[item]["prediction"],
                new_label,
                self.data[item]["input"],
            ),
        }

        return output

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        
        ne = self.config.data.n_edits
        """ 
        ! original line
        trg = (
            [b["answers"][0] for b in batch[:-ne]] +
            [b["alt"] for b in batch[-ne:]]
        )
        """
        trg = (
            [b["answers"][0] for b in batch[:-ne]] +
            [b["alt"] for b in batch[-ne:]]
        )
        
        
        trg = [("" if len(trg_) != 0 and trg_[0] == " " else " ") + trg_ for trg_ in trg]
        src = [src_ + trg_ for src_, trg_ in zip(src, trg)]
        
        rephrase = [b["rephrase"] for b in batch]
        rephrase = [rephrase_ + trg_ for rephrase_, trg_ in zip(rephrase, trg)]
        

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
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": [b["cond"] for b in batch[-ne:]],
                "rephrase": rephrase[-ne:],
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                add_special_tokens=k1 == "src",
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["raw"] = batch
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
            idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in idxs])

            ne = self.config.data.n_edits
            edit_labels = self.get_edit_labels(toks["trg_input_ids"][-ne:])
            label_length = edit_labels.size(1)

            edit_inner = {}
            edit_inner["input_ids"] = toks["src_input_ids"][-ne:]
            edit_inner["attention_mask"] = toks["src_attention_mask"][-ne:]
            edit_inner["labels"] = edit_labels
            assert label_length <= edit_inner["input_ids"].size(1)

            if self.config.data.rephrase:
                edit_outer = {}
                edit_outer["input_ids"] = toks["rephrase_input_ids"]
                edit_outer["attention_mask"] = toks["rephrase_attention_mask"]
                rephrase_length = edit_outer["input_ids"].size(1)
                # this is bc rephrase_length might be shorter than src_length
                if label_length > rephrase_length:
                    assert (edit_labels[:, :label_length-rephrase_length] == -100).all()
                    edit_labels = edit_labels[:, -rephrase_length:]
                
                edit_outer["labels"] = edit_labels
            else:
                edit_outer = edit_inner

            loc = {}
            if self.use_nq:
                batch = [self.nq[idx] for idx in loc_idxs]
                questions = [b[0] for b in batch]
                answers = [b[1] for b in batch]
                answers = [("" if answer[0] == " " else " ") + answer for answer in answers]
                questions = [q + a for (q, a) in zip(questions, answers) ]
                
                loc = dict(self.tok(questions, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True))
                trg_toks = dict(self.tok(answers, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True, add_special_tokens=False))
                loc["labels"] = self.get_edit_labels(trg_toks["input_ids"])
            else:
                loc["input_ids"] = toks["src_input_ids"][:-ne]
                loc["attention_mask"] = toks["src_attention_mask"][:-ne]
                loc["labels"] = self.get_edit_labels(toks["trg_input_ids"][:-ne])

            cond = {k[5:]: v for k, v in toks.items() if k.startswith("cond")}

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
                "cond": cond,
                "raw": toks["raw"]
            }

            yield dict_to(batch, self.config.device)
