import jsonlines
from torch.utils.data import Dataset
import random
from utils import EditBatchSampler, dict_to
import torch
from transformers import BartTokenizerFast, BartTokenizer
import logging
import typing
import json
from utils import StrEnum, load_jsonlines
import numpy as np
from copy import deepcopy


LOG = logging.getLogger(__name__)


class SynStoryDataset(Dataset):
    """

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
        is_eval=False,
    ):
        super().__init__()
        self.tok = tokenizer
        self.data = load_jsonlines(data_path)
        self.config = config

        if size is not None:
            self.data = self.data[:size]
        self.show_first_example = False
        self.is_eval = is_eval
        assert self.config.heavy_outerloop, "heavy_outerloop must be used."
        assert self.config.data.rephrase, "propogation question must be used."
        self.max_length = max_length
        if self.config.data.zsre_nq:  # ! Leo: original if-condition: `and "train" not in data_path`
            self.use_nq = True
            LOG.info("** Using natural questions for zsre base samples **")
            from data_classes.nq import NQDataset

            self.nq = NQDataset(
                self.config.data.nq_path + ("/train.json" if "train" in data_path else "/validation.json"),
                tokenizer,
                config,
            )
        else:
            self.use_nq = False

    def __len__(self):
        if self.config.heavy_outerloop or self.is_eval:
            return len(self.data)
        else:
            return len(self.data) * len(self.data[0]["questions"])

    def __getitem__(self, item, seed=None):
        assert all(e in self.data[item] for e in ["text", "questions"])

        texts = deepcopy([self.data[item]["text"]])

        if self.config.heavy_outerloop or self.is_eval:
            qas = deepcopy(self.data[item]["questions"])
        else:
            # randomly sample one from the list of questions
            qas = deepcopy([random.choice(self.data[item]["questions"])])

        # ! this is to avoid model exploiting potential heuristics in data order.
        np.random.shuffle(texts)
        np.random.shuffle(qas)
        answers = [str(qa["answer"]) for qa in qas]
        answers = [("" if len(a) != 0 and a[0] == " " else " ") + a for a in answers]

        questions = [qa["alias_question"] for qa in qas]
        questions = [q_ + ans_ for q_, ans_ in zip(questions, answers)]

        output = {
            "texts": texts,
            "questions": questions,
            "answers": answers,
        }
        return output

    def collate_fn(self, batch):
        texts = [s for b in batch for s in b["texts"]]

        """ 
        ! original line
        trg = (
            [b["answers"][0] for b in batch[:-ne]] +
            [b["alt"] for b in batch[-ne:]]
        )
        """
        answers = [s for b in batch for s in b["answers"]]
        questions = [s for b in batch for s in b["questions"]]

        batches = {
            f"{k1}_{k2}": torch.concat(
                [
                    v2,
                    torch.full(
                        (v2.shape[0], 1),  # shape of the constant tensor
                        (
                            1
                            if k2 == "attention_mask"
                            else self.tok.eos_token_id  # this is to teach the model to end after outputing the answer.
                        ),
                    ),
                ],
                dim=-1,
            )
            for k1, v1 in {
                "texts": texts,
                "questions": questions,
                "answers": answers,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                add_special_tokens="answers" not in k1,  # make the SFT label free of BOS
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
        sampler = EditBatchSampler(
            n, memorize_mode=self.config.single_batch, loc_disjoint=not self.use_nq, seed=self.config.seed
        )

        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            assert len(edit_idxs) == 1
            # idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in edit_idxs])

            # ne = self.config.data.n_edits
            edit_inner = {}
            edit_inner["input_ids"] = toks["texts_input_ids"]
            edit_inner["attention_mask"] = toks["texts_attention_mask"]
            edit_inner["labels"] = self.get_edit_labels(toks["texts_input_ids"])

            assert edit_inner["labels"].size(1) <= edit_inner["input_ids"].size(1)

            # in this case, rephrase means using propogation questions for L_e
            edit_outer = {}
            edit_outer["input_ids"] = toks["questions_input_ids"]
            edit_outer["attention_mask"] = toks["questions_attention_mask"]
            edit_outer["labels"] = self.get_edit_labels(toks["answers_input_ids"])

            loc = {}
            if self.use_nq:
                batch = [self.nq[idx] for idx in loc_idxs]
                questions = [b[0] for b in batch]
                answers = [b[1] for b in batch]
                answers = [("" if answer[0] == " " else " ") + answer for answer in answers]
                questions = [q + a for (q, a) in zip(questions, answers)]

                loc = dict(
                    self.tok(questions, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)
                )
                trg_toks = dict(
                    self.tok(
                        answers,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.max_length,
                        truncation=True,
                        add_special_tokens=False,
                    )
                )
                loc["labels"] = self.get_edit_labels(trg_toks["input_ids"])
            else:
                loc = edit_inner

            if not self.show_first_example:
                LOG.info("is_eval: " + str(self.is_eval))
                LOG.info("Edit_inner:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(edit_inner["input_ids"])))
                LOG.info(
                    "Label: "
                    + "\n@@\n".join(
                        self.tok.batch_decode(
                            torch.where(edit_inner["labels"] == -100, self.tok.pad_token_id, edit_inner["labels"])
                        )
                    )
                )

                LOG.info("Edit_outer:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(edit_outer["input_ids"])))
                LOG.info(
                    "Label: "
                    + "\n@@\n".join(
                        self.tok.batch_decode(
                            torch.where(edit_outer["labels"] == -100, self.tok.pad_token_id, edit_outer["labels"])
                        )
                    )
                )

                LOG.info("loc:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(loc["input_ids"])))
                LOG.info(
                    "Label: "
                    + "\n@@\n".join(
                        self.tok.batch_decode(torch.where(loc["labels"] == -100, self.tok.pad_token_id, loc["labels"]))
                    )
                )

                self.show_first_example = True
            # cond = {k[5:]: v for k, v in toks.items() if k.startswith("cond")}

            batch = {"edit_inner": edit_inner, "edit_outer": edit_outer, "loc": loc, "cond": None, "raw": toks["raw"]}

            yield dict_to(batch, self.config.device)
