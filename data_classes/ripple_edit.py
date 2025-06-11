import jsonlines
from torch.utils.data import Dataset
import random
from utils import EditBatchSampler, dict_to
import torch
import logging
import typing
import json
from utils import StrEnum, load_jsonlines
import numpy as np
from copy import deepcopy


LOG = logging.getLogger(__name__)


class RippleEditsDataset(Dataset):
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

        assert self.config.heavy_outerloop, "Only heavy outerloop is supported for now."

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
        return len(self.data)

    def __getitem__(self, item, seed=None):
        assert all(
            e in self.data[item]
            for e in [
                "edit",
                "Logical_Generalization",
                "Compositionality_I",
                "Compositionality_II",
                "Subject_Aliasing",
                "Relation_Specificity",
                "Forgetfulness",
            ]
        )

        texts = deepcopy([self.data[item]["edit"]["prompt"]])

        assert self.config.heavy_outerloop
        outerloop_instances = deepcopy(
            self.data[item]["Logical_Generalization"]
            + self.data[item]["Compositionality_I"]
            + self.data[item]["Compositionality_II"]
            + self.data[item]["Subject_Aliasing"]
        )
        locality_instances = deepcopy(self.data[item]["Relation_Specificity"] + self.data[item]["Forgetfulness"])
        outerloop_queries = [q for instance in outerloop_instances for q in instance["test_queries"]]
        locality_queries = [q for instance in locality_instances for q in instance["test_queries"]]

        # ! this is to avoid model exploiting potential heuristics in data order.
        np.random.shuffle(texts)
        np.random.shuffle(outerloop_queries)
        np.random.shuffle(locality_queries)

        output = {
            "texts": texts,
        }
        if len(outerloop_queries) > 0:
            outerloop_queries = [q for q in outerloop_queries if len(q["answers"]) > 0]
            outerloop_queries = [
                q
                for q in outerloop_queries
                if len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0
            ]
            assert len(outerloop_queries) > 0

            outer_questions = [qa["prompt"] for qa in outerloop_queries]
            outer_answers_candidates = [
                [a["value"] for a in q["answers"] if len(a["value"].strip()) > 0] for q in outerloop_queries
            ]
            assert all(len(c) > 0 for c in outer_answers_candidates)
            outer_answers = [np.random.choice(candidates) for candidates in outer_answers_candidates]
            outer_answers = [("" if ans_[0] == " " else " ") + ans_ for ans_ in outer_answers]
            outer_questions = [q_ + ans_ for q_, ans_ in zip(outer_questions, outer_answers)]

            output["outer_questions"] = outer_questions
            output["outer_answers"] = outer_answers

        if len(locality_queries) > 0:
            locality_queries = [q for q in locality_queries if len(q["answers"]) > 0]
            locality_queries = [
                q
                for q in locality_queries
                if len([a["value"] for a in q["answers"] if len(a["value"].strip()) > 0]) > 0
            ]
            assert len(locality_queries) > 0

            loc_questions = [qa["prompt"] for qa in locality_queries]
            loc_answers_candidates = [
                [a["value"] for a in q["answers"] if len(a["value"].strip()) > 0] for q in locality_queries
            ]
            assert all(len(c) > 0 for c in loc_answers_candidates)
            loc_answers = [np.random.choice(candidates) for candidates in loc_answers_candidates]
            loc_answers = [("" if ans_[0] == " " else " ") + ans_ for ans_ in loc_answers]
            loc_questions = [q_ + ans_ for q_, ans_ in zip(loc_questions, loc_answers)]
            if self.config.all_propagation_in_outerloop:
                if len(outerloop_queries) == 0:
                    output["outer_questions"] = []
                    output["outer_answers"] = []

                output["outer_questions"] += loc_questions
                output["outer_answers"] += loc_answers
            else:
                output["loc_questions"] = loc_questions
                output["loc_answers"] = loc_answers
        return output

    def collate_fn(self, batch):
        texts = [s for b in batch for s in b["texts"]]
        all_input_from_batchs = {
            "texts": texts,
        }
        """ 
        ! original line
        trg = (
            [b["answers"][0] for b in batch[:-ne]] +
            [b["alt"] for b in batch[-ne:]]
        )
        """

        if "outer_questions" in batch[0]:
            outer_answers = [s for b in batch for s in b["outer_answers"]]
            outer_questions = [s for b in batch for s in b["outer_questions"]]
            all_input_from_batchs["outer_questions"] = outer_questions
            all_input_from_batchs["outer_answers"] = outer_answers

        if "loc_questions" in batch[0]:
            loc_answers = [s for b in batch for s in b["loc_answers"]]
            loc_questions = [s for b in batch for s in b["loc_questions"]]
            all_input_from_batchs["loc_questions"] = loc_questions
            all_input_from_batchs["loc_answers"] = loc_answers

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
            for k1, v1 in all_input_from_batchs.items()
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
            if "outer_questions" in toks["raw"][0]:
                edit_outer["input_ids"] = toks["outer_questions_input_ids"]
                edit_outer["attention_mask"] = toks["outer_questions_attention_mask"]
                edit_outer["labels"] = self.get_edit_labels(toks["outer_answers_input_ids"])
            else:
                edit_outer = deepcopy(edit_inner)

            loc = {}
            assert self.use_nq
            batch = [self.nq[idx] for idx in loc_idxs]
            nq_questions = [b[0] for b in batch]
            nq_answers = [b[1] for b in batch]
            nq_answers = [("" if answer[0] == " " else " ") + answer for answer in nq_answers]
            nq_questions = [q + a for (q, a) in zip(nq_questions, nq_answers)]
            all_loc_questions = nq_questions
            all_loc_answers = nq_answers

            if "loc_questions" in toks["raw"][0]:
                ripple_edit_loc_questions = [loc_q for ins in toks["raw"] for loc_q in ins["loc_questions"]]
                ripple_edit_loc_answers = [loc_a for ins in toks["raw"] for loc_a in ins["loc_answers"]]
                all_loc_questions += ripple_edit_loc_questions
                all_loc_answers += ripple_edit_loc_answers
            assert len(all_loc_questions) == len(all_loc_answers)
            # shuffle
            zipped = list(zip(all_loc_questions, all_loc_answers))
            random.shuffle(zipped)
            all_loc_questions, all_loc_answers = zip(*zipped)

            loc = dict(
                self.tok(
                    all_loc_questions, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True
                )
            )
            trg_toks = dict(
                self.tok(
                    all_loc_answers,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=False,
                )
            )
            loc["labels"] = self.get_edit_labels(trg_toks["input_ids"])

            if not self.show_first_example:
                LOG.info("is_eval: " + str(self.is_eval))
                LOG.info("Edit_inner:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(edit_inner["input_ids"])))
                LOG.info(
                    "Label:"
                    + "\n@@\n".join(
                        self.tok.batch_decode(
                            torch.where(edit_inner["labels"] == -100, self.tok.pad_token_id, edit_inner["labels"])
                        )
                    )
                )
                LOG.info("\n\n")

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
                LOG.info("\n\n")

                LOG.info("loc:")
                LOG.info("Input: " + "\n@@\n".join(self.tok.batch_decode(loc["input_ids"])))
                LOG.info(
                    "Label: "
                    + "\n@@\n".join(
                        self.tok.batch_decode(torch.where(loc["labels"] == -100, self.tok.pad_token_id, loc["labels"]))
                    )
                )

                self.show_first_example = True

            batch = {"edit_inner": edit_inner, "edit_outer": edit_outer, "loc": loc, "cond": None, "raw": toks["raw"]}

            yield dict_to(batch, self.config.device)
