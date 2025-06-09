import datetime
import typing
import numpy as np
import struct
import os
import getpass
import hydra
import logging
import torch
from collections import defaultdict

import pandas as pd
from losses import multiclass_log_probs
import math
import pdb


LOG = logging.getLogger(__name__)


from enum import Enum
import os
import json
import pickle
import functools
from sqlitedict import SqliteDict


def load_jsonlines(fname: str):
    """Read jsonlines file."""
    with open(fname, "r") as f:
        return [json.loads(line) for line in f]


def dump_jsonlines(obj, fname: str, indent: int = None):
    """Dump jsonlines file."""
    with open(fname, "w", encoding="utf-8") as outfile:
        for entry in obj:
            json.dump(entry, outfile, indent=indent)
            outfile.write("\n")


def load_json(fname: str):
    """Read json file."""
    with open(fname, "r") as f:
        return json.load(f)


def dump_json(obj, fname: str, indent: int = None):
    """Dump json file."""
    with open(fname, "w", encoding="utf-8") as f:
        return json.dump(obj, f, indent=indent)


def load_bin(fname: str):
    """Load binary file."""
    with open(fname, "rb") as f:
        return pickle.load(f)


def dump_bin(obj, fname: str):
    """Dump binary file."""
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


class SQLiteCache:
    """Cache class using sqlite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = SqliteDict(self.db_path, autocommit=True)

    def cache_func(self, func, hash_func=None):
        """Cache wrapper."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if hash_func is not None:
                key = hash_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"
            if key in self.db:
                return self.db[key]
            result = func(*args, **kwargs)
            self.db[key] = result
            return result

        return wrapper

    def close(self):
        """Close the database."""
        self.db.close()


def remove_last_extension(fname):
    return os.path.splitext(fname)[0]


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class EditLoss(StrEnum):
    sft = "sft"

    clm = "clm"


class EditInput(StrEnum):
    question = "question"

    two_doc = "2doc"

    single_doc = "1doc"


def _inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]


def shift_targets(config):
    return "t5" not in config.model.name.lower()


def scr():
    # set this to your shared resources directory --- assuming you have your hf data and models in this directory
    return "/home/zliu/shared_resources" 


def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack("I", os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value


def formatted_timestamp(time=None):
    if time is None:
        time = datetime.datetime.now()
    return time.strftime("%d/%m/%Y-%H:%M:%S/%f")


def time_delta_seconds(start, finish=None):
    assert type(start) == str

    t1 = datetime.datetime.strptime(start, "%d/%m/%Y-%H:%M:%S/%f")
    if finish is not None:
        assert type(finish) == str
        t2 = datetime.datetime.strptime(finish, "%d/%m/%Y-%H:%M:%S/%f")
    else:
        t2 = datetime.datetime.now()

    return (t2 - t1).total_seconds()


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict


def safe_backward(loss, parameters, accumulate=1, allow_unused=False):
    parameters = list(parameters)  # Capture the generator output
    grads = torch.autograd.grad(loss, parameters, allow_unused=allow_unused)
    nan, inf = False, False
    for g in grads:
        if g is not None:
            nan |= g.isnan().any().item()
            inf |= g.isinf().any().item()

    if not (nan or inf):
        for p, g in zip(parameters, grads):
            if g is None:
                continue

            if p.grad is None:
                p.grad = g / accumulate
            else:
                p.grad += g / accumulate
    else:
        LOG.info(f"Skipping grad accumulation because inf: {inf} nan: {nan}")


def _logits(x):
    return x if not hasattr(x, "logits") else x.logits


def load_archive(path):
    import torch

    if not os.path.exists(path):
        # We've not passed an explicit path, but a part of the filename
        wd = hydra.utils.get_original_cwd()
        # directories = ["outputs", "multirun"]
        directories = [
            "outputs",
        ]
        matches = []
        for d in directories:
            search = os.path.join(wd, d)
            for run_dir in os.listdir(search):
                if path in run_dir:
                    matches.append(os.path.join(search, run_dir))
        # pdb.set_trace()
        assert len(matches) == 1, f">1 matches for search {path}; specify exact path"

        full_run_dir = matches[0]
        if "0" in os.listdir(full_run_dir):
            full_run_dir = os.path.join(full_run_dir, "0")
        models_dir = os.path.join(full_run_dir, "models")
        models = os.listdir(models_dir)
        # non_bk = [m for m in models if not m.endswith(".bk")]
        non_bk = [m for m in models if ".bk" not in m]
        assert len(non_bk) == 1, f"Expected a single model in {models_dir}, got {len(non_bk)}"
        path = os.path.join(models_dir, non_bk[0])

    LOG.info(f"Loading checkpoint from {path}")
    archive = torch.load(path, map_location="cpu")
    LOG.info("Load complete.")

    return archive, path


def flatten_dict(d):
    to_process = list(d.items())
    output = {}
    while len(to_process):
        k, v = to_process.pop()
        if isinstance(v, typing.MutableMapping):
            to_process.extend([(f"{k}.{k_}", v_) for (k_, v_) in v.items()])
        else:
            assert k not in output.keys(), "Somehow ended up with duplicate keys"
            output[k] = v

    return output


class EarlyStopper:
    def __init__(self, patience: int, key: str):
        self.best_value = 1e9
        self.best_iter = 0
        self.current_iter = 0
        self.key = key
        self.patience = patience
        self._stop = False

    def update(self, idx, stats):
        assert self.key in stats, f"'{self.key}' not in stats dict"
        value = stats[self.key]
        new_best = value < self.best_value
        if new_best:
            self.best_value = value
            self.best_iter = idx

        self.current_iter = idx
        return new_best

    def should_stop(self):
        self._stop |= self.current_iter - self.best_iter >= self.patience
        return self._stop


class RunningStatAverager:
    def __init__(self, suffix="", exclude=["grad/"], compute_ppl: bool = True):
        self.underlying = None
        self.suffix = suffix
        self.exclude = exclude
        self.compute_ppl = compute_ppl

        self.reset()

    def add(self, d: dict):
        for k, v in d.items():
            if not any([k.startswith(prefix) for prefix in self.exclude]):
                if len(self.suffix):
                    self.underlying[f"{k}_{self.suffix}"].append(v)
                else:
                    self.underlying[k].append(v)

    def average(self):
        average = {}
        for k, v in self.underlying.items():
            if not k.startswith("nll/"):
                average[k] = sum(v) / len(v)
            else:
                assert len(k.split("/")) == 2, f"Invalid key {k}"
                name = k.split("/")[1]
                token_counts = self.underlying[f"n_tokens/{name}"]
                total_nll = sum([nll * c for nll, c in zip(v, token_counts)])
                average[k] = total_nll / sum(token_counts)
                if self.compute_ppl:
                    average[f"perplexity/{name}"] = math.e ** average[k]

        return {k: v if not isinstance(v, torch.Tensor) else v.item() for k, v in average.items()}

    def reset(self):
        self.underlying = defaultdict(list)


class EditBatchSampler:
    def __init__(self, n, n_edits=1, memorize_mode=False, loc_disjoint=True, seed=0):
        self.memorize_mode = memorize_mode
        self.n = n
        self.n_edits = n_edits
        self.loc_disjoint = loc_disjoint
        self.rng = np.random.default_rng(seed)
        self._init()

    def _init(self):
        self.perm = self.rng.permutation(self.n)
        self.edit_position = 0

    def sample(self, batch_size):
        assert batch_size > self.n_edits, "Batch size is interpreted such that batch_size = n_edits + n_loc"

        if self.memorize_mode:
            return list(range(self.n_edits)), list(range(batch_size - self.n_edits))

        if self.edit_position >= self.n:
            self._init()

        edit_idxs = self.perm[self.edit_position : self.edit_position + self.n_edits]
        self.edit_position += self.n_edits

        loc_idxs = self.rng.choice(self.n, batch_size - self.n_edits)
        if self.loc_disjoint:
            while len(np.intersect1d(edit_idxs, loc_idxs)) > 0:
                loc_idxs = self.rng.choice(self.n, batch_size - self.n_edits)

        return edit_idxs.tolist(), loc_idxs.tolist()


def parent_module(model, pname):
    comps = pname.split(".")
    parent = model
    for comp in comps[:-1]:
        if hasattr(parent, comp):
            parent = getattr(parent, comp)
        elif comp.isdigit():
            parent = parent[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    assert hasattr(parent, comps[-1])
    return parent

# Propmend utils

def add_eos(tokenizer_output, eos_token_id, ignore=False):
    
    if ignore:
        return tokenizer_output
    return {
        k: torch.concat(
            [
                v, 
                torch.full(
                    (v.shape[0], 1), # shape of the constant tensor
                    (
                        1 
                        if k == "attention_mask" else
                        eos_token_id # this is to teach the model to end after outputing the answer.
                    )
                )
            ], 
            dim=-1
        )
        for k, v in tokenizer_output.items()
    }


def generate(context: str, answer: str, config, model, tokenizer, generation_config, ):
    inputs = tokenizer([context], return_tensors="pt", padding=True, add_special_tokens=config.add_bos)
    ctx_decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]
    
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    logging.info("Input for generation: " + "["+ "\n\n".join(f"[[{s}]]" for s in tokenizer.batch_decode(inputs["input_ids"])) +"]")
    logging.info("Label for generation: " + "["+ answer +"]")

    
    generation_output = model.generate(
        **inputs,
        generation_config=generation_config,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    generated_texts = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
    generated_texts = [t.replace(ctx_decoded, "") for t in generated_texts]
    model_response_content = []
    for g_i, generated_text in enumerate(generated_texts):
        predicted_answer = generated_text.strip()
        model_response_content.append(
            {
                "question": context, "answer": answer.strip(), 
                "predicted_answer_idx": g_i,
                "predicted_answer": predicted_answer, 
            }
        )
    model_response = pd.DataFrame(model_response_content)
    
    
    return model_response


def get_edit_labels(labels, tokenizer):
    return labels.masked_fill(labels == tokenizer.pad_token_id, -100)


def get_eval_result(question, answer, model, tokenizer, config, generation_config):
    test_queries_str = [question + (" " if answer[0] != " " else "") + answer]

    eos_token_id = tokenizer.eos_token_id
    
    acc_toks = add_eos(tokenizer(test_queries_str, padding=True, return_tensors="pt", add_special_tokens=config.add_bos), eos_token_id, ignore=not config.add_eos_accuracy)
    acc_toks = dict_to(acc_toks, config.device)
    sft_labels = get_edit_labels(
        add_eos(
            tokenizer(
                [
                    (" " if answer[0] != " " else "") + answer
                ], padding=True, return_tensors="pt", add_special_tokens=False), 
            eos_token_id, ignore=not config.add_eos_accuracy
        )["input_ids"], tokenizer
    ).to(config.device)

    clm_labels = get_edit_labels(acc_toks["input_ids"], tokenizer).to(config.device)
    
    logging.info("Input for [Q][A] Accuracy: ")
    logging.info("["+tokenizer.decode(acc_toks["input_ids"][0])+"]")
    logging.info("SFT label: " + "["+tokenizer.decode(sft_labels[0])+"]")
    logging.info("CLM label(before ShiftLeft): " + "["+tokenizer.decode(clm_labels[0])+"]")
    logging.info("")
    
    model.eval()
            
    with torch.no_grad():
        
        model_output = model(
            input_ids=acc_toks["input_ids"],
            attention_mask=acc_toks["attention_mask"]
        )
        if isinstance(model_output, torch.Tensor):
            model_logits = model_output
        else:
            model_logits = model_output.logits
        model_sft_em_dict = multiclass_log_probs(model_logits, sft_labels, exact_match=True)
        model_sft_pm_dict = multiclass_log_probs(model_logits, sft_labels, exact_match=False)
        model_clm_em_dict = multiclass_log_probs(model_logits, clm_labels, exact_match=True)
        model_clm_pm_dict = multiclass_log_probs(model_logits, clm_labels, exact_match=False)
        
        model_result_df = generate(question, answer, config, model, tokenizer, generation_config)
        
    model_result_df.insert(model_result_df.shape[-1], "[A]|[Q] Acc EM", model_sft_em_dict["acc"].item())
    model_result_df.insert(model_result_df.shape[-1], "[A]|[Q] Acc PM", model_sft_pm_dict["acc"].item())
    model_result_df.insert(model_result_df.shape[-1], "[Q][A] Acc EM", model_clm_em_dict["acc"].item())
    model_result_df.insert(model_result_df.shape[-1], "[Q][A] Acc PM", model_clm_pm_dict["acc"].item())
    
    return model_result_df


import numpy as np
from typing import List


def is_significantly_different(
    scores_A: List,
    scores_B: List,
    alpha: float = 0.05,
    n_trial: int = 10000,
    verbose: bool = False,
) -> bool:
    """Determine if the two lists of model performance are significantly
    different from each other by conducting paired bootstrapping test.

    ! Note: `scores_A` and `scores_B` need to be paired; otherwise, the result is not meaningful.

    Args:
        scores_A (List): First list of score.
        scores_B (List): Second list of score.
        alpha (float, optional): threshold for p-value (below which to be significant). Defaults to 0.05.
        n_trial (int, optional): number of bootstrap sampling to conduct. Defaults to 10000.
        verbose (bool, optional): Whether to print some intermediate results. Defaults to False.

    Returns:
        bool: whether scores_A and scores_B are significantly different from each other.
    """
    scores_A = np.array(scores_A)
    scores_B = np.array(scores_B)
    assert len(scores_A) == len(scores_B)

    # Get the inequality direction (or null hypothesis) we want to validate
    # (by calculating the raw average difference).
    # In this context, let's just call it "the ranking".
    scores_A_mean = scores_A.mean()
    scores_B_mean = scores_B.mean()
    delta = scores_B_mean - scores_A_mean

    count = 0
    n_boostrap = len(scores_A)
    for _ in range(n_trial):
        rand_ids = np.random.choice(len(scores_A), size=n_boostrap, replace=True)
        bootstrapped_scores_A = scores_A[rand_ids]
        bootstrapped_scores_B = scores_B[rand_ids]

        # Count how many times that the bootstrapped average *follows* the ranking
        if delta > 0:
            count += bootstrapped_scores_B.mean() > bootstrapped_scores_A.mean()
        else:
            count += bootstrapped_scores_B.mean() < bootstrapped_scores_A.mean()

    # how many times that the randomness (from bootstrap) causes the ranking to be violated.
    p = 1 - count / n_trial
    # if the amount of violation is below the specified threshold,
    # then it's significant difference.
    is_sig_diff = p <= alpha

    if verbose:
        print(f"Score_A avg: {np.round(scores_A_mean, 2)}")
        print(f"Score_B avg: {np.round(scores_B_mean, 2)}")
        print(f"Delta (B - A): {np.round(delta, 1)}")
        print(f"p: {p} (threshold = {alpha})")
        if is_sig_diff:
            print("Significant")
        else:
            print("*Not* Significant")

    return is_sig_diff



if __name__ == "__main__":
    import random

    stopper = EarlyStopper(1000, "loss/edit")

    data = [(100 * idx, {"loss/edit": 2 ** (1 - idx / 10) + random.random()}) for idx in range(100)]

    for d in data:
        stopper.update(*d)
        print(stopper.current_iter, stopper.should_stop(), stopper.best_iter, d[1]["loss/edit"])
