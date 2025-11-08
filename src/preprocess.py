"""Data loading & tokenisation utilities.

Responsible for building PyTorch DataLoaders according to the dataset / model
configuration supplied through Hydra.
"""
from __future__ import annotations

import functools
import os
from typing import Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import DictConfig

_CACHE_DIR = ".cache"  # shared cache for models + datasets

# -----------------------------------------------------------------------------
#                               TOKENISATION
# -----------------------------------------------------------------------------

def _tokenise_function(example, tokenizer: AutoTokenizer, cfg: DictConfig):
    prompt = example.get("question", "")
    # GSM8K answers are inside `answer` key preceded by `#### `
    answer = example.get("answer", "")
    # Build input = question only (teacher forcing on target)
    model_input = tokenizer(prompt, truncation=True, max_length=cfg.dataset.max_tokens)
    with tokenizer.as_target_tokenizer():  # type: ignore[attr-defined]
        label_out = tokenizer(answer, truncation=True, max_length=cfg.dataset.max_tokens)
    model_input["labels"] = label_out["input_ids"]
    return model_input


# -----------------------------------------------------------------------------
#                             DATALOADER BUILDER
# -----------------------------------------------------------------------------

def build_dataloaders(cfg: DictConfig):
    """Return (tokenizer, train_loader, eval_loader)."""
    os.environ.setdefault("HF_HOME", _CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", _CACHE_DIR)
    os.environ.setdefault("HF_DATASETS_CACHE", _CACHE_DIR)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=_CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad token exists

    # ----------------------------- dataset ----------------------------------
    ds_train = load_dataset(
        cfg.dataset.name,
        "main",  # GSM8K requires config name ('main' or 'socratic')
        split=cfg.dataset.train_split,
        streaming=cfg.dataset.streaming,
        cache_dir=_CACHE_DIR,
    )
    ds_eval = load_dataset(
        cfg.dataset.name,
        "main",  # GSM8K requires config name ('main' or 'socratic')
        split=cfg.dataset.eval_split,
        streaming=False,  # eval set is usually small – keep in memory for speed
        cache_dir=_CACHE_DIR,
    )

    tokenise_fn = functools.partial(_tokenise_function, tokenizer=tokenizer, cfg=cfg)
    ds_train = ds_train.map(tokenise_fn, remove_columns=ds_train.features)
    ds_eval = ds_eval.map(tokenise_fn, remove_columns=ds_eval.column_names)

    # ------------------------- torch DataLoader -----------------------------
    def _collate(batch):
        input_ids = [torch.tensor(d["input_ids"]) for d in batch]
        labels = [torch.tensor(d["labels"]) for d in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.training.micro_batch_size,
        collate_fn=_collate,
    )
    eval_loader = DataLoader(
        ds_eval,
        batch_size=cfg.training.micro_batch_size,
        collate_fn=_collate,
    )
    return tokenizer, train_loader, eval_loader
