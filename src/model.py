"""Model construction & LR-planner modules.

This file isolates *all* model-specific logic in one place so that the training
script stays lean.  It supports two schedulers:
    • GRAFF  – differentiable graph-recurrent planner (our method)
    • REACTOR – policy-gradient baseline (simplified)

Both expose a uniform forward(entropy: float) -> lr_multiplier interface used in
`src.train`.  They are intentionally light-weight so that even the CI runner can
execute a few steps.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

_CACHE_DIR = ".cache"

# -----------------------------------------------------------------------------
#                          BASE MODEL + LoRA WRAPPER
# -----------------------------------------------------------------------------

def _apply_lora(model: nn.Module, cfg: DictConfig):
    lora_cfg = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # common for Qwen & GPT-style
    )
    model = get_peft_model(model, lora_cfg)
    return model


def create_model(cfg: DictConfig, tokenizer):
    """Load HF model, apply LoRA, enable gradient checkpointing, return nn.Module."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=_CACHE_DIR,
        torch_dtype=torch.bfloat16 if cfg.model.dtype in ("bf16", "bfloat16") else None,
        device_map="auto",
    )
    model = _apply_lora(model, cfg)

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Tie tokenizer pad token to eos if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model

# -----------------------------------------------------------------------------
#                       GRAFF – differentiable LR planner
# -----------------------------------------------------------------------------

class _SimpleGNN(nn.Module):
    def __init__(self, feat_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.msg = nn.Linear(feat_dim * 2, hidden)
        self.upd = nn.GRUCell(hidden, feat_dim)

    def forward(self, h: torch.Tensor):  # (N, D)
        # Fully connected messages for simplicity
        N = h.size(0)
        src_idx = torch.arange(N, device=h.device).repeat_interleave(N)
        dst_idx = torch.arange(N, device=h.device).repeat(N)
        m = self.msg(torch.cat([h[src_idx], h[dst_idx]], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst_idx, m)
        h = self.upd(agg, h)
        return h


class GraffScheduler(nn.Module):
    """Differentiable planner that outputs a **scalar** LR multiplier ∈ (0, ∞).

    For compute reasons we simplify the design: a per-step latent state h is
    stored as a Parameter (not persisted between batches – fine for demo).
    """

    def __init__(self, cfg: DictConfig, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        # One node per *adapter block* – but we do not have explicit block list,
        # so use a small fixed number.
        num_nodes = 8
        self.h = nn.Parameter(torch.zeros(num_nodes, 8))
        self.gnn = _SimpleGNN(8, hidden=32)
        self.decoder = nn.Linear(8, 1)
        self.register_buffer("running_time", torch.zeros(1))

    def forward(self, entropy: float) -> float:  # noqa: D401  (simple interface)
        # Append features: [entropy, remaining_budget] – replicated per node
        B = self.cfg.training.budget_minutes * 60
        # running_time updated by caller through .running_time += dt if needed
        rem = max(B - float(self.running_time.item()), 0.0)
        rem_norm = rem / B
        feat = torch.tensor([entropy, rem_norm] + [0.0] * 6, device=self.h.device)
        self.h.data += feat  # naive update – demonstration only
        self.h.data = self.gnn(self.h)
        lr_log = self.decoder(self.h.mean(0))
        lr = torch.exp(lr_log).clamp(1e-3, 10.0).item()
        # Edge-of-stability guard (heuristic)
        lr = min(lr, self.cfg.training.scheduler.stability_guard)
        return lr

# -----------------------------------------------------------------------------
#                        REACTOR – policy-gradient baseline
# -----------------------------------------------------------------------------

class ReactorScheduler(nn.Module):
    """Toy REACTOR imitation – outputs a LR multiplier via a tiny MLP.

    Real REACTOR uses REINFORCE; here we skip RL and simply learn via gradient
    descent on loss like a normal module (makes demo tractable).
    """

    def __init__(self, cfg: DictConfig, model: nn.Module):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        self.cfg = cfg

    def forward(self, entropy: float):
        x = torch.tensor([[entropy]], dtype=torch.float32, device=next(self.mlp.parameters()).device)
        lr = torch.exp(self.mlp(x)).clamp(1e-3, 10.0).item()
        return lr
