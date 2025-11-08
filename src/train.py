"""Single-run experiment executor (training script).

This file is launched as a **sub-process** by `src.main` so that each run has a
fresh CUDA context and an independent Hydra run-dir.  All configuration comes
from Hydra – the launch command assembled by `src.main` is roughly:

    uv run python -u -m src.train \
        run=<RUN_ID> mode=<full|trial> results_dir=<DIR>

The Hydra *defaults* tree is anchored in `config/`.  `src.train` therefore
carries its *own* @hydra.main so that the sub-process can re-compose the
configuration without receiving Python objects from the parent process.
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 3rd-party deps (heavy ‑ import lazily where possible to keep start-up fast)
# -----------------------------------------------------------------------------
import hydra  # noqa: E402  (hydra must be imported after we set env vars)
from hydra.utils import instantiate  # noqa: E402

# We need HF/PEFT only during model construction, so delay import.  Same for
# datasets.

# -----------------------------------------------------------------------------
# Local modules (data pipeline & model abstractions)
# -----------------------------------------------------------------------------
from src.model import GraffScheduler, ReactorScheduler, create_model  # noqa: E402
from src.preprocess import build_dataloaders  # noqa: E402

# -----------------------------------------------------------------------------
#                                  HELPERS
# -----------------------------------------------------------------------------

def _wandb_init(cfg: DictConfig):  # lazy import
    if cfg.wandb.mode == "disabled":
        # Environment variable respected by WandB itself – no actual init()
        os.environ["WANDB_MODE"] = "disabled"
        return None

    import wandb  # local import – not needed in trial mode

    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run.run_id,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
        mode=cfg.wandb.mode,
    )
    print("[wandb] run url:", run.url)
    return run


def _exact_match(pred: str, ref: str) -> bool:
    # Strip / normalise minor formatting artefacts
    return pred.strip().lower() == ref.strip().lower()


def _evaluate(
    model: torch.nn.Module,
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
    max_gen_tokens: int = 32,
) -> Tuple[float, float]:
    """Run inference on *eval* split & compute EM + loss."""
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward for loss (teacher forcing)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            running_loss += outputs.loss.item() * input_ids.size(0)

            # Autoregressive generation for EM
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_tokens,
            )
            preds = tokenizer.batch_decode(gen_ids[:, input_ids.size(1):], skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for p, r in zip(preds, refs):
                correct += int(_exact_match(p, r))
            total += len(preds)
    em = (correct / max(total, 1)) * 100.0
    loss = running_loss / max(total, 1)
    model.train()
    return em, loss


# -----------------------------------------------------------------------------
#                                TRAINING LOOP
# -----------------------------------------------------------------------------


def _train_single_run(cfg: DictConfig) -> None:
    """Main training routine for *one* run (best hyper-params already baked in)."""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ set-up randomness & device ~~~~~~~~~~~~~~~~~~~~
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ data + model  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tokenizer, train_loader, eval_loader = build_dataloaders(cfg)
    model = create_model(cfg, tokenizer)
    model.to(device)

    # Parameter groups – optimiser for *adapter* weights
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        adapter_params,
        lr=cfg.training.optimizer.lr_init,
        betas=tuple(cfg.training.optimizer.betas),
        weight_decay=cfg.training.optimizer.weight_decay,
    )

    # Optional auxiliary optimiser for scheduler params (GRAFF / REACTOR)
    if cfg.training.scheduler.type == "graff":
        scheduler_mod = GraffScheduler(cfg, model)
        sched_optim = torch.optim.Adam(
            scheduler_mod.parameters(), lr=cfg.training.scheduler.outer_lr
        )
    elif cfg.training.scheduler.type == "reactor":
        scheduler_mod = ReactorScheduler(cfg, model)
        sched_optim = torch.optim.Adam(
            scheduler_mod.parameters(), lr=cfg.training.scheduler.controller_lr
        )
    else:
        scheduler_mod, sched_optim = None, None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ WandB logging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    wb_run = _wandb_init(cfg)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ budget timers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    start_wall = time.time()
    budget_sec = cfg.training.budget_minutes * 60

    step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision in ("fp16", "bf16"))

    for batch in train_loader:
        if cfg.mode == "trial" and step >= 2:
            # Limit batches to keep CI fast – enforced only in trial mode
            break

        cur_wall = time.time()
        if cur_wall - start_wall >= budget_sec:
            print("[budget] wall-clock budget exhausted; stopping training…")
            break

        step += 1
        optimiser.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cuda', enabled=cfg.training.mixed_precision in ("fp16", "bf16")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        # ----------------------  scheduler interaction  ---------------------
        if scheduler_mod is not None:
            # Collect simple probe features: token entropy & grad norm proxy
            with torch.no_grad():
                logits = outputs.logits.float()
                probs = logits.softmax(-1)
                entropy = -(probs * probs.log()).sum(-1).mean().item()
            # Update LR via scheduler module (returns scalar LR multiplier)
            lr_mul = scheduler_mod(entropy=entropy)
            for group in optimiser.param_groups:
                group["lr"] = cfg.training.optimizer.lr_init * lr_mul

        # ---------------- Gradient backward / optimisation ------------------
        scaler.scale(loss).backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(adapter_params, cfg.training.clip_grad_norm)
        scaler.step(optimiser)
        scaler.update()

        if sched_optim is not None:
            sched_optim.step()  # update planner weights

        # -----------------------  metrics & logging -------------------------
        if wb_run is not None:
            wb_run.log({"train_loss": loss.item(), "step": step}, step=step)

        # Periodic evaluation
        if step % cfg.evaluation.interval_steps == 0:
            em, val_loss = _evaluate(model, tokenizer, eval_loader, device)
            if wb_run is not None:
                wb_run.log({"dev_exact_match": em, "dev_loss": val_loss, "step": step}, step=step)
            print(f"[eval] step={step}  dev_EM={em:.2f}  loss={val_loss:.4f}")

        # Safety-cap on total steps
        if step >= cfg.training.total_steps:
            break

    # ------------------------- final evaluation -----------------------------
    em_final, loss_final = _evaluate(model, tokenizer, eval_loader, device)
    if wb_run is not None:
        wb_run.summary["final_dev_em"] = em_final
        wb_run.summary["final_dev_loss"] = loss_final
        wb_run.finish()

    print(f"[done] final dev EM = {em_final:.2f}% – loss = {loss_final:.4f}")


# -----------------------------------------------------------------------------
#                             HYDRA ENTRYPOINT
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D401
    """Entrypoint.  The *effective* cfg already has mode-based overrides."""
    # ^ Hydra changes the working dir to a dedicated run folder; we can keep it.

    # ~~~~~~~~~~~~~~~~~~~~~ load & merge run-specific cfg ~~~~~~~~~~~~~~~~~~~~~~
    repo_root = Path(__file__).resolve().parent.parent
    run_cfg_path = repo_root / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Run-config not found: {run_cfg_path}")
    run_cfg = OmegaConf.load(run_cfg_path)
    # Disable struct mode to allow new keys from run-specific config
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, run_cfg)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ mode-specific knobs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.total_steps = 2  # keep it tiny
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~ optuna hyper-search ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if cfg.optuna.n_trials > 0:
        import optuna  # local import – only when required

        def _objective(trial: optuna.Trial):
            # Suggest hyper-params according to search space declared in cfg
            for hp_name, hp_spec in cfg.optuna.search_space.items():
                if hp_spec.type == "loguniform":
                    val = trial.suggest_float(hp_name, hp_spec.low, hp_spec.high, log=True)
                elif hp_spec.type == "categorical":
                    val = trial.suggest_categorical(hp_name, hp_spec.choices)
                else:
                    raise ValueError("Unsupported hp type: " + str(hp_spec.type))
                OmegaConf.update(cfg, hp_name.replace("_", "."), val, merge=False)
            # Run *extremely* short train for speed (10 steps)
            cfg.training.total_steps = 10
            cfg.wandb.mode = "disabled"
            _train_single_run(cfg)
            return float(OmegaConf.to_container(cfg).get("final_dev_em", 0.0))

        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(_objective, n_trials=cfg.optuna.n_trials)
        print("[optuna] best params:", study.best_params)
        # Update cfg with best params
        for k, v in study.best_params.items():
            OmegaConf.update(cfg, k.replace("_", "."), v, merge=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ final training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _train_single_run(cfg)


if __name__ == "__main__":
    # Make Ctrl-C kill *only* this process, not the whole test runner.
    signal.signal(signal.SIGINT, lambda *_: exit(130))
    main()
