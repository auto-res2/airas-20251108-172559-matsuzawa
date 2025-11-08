"""Main orchestrator – spawns `src.train` as a *sub-process* with the appropriate
Hydra overrides derived from CLI arguments.

Usage (mandated by the spec):
    uv run python -u -m src.main run=<run_id> results_dir=<path> mode=<trial|full>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D401
    # ---------------- CLI sanity checks ----------------
    if cfg.run is None:
        sys.exit("[main] parameter 'run' is required.  Example: run=proposed-foo")
    if cfg.mode not in ("trial", "full"):
        sys.exit("[main] 'mode' must be 'trial' or 'full'")
    if cfg.results_dir is None:
        sys.exit("[main] parameter 'results_dir' is required")

    # Determine python executable (works inside uv/virtualenv)
    python_bin = sys.executable

    # Build command – we forward the *same* overrides to the sub-process so that
    # `src.train` can reproduce the config composition itself.
    cmd = [
        python_bin,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"mode={cfg.mode}",
        f"results_dir={cfg.results_dir}",
    ]

    # Spawn child process, inherit stdout/stderr
    print("[main] launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"[main] train sub-process failed with code {proc.returncode}")


if __name__ == "__main__":
    main()
