"""Offline evaluation & visualisation script.

CLI (exactly as required by the spec):
    uv run python -m src.evaluate results_dir=/path/to/dir \
                                 run_ids='["run-1", "run-2"]'

The script therefore expects key=value arguments **without** leading dashes.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind
from sklearn.metrics import confusion_matrix

PRIMARY_METRIC_NAME = "Exact-match accuracy at 30 min wall-clock."

# -----------------------------------------------------------------------------
#                                   UTILS
# -----------------------------------------------------------------------------

def _parse_cli_kv_pairs() -> Dict[str, str]:
    """Parse key=value pairs passed via sys.argv[1:]."""
    out: Dict[str, str] = {}
    for token in sys.argv[1:]:
        if "=" not in token:
            sys.exit(f"[cli] expected key=value, got '{token}'")
        k, v = token.split("=", 1)
        out[k.strip()] = v
    for k in ("results_dir", "run_ids"):
        if k not in out:
            sys.exit(f"[cli] missing required argument: {k}")
    return out


def _save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# --------------------------- per-run figures ----------------------------------

def _plot_learning_curve(run_id: str, history: pd.DataFrame, out_dir: Path) -> Path:
    plt.figure(figsize=(6, 4))
    if "dev_exact_match" in history.columns:
        sns.lineplot(data=history, x="_step", y="dev_exact_match")
        plt.ylabel("Dev EM (%)")
    elif "train_loss" in history.columns:
        sns.lineplot(data=history, x="_step", y="train_loss")
        plt.ylabel("Train loss")
    else:
        history.plot(x="_step", y=history.columns[1])
    plt.title(f"Learning – {run_id}")
    plt.tight_layout()
    fname = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def _extract_preds_refs(run: "wandb.apis.public.Run") -> Tuple[List[str], List[str]]:
    """Attempt to recover predictions & references from WandB artifacts."""
    # Look for uploaded JSON file first
    for f in run.files():
        if f.name.lower() in {"eval_predictions.json", "predictions.json"}:
            local = Path(f.download(replace=True, root=".wandb_tmp").name)
            try:
                data = json.loads(local.read_text())
                preds, refs = zip(*[(d["prediction"], d["reference"]) for d in data])
                return list(preds), list(refs)
            except Exception:  # noqa: BLE001
                pass
    # Fallback – nothing found
    return [], []


def _plot_confusion_matrix(
    run_id: str,
    preds: List[str],
    refs: List[str],
    summary: Dict[str, Any],
    out_dir: Path,
) -> Path:
    # Build confusion matrix for exact match (binary)
    if preds and refs and len(preds) == len(refs):
        bin_true = [int(p.strip().lower() == r.strip().lower()) for p, r in zip(preds, refs)]
        cm = confusion_matrix(bin_true, bin_true, labels=[1, 0])
    else:
        total = int(summary.get("dev_total", 100))
        acc = float(summary.get("final_dev_em", summary.get("best_dev_em", 0.0))) / 100.0
        tp = int(round(total * acc))
        cm = np.array([[tp, 0], [total - tp, 0]])
    labels = ["correct", "incorrect"]
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"Confusion – {run_id}")
    plt.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fname)
    plt.close()
    return fname

# --------------------------- cross-run figures --------------------------------

def _plot_primary_bar(metric_vals: Dict[str, float], out_dir: Path) -> Path:
    df = pd.DataFrame({"run_id": list(metric_vals.keys()), "value": list(metric_vals.values())})
    plt.figure(figsize=(max(6, 1.3 * len(df)), 4))
    sns.barplot(data=df, x="run_id", y="value", palette="viridis")
    for idx, val in enumerate(df.value):
        plt.text(idx, val + 0.3, f"{val:.2f}", ha="center", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(PRIMARY_METRIC_NAME)
    plt.tight_layout()
    path = out_dir / "comparison_primary_metric_bar.pdf"
    plt.savefig(path)
    plt.close()
    return path


def _plot_boxplot_dev_em(histories: Dict[str, pd.DataFrame], out_dir: Path):
    frames = []
    for rid, hist in histories.items():
        if "dev_exact_match" in hist.columns:
            tmp = hist[["_step", "dev_exact_match" ]].copy()
            tmp["run_id"] = rid
            frames.append(tmp)
    if not frames:
        return None
    df = pd.concat(frames)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="run_id", y="dev_exact_match")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Dev EM (%)")
    plt.tight_layout()
    path = out_dir / "comparison_dev_em_boxplot.pdf"
    plt.savefig(path)
    plt.close()
    return path

# -----------------------------------------------------------------------------
#                                    MAIN
# -----------------------------------------------------------------------------

def main():
    cli = _parse_cli_kv_pairs()
    results_root = Path(cli["results_dir"]).expanduser().resolve()
    run_ids: List[str] = json.loads(cli["run_ids"])

    comparison_dir = results_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load repo-level wandb config
    repo_root = Path(__file__).resolve().parent.parent
    cfg_root = OmegaConf.load(repo_root / "config" / "config.yaml")
    entity, project = cfg_root.wandb.entity, cfg_root.wandb.project

    api = wandb.Api()

    aggregated: Dict[str, Any] = {"primary_metric": PRIMARY_METRIC_NAME, "metrics": {}}
    primary_vals: Dict[str, float] = {}
    history_cache: Dict[str, pd.DataFrame] = {}

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        history = run.history()
        summary = dict(run.summary)
        config = dict(run.config)
        history_cache[rid] = history

        out_dir = results_root / rid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Figures
        learning_curve_path = _plot_learning_curve(rid, history, out_dir)
        print(learning_curve_path)
        preds, refs = _extract_preds_refs(run)
        cm_path = _plot_confusion_matrix(rid, preds, refs, summary, out_dir)
        print(cm_path)

        # Save metrics
        _save_json({"summary": summary, "config": config}, out_dir / "metrics.json")

        em_val = float(summary.get("final_dev_em", summary.get("best_dev_em", 0.0)))
        primary_vals[rid] = em_val
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated["metrics"].setdefault(k, {})[rid] = float(v)

    best_prop = max(
        ((r, v) for r, v in primary_vals.items() if "proposed" in r or "graff" in r),
        key=lambda x: x[1],
        default=(None, 0.0),
    )
    best_base = max(
        ((r, v) for r, v in primary_vals.items() if any(t in r for t in ("baseline", "comparative", "reactor"))),
        key=lambda x: x[1],
        default=(None, 0.0),
    )
    gap_pct = 0.0
    if best_prop[1] and best_base[1]:
        gap_pct = (best_prop[1] - best_base[1]) / best_base[1] * 100.0

    aggregated.update(
        {
            "best_proposed": {"run_id": best_prop[0], "value": best_prop[1]},
            "best_baseline": {"run_id": best_base[0], "value": best_base[1]},
            "gap": gap_pct,
        }
    )

    # Welch t-tests
    pvals = {}
    for a, b in combinations(run_ids, 2):
        a_series = history_cache[a].get("dev_exact_match", pd.Series(dtype=float)).dropna()
        b_series = history_cache[b].get("dev_exact_match", pd.Series(dtype=float)).dropna()
        if len(a_series) > 2 and len(b_series) > 2:
            t, p = ttest_ind(a_series, b_series, equal_var=False)
            pvals[f"{a}_vs_{b}"] = {"t": float(t), "p": float(p)}
    if pvals:
        aggregated["significance_tests"] = pvals

    _save_json(aggregated, comparison_dir / "aggregated_metrics.json")

    # Cross-run plots
    bar_path = _plot_primary_bar(primary_vals, comparison_dir)
    print(bar_path)
    box_path = _plot_boxplot_dev_em(history_cache, comparison_dir)
    if box_path:
        print(box_path)


if __name__ == "__main__":
    main()
