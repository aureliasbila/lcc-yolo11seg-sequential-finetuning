#!/usr/bin/env python3
"""
Stage-2 fine-tuning grid search (freeze depth x learning rate) for YOLO11n-seg.

Starting from the Stage-1 weights, the model is fine-tuned on the target-domain
training subset for every combination of

    freeze (number of frozen leading layers)  in {0, 5, 10}
    lr0    (initial learning rate)            in {0.0001, 0.0003, 0.0005}

(Table 13 of the paper).  Each configuration is trained with AdamW, seed 0 and
50 epochs, and evaluated on the VALIDATION subset only.  The configuration with
the highest validation mask mAP@0.5:0.95 is reported as the selected one; the
test subset is not read unless ``--eval-split test`` is passed explicitly.

``optimizer="AdamW"`` is passed explicitly: with ``optimizer="auto"``
Ultralytics ignores ``lr0``.  After every run the stored ``args.yaml`` is read
back and the script aborts the run's bookkeeping if optimizer, lr0 or freeze
differ from what was requested.

Usage
-----
    pip install ultralytics pyyaml pandas

    python finetune_grid.py \
        --stage1-weights  runs/stage1/weights/best.pt \
        --data-yaml       /path/to/target_domain/data.yaml \
        --output-dir      runs/finetune_grid

Outputs (in ``--output-dir``)
-----------------------------
    runs/ft_freeze<F>_lr<LR>/        Ultralytics run directories
    finetune_grid_results.csv        one row per configuration (validation metrics,
                                     GFLOPs, latency, weight size, run name, weight path)
    finetune_grid_best.json          the selected configuration

The script is resumable: configurations whose ``best.pt`` exists and whose
``args.yaml`` passes the check are not retrained.

Notes
-----
* ``--patience`` defaults to 20.  Make sure this matches the value reported in
  the paper and in ``configs/train_args_reference.yaml`` before release.
* ``ms_per_image`` in the CSV is the wall-clock time of the validation call
  divided by the number of images (it includes data loading and metric
  computation); use ``scripts/evaluation/latency_benchmark.py`` for latency.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FREEZE_LIST = [0, 5, 10]
LR_LIST = [0.0001, 0.0003, 0.0005]
OPTIMIZER = "AdamW"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_images(data_yaml: Path, split_key: str) -> int:
    """Number of images in the split referenced by ``data_yaml`` (split_key: train/val/test)."""
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    rel = cfg.get(split_key)
    if rel is None:
        return 0
    folder = Path(rel) if Path(rel).is_absolute() else root / rel
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for f in folder.glob("*") if f.suffix.lower() in exts) if folder.exists() else 0


def verify_run(run_dir: Path, freeze: int, lr0: float) -> tuple[bool, str]:
    """Check optimizer, lr0 and freeze in the Ultralytics args.yaml of a finished run."""
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.exists():
        return False, f"args.yaml not found in {run_dir}"
    cfg = yaml.safe_load(args_yaml.read_text())
    problems = []
    if str(cfg.get("optimizer", "")).lower() != OPTIMIZER.lower():
        problems.append(f"optimizer={cfg.get('optimizer')!r} (expected {OPTIMIZER})")
    if cfg.get("lr0") is None or abs(float(cfg["lr0"]) - lr0) > 1e-12:
        problems.append(f"lr0={cfg.get('lr0')} (expected {lr0})")
    if int(cfg.get("freeze") or 0) != freeze:
        problems.append(f"freeze={cfg.get('freeze')} (expected {freeze})")
    return (not problems), "; ".join(problems) or "ok"


def run_name(freeze: int, lr0: float) -> str:
    return f"ft_freeze{freeze}_lr{lr0}"


def train_one(args: argparse.Namespace, freeze: int, lr0: float, runs_dir: Path) -> Path:
    from ultralytics import YOLO  # lazy import so that --help works without it

    name = run_name(freeze, lr0)
    run_dir = runs_dir / name
    best = run_dir / "weights" / "best.pt"

    if best.exists() and verify_run(run_dir, freeze, lr0)[0]:
        print(f"[SKIP] {name}: already trained and verified")
        return best

    print(f"[TRAIN] {name}: freeze={freeze}, lr0={lr0}, epochs={args.epochs}, "
          f"patience={args.patience}, seed={args.seed}")
    set_seed(args.seed)
    model = YOLO(str(args.stage1_weights))
    model.train(
        data=str(args.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        optimizer=OPTIMIZER,   # required: "auto" would ignore lr0
        lr0=lr0,
        freeze=freeze,
        patience=args.patience,
        project=str(runs_dir),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    ok, note = verify_run(run_dir, freeze, lr0)
    if not ok:
        raise RuntimeError(f"{name}: training settings differ from the request - {note}")
    return best


def evaluate_one(args: argparse.Namespace, weights: Path, split: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    n_images = count_images(args.data_yaml, "test" if split == "test" else "val")
    t0 = time.time()
    metrics = model.val(
        data=str(args.data_yaml),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
        verbose=False,
    )
    total_time = time.time() - t0
    try:
        gflops = float(model.info(verbose=False)[3])
    except Exception:
        gflops = float("nan")
    return {
        "map50": float(metrics.seg.map50),
        "map50_95": float(metrics.seg.map),
        "precision": float(metrics.seg.mp),
        "recall": float(metrics.seg.mr),
        "gflops": gflops,
        "ms_per_image": 1000.0 * total_time / n_images if n_images else float("nan"),
        "total_time_s": total_time,
        "size_mb": weights.stat().st_size / (1024 * 1024),
        "n_images": n_images,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Stage-2 fine-tuning grid search (freeze x lr0) for YOLO11n-seg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage1-weights", type=Path, required=True,
                   help="Stage-1 checkpoint (best.pt) used to initialise every configuration.")
    p.add_argument("--data-yaml", type=Path, required=True,
                   help="Ultralytics data.yaml of the dataset used for fine-tuning "
                        "(train + val; test only if --eval-split test).")
    p.add_argument("--output-dir", type=Path, default=Path("runs/finetune_grid"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stopping patience in epochs (verify against the paper).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="CUDA device id(s) or 'cpu'.")
    p.add_argument("--eval-split", choices=["val", "test"], default="val",
                   help="Split used for the reported metrics. Keep 'val' for model selection.")
    args = p.parse_args(argv)
    args.device = int(args.device) if str(args.device).isdigit() else args.device

    if not args.stage1_weights.exists():
        p.error(f"Stage-1 weights not found: {args.stage1_weights}")
    if not args.data_yaml.exists():
        p.error(f"data.yaml not found: {args.data_yaml}")
    if args.eval_split == "test":
        print("[WARNING] Evaluating on the test split. Do NOT use these numbers to select "
              "the configuration.")

    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for freeze in FREEZE_LIST:
        for lr0 in LR_LIST:
            best = train_one(args, freeze, lr0, runs_dir)
            row = {"freeze_layer": freeze, "lr0": lr0}
            row.update(evaluate_one(args, best, args.eval_split))
            row.update({"run_name": run_name(freeze, lr0), "best_weight_path": str(best)})
            rows.append(row)
            print(f"[RESULT] {row['run_name']}: mAP@0.5 = {row['map50']:.4f} | "
                  f"mAP@0.5:0.95 = {row['map50_95']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "finetune_grid_results.csv"
    df.to_csv(csv_path, index=False)

    best_row = df.loc[df["map50_95"].idxmax()]
    best_json = {
        "selection_split": args.eval_split,
        "criterion": "mask mAP@0.5:0.95",
        "freeze_layer": int(best_row["freeze_layer"]),
        "lr0": float(best_row["lr0"]),
        "map50_95": float(best_row["map50_95"]),
        "run_name": best_row["run_name"],
        "best_weight_path": best_row["best_weight_path"],
    }
    (args.output_dir / "finetune_grid_best.json").write_text(json.dumps(best_json, indent=2))

    print("\n" + df[["freeze_layer", "lr0", "map50", "map50_95", "precision", "recall"]]
          .round(4).to_string(index=False))
    print(f"\nSelected: freeze={best_json['freeze_layer']}, lr0={best_json['lr0']} "
          f"(mAP@0.5:0.95 = {best_json['map50_95']:.4f})")
    print(f"Saved {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
