#!/usr/bin/env python3
"""
Bayesian optimization of the image-preprocessing parameters for YOLO11n-seg.

Pipeline evaluated in every Optuna trial
----------------------------------------
    adaptive gamma correction -> edge enhancement (unsharp masking)
    -> HSV saturation gain -> YOLO11n-seg training -> validation mask mAP@0.5:0.95

Objective (maximized)
---------------------
    J(theta) = mAP@0.5:0.95(theta) - lambda * DeltaE(theta)

where mAP@0.5:0.95 is the mask mAP of the best checkpoint on the validation
subset and DeltaE(theta) is the mean CIEDE2000 colour difference between the
original and the preprocessed images, averaged over all pixels of each image and
then over the first N validation images (sorted by file name).

Protocol (as used for the paper)
--------------------------------
* Sampler: Optuna TPE, seed 42 (10 random start-up trials, then TPE-guided).
* One training run per trial: 60 epochs, batch 16, 640 x 640, seed 42,
  deterministic mode, COCO-pretrained yolo11n-seg.pt.
* Optimizer: the original runs did NOT set ``optimizer``/``lr0`` explicitly, so
  Ultralytics used ``optimizer="auto"``.  For short schedules (< 10,000
  iterations) this resolves to AdamW with lr = 0.002 * 5 / (4 + nc), i.e. about
  0.001429 for nc = 3, and ``lr0`` is ignored.  This is the default here
  (``--optimizer auto``) so that the script reproduces the original behaviour.
  Pass ``--optimizer AdamW --lr0 0.002`` to force an explicit setting instead.
  Always check the ``args.yaml`` / ``results.csv`` of a run to see the value
  that was really applied.
* The training and validation subsets are preprocessed for every trial; the
  test subset is never read.
* Study-level early stopping: stop after ``--early-stop-patience`` consecutive
  trials without improvement in J (default 10).
* Search space:

      gamma_min            U(0.60, 0.80)
      gamma_max            U(1.20, 1.40)
      gamma_target_mean    U(0.45, 0.55)
      edge_blur_sigma      U(0.50, 0.90)
      edge_strength        U(0.02, 0.08)
      hsv_saturation_gain  U(0.98, 1.05)

The study is stored in SQLite and every finished trial writes
``final_result.json``, so an interrupted search can be resumed by running the
same command again.

The preprocessing operators themselves live in ``build_processed_dataset.py``
(``preprocess_image``) and are imported, not duplicated.

Dataset layout (YOLO segmentation format, e.g. a Roboflow export)
-----------------------------------------------------------------
    <data-root>/
        data.yaml
        train/images, train/labels
        valid/images, valid/labels

Usage
-----
    pip install ultralytics optuna scikit-image opencv-python pyyaml tqdm

    python bo_optimization.py --data-root /path/to/Dataset_Roboflow \
                              --output-dir runs/bo_preprocessing

    # quick smoke test (1 trial, 5 epochs)
    python bo_optimization.py --data-root /path/to/Dataset_Roboflow \
                              --output-dir runs/bo_smoke --test-mode

Outputs (in ``--output-dir``)
-----------------------------
    optuna_study.db          Optuna SQLite storage (resumable)
    runs/trial_<n>/          Ultralytics run + final_result.json per trial
    best_parameters.json     parameters and scores of the best trial

Post-hoc analysis (trial table, fANOVA importance with bootstrap, recomputed
Delta-E, figures) is done by ``scripts/analysis/bo_analysis.py``.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import optuna
import yaml
from skimage.color import deltaE_ciede2000, rgb2lab
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
STUDY_NAME = "bo_preprocessing_yolo11_seg"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

SEARCH_SPACE = {
    "gamma_min": (0.60, 0.80),
    "gamma_max": (1.20, 1.40),
    "gamma_target_mean": (0.45, 0.55),
    "edge_blur_sigma": (0.5, 0.9),
    "edge_strength": (0.02, 0.08),
    "hsv_saturation_gain": (0.98, 1.05),
}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Preprocessing (implemented once, in the dataset-preparation script)
# --------------------------------------------------------------------------- #
# The adaptive gamma -> unsharp masking -> HSV saturation pipeline is NOT
# re-implemented here.  It is imported from build_processed_dataset.py, so the
# optimized and the final dataset use exactly the same code.  Keep that file
# importable (next to this script or on PYTHONPATH).
from build_processed_dataset import preprocess_image


def preprocess_pipeline(img_bgr: np.ndarray, params: dict) -> np.ndarray:
    """Apply the shared pipeline with the trial's parameters (dict keys: gamma_min,
    gamma_max, gamma_target_mean, edge_blur_sigma, edge_strength, hsv_saturation_gain)."""
    processed, _ = preprocess_image(img_bgr, params)
    return processed


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def get_image_paths(image_dir: Path) -> list[str]:
    paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(str(image_dir), ext)))
    return sorted(set(paths))


def validate_dataset(data_root: Path) -> None:
    required = [
        data_root / "data.yaml",
        data_root / "train" / "images",
        data_root / "train" / "labels",
        data_root / "valid" / "images",
        data_root / "valid" / "labels",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Dataset path(s) not found:\n  " + "\n  ".join(missing))


def compute_color_shift_penalty(reference_paths: list[str], params: dict) -> float:
    """Mean CIEDE2000 between original and preprocessed images (image-wise mean of
    the per-pixel DeltaE, then mean over images)."""
    deltas = []
    for path in tqdm(reference_paths, desc="Delta-E", unit="img", leave=False):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        processed_bgr = preprocess_pipeline(img_bgr, params)
        original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        delta_e = deltaE_ciede2000(rgb2lab(original_rgb), rgb2lab(processed_rgb))
        deltas.append(float(np.mean(delta_e)))
    return float(np.mean(deltas)) if deltas else 0.0


def build_preprocessed_dataset(args: argparse.Namespace, class_cfg: dict,
                               params: dict, trial_number: int) -> tuple[Path, Path]:
    """Write a preprocessed copy of the train and validation subsets (labels are
    copied unchanged).  The test subset is never touched."""
    trial_dir = args.workdir / f"trial_{trial_number}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True)

    for split in ("train", "valid"):
        src_img = args.data_root / split / "images"
        src_lbl = args.data_root / split / "labels"
        dst_img = trial_dir / split / "images"
        dst_lbl = trial_dir / split / "labels"
        dst_img.mkdir(parents=True)
        dst_lbl.mkdir(parents=True)

        ok = failed = 0
        for image_path in tqdm(get_image_paths(src_img), desc=f"Preprocessing {split}",
                               unit="img", leave=False):
            filename = os.path.basename(image_path)
            stem = os.path.splitext(filename)[0]
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                failed += 1
                continue
            if not cv2.imwrite(str(dst_img / filename), preprocess_pipeline(img_bgr, params)):
                failed += 1
                continue
            label = src_lbl / f"{stem}.txt"
            if label.exists():
                shutil.copy2(label, dst_lbl / label.name)
            ok += 1
        print(f"[{split}] preprocessed: {ok} | failed: {failed}")

    data_cfg = {
        "path": str(trial_dir),
        "train": "train/images",
        "val": "valid/images",
        "names": class_cfg["names"],
    }
    if "nc" in class_cfg:
        data_cfg["nc"] = class_cfg["nc"]
    yaml_path = trial_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)
    return yaml_path, trial_dir


# --------------------------------------------------------------------------- #
# One trial: preprocess -> train -> validate -> penalty
# --------------------------------------------------------------------------- #
def train_and_evaluate(args: argparse.Namespace, class_cfg: dict, reference_paths: list[str],
                       params: dict, trial_number: int) -> tuple[float, float]:
    from ultralytics import YOLO  # imported lazily so that --help works without it

    runs_dir = args.output_dir / "runs"
    trial_name = f"trial_{trial_number}"
    trial_run_dir = runs_dir / trial_name
    last_ckpt = trial_run_dir / "weights" / "last.pt"
    best_ckpt = trial_run_dir / "weights" / "best.pt"
    result_json = trial_run_dir / "final_result.json"

    # Finished trial (e.g. after a restart): reuse the stored result.
    if result_json.exists():
        print(f"Trial {trial_number} already completed - using stored result.")
        with open(result_json) as f:
            res = json.load(f)
        return res["map50_95"], res["color_shift_deltaE"]

    data_yaml, trial_local_dir = build_preprocessed_dataset(args, class_cfg, params, trial_number)

    try:
        if not last_ckpt.exists():
            print(f"Training trial {trial_number}: {args.epochs} epochs, base weights {args.weights}")
            train_kwargs = dict(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                seed=args.seed,
                deterministic=True,
                project=str(runs_dir),
                name=trial_name,
                exist_ok=True,
                save=True,
                save_period=args.save_period,
                verbose=True,
                plots=False,
            )
            # Default "auto" reproduces the original runs (lr0 is ignored by Ultralytics
            # in that case).  An explicit optimizer is only passed when requested.
            if args.optimizer != "auto":
                train_kwargs["optimizer"] = args.optimizer
                train_kwargs["lr0"] = args.lr0
            model = YOLO(args.weights)
            model.train(**train_kwargs)
        else:
            print(f"Resuming interrupted training of trial {trial_number}")
            model = YOLO(str(last_ckpt))
            model.train(resume=True)

        eval_model = YOLO(str(best_ckpt)) if best_ckpt.exists() else model
        metrics = eval_model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split="val",
            verbose=False,
            plots=False,
        )
        map50 = float(metrics.seg.map50)
        map50_95 = float(metrics.seg.map)
        print(f"Validation mask mAP@0.5 = {map50:.6f} | mAP@0.5:0.95 = {map50_95:.6f}")

        color_penalty = compute_color_shift_penalty(reference_paths, params)
        print(f"Delta-E (CIEDE2000) = {color_penalty:.6f}")

        trial_run_dir.mkdir(parents=True, exist_ok=True)
        with open(result_json, "w") as f:
            json.dump({
                "trial_number": trial_number,
                "map50": map50,
                "map50_95": map50_95,
                "color_shift_deltaE": color_penalty,
                "parameters": params,
            }, f, indent=4)
    finally:
        shutil.rmtree(trial_local_dir, ignore_errors=True)
        gc.collect()

    return map50_95, color_penalty


# --------------------------------------------------------------------------- #
# Optuna objective, callbacks and study driver
# --------------------------------------------------------------------------- #
def make_objective(args: argparse.Namespace, class_cfg: dict, reference_paths: list[str]):
    def objective(trial: optuna.Trial) -> float:
        print("\n" + "#" * 78 + f"\nOPTUNA TRIAL {trial.number}\n" + "#" * 78)
        params = {name: trial.suggest_float(name, low, high)
                  for name, (low, high) in SEARCH_SPACE.items()}
        for k, v in params.items():
            print(f"  {k:<22}{v:.6f}")

        map_score, color_penalty = train_and_evaluate(
            args, class_cfg, reference_paths, params, trial.number)

        trial.set_user_attr("map50_95", map_score)
        trial.set_user_attr("color_shift_deltaE", color_penalty)

        score = map_score - args.lambda_color * color_penalty
        print(f"Trial {trial.number}: mAP@0.5:0.95 = {map_score:.6f} | "
              f"Delta-E = {color_penalty:.6f} | J = {score:.6f}")
        return score

    return objective


def make_early_stopping_callback(patience: int):
    state = {"best_value": -np.inf, "no_improve": 0}

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        current_best = study.best_value
        if current_best > state["best_value"] + 1e-6:
            state["best_value"] = current_best
            state["no_improve"] = 0
        else:
            state["no_improve"] += 1
        print(f"Early-stopping counter: {state['no_improve']}/{patience}")
        if state["no_improve"] >= patience:
            print("Stopping the study: no improvement in J for "
                  f"{patience} consecutive trials.")
            study.stop()

    return callback


def run_study(args: argparse.Namespace) -> optuna.Study:
    validate_dataset(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    with open(args.data_root / "data.yaml") as f:
        class_cfg = yaml.safe_load(f)

    train_images = get_image_paths(args.data_root / "train" / "images")
    valid_images = get_image_paths(args.data_root / "valid" / "images")
    reference_paths = valid_images[: args.n_reference_images]
    print(f"Train: {len(train_images)} | Valid: {len(valid_images)} images; "
          f"Delta-E reference images: {len(reference_paths)} (first by file name)")

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.n_startup_trials)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{(args.output_dir / 'optuna_study.db').resolve()}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    n_completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, args.n_trials - n_completed)
    print(f"Target trials: {args.n_trials} | completed: {n_completed} | remaining: {remaining}")

    if remaining > 0:
        study.optimize(
            make_objective(args, class_cfg, reference_paths),
            n_trials=remaining,
            callbacks=[make_early_stopping_callback(args.early_stop_patience)],
            gc_after_trial=True,
        )
    else:
        print("Nothing to run.")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for rank, t in enumerate(sorted(completed, key=lambda x: x.value, reverse=True)[: args.top_n], 1):
        print(f"\nRank {rank} | trial {t.number} | J = {t.value:.6f} | "
              f"mAP@0.5:0.95 = {t.user_attrs.get('map50_95', float('nan')):.6f} | "
              f"Delta-E = {t.user_attrs.get('color_shift_deltaE', float('nan')):.6f}")
        for k, v in t.params.items():
            print(f"    {k}: {v:.6f}")

    if completed:
        best = study.best_trial
        best_result = {
            "best_trial": best.number,
            "combined_score": best.value,
            "map50_95": best.user_attrs.get("map50_95"),
            "color_shift_deltaE": best.user_attrs.get("color_shift_deltaE"),
            "parameters": best.params,
        }
        out = args.output_dir / "best_parameters.json"
        with open(out, "w") as f:
            json.dump(best_result, f, indent=4)
        print("\nBest configuration:\n" + json.dumps(best_result, indent=4))
        print(f"Saved to {out}")
    return study


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bayesian optimization (Optuna TPE) of image-preprocessing parameters "
                    "for YOLO11n-seg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True,
                   help="Dataset root containing data.yaml, train/ and valid/ (test/ is not read).")
    p.add_argument("--output-dir", type=Path, default=Path("runs/bo_preprocessing"),
                   help="Where the Optuna DB, per-trial runs and best_parameters.json are stored.")
    p.add_argument("--workdir", type=Path, default=None,
                   help="Scratch directory for the temporary preprocessed dataset of each trial "
                        "(default: <output-dir>/_tmp_preprocessed). Use a fast local disk.")
    p.add_argument("--weights", default="yolo11n-seg.pt", help="Pretrained base weights.")
    p.add_argument("--n-trials", type=int, default=40,
                   help="Target number of COMPLETE trials (the study may stop earlier).")
    p.add_argument("--epochs", type=int, default=60, help="Epochs per trial.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="CUDA device id(s) or 'cpu'.")
    p.add_argument("--seed", type=int, default=42, help="Seed for training and the TPE sampler.")
    p.add_argument("--save-period", type=int, default=5, help="Checkpoint every N epochs.")
    p.add_argument("--lambda-color", type=float, default=0.01,
                   help="Weight of the CIEDE2000 penalty in J = mAP - lambda * DeltaE.")
    p.add_argument("--n-reference-images", type=int, default=30,
                   help="Number of validation images (first N by file name) for DeltaE.")
    p.add_argument("--n-startup-trials", type=int, default=10,
                   help="Random start-up trials of the TPE sampler (Optuna default = 10).")
    p.add_argument("--early-stop-patience", type=int, default=10,
                   help="Stop the study after this many trials without improvement in J.")
    p.add_argument("--top-n", type=int, default=5, help="Number of top trials printed at the end.")
    p.add_argument("--optimizer", default="auto",
                   help="Ultralytics optimizer. 'auto' reproduces the original runs (AdamW, "
                        "lr ~ 0.001429 for 3 classes, lr0 ignored). Set e.g. AdamW to force "
                        "an explicit optimizer together with --lr0.")
    p.add_argument("--lr0", type=float, default=0.002,
                   help="Initial learning rate; only used when --optimizer is not 'auto'.")
    p.add_argument("--test-mode", action="store_true",
                   help="Smoke test: 1 trial of 5 epochs.")
    args = p.parse_args(argv)

    args.device = int(args.device) if str(args.device).isdigit() else args.device
    if args.workdir is None:
        args.workdir = args.output_dir / "_tmp_preprocessed"
    if args.test_mode:
        args.n_trials, args.epochs = 1, 5
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    run_study(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
