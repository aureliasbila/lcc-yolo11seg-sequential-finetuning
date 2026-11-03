#!/usr/bin/env python3
"""
Ablation study of training strategies for YOLO11n-seg (six configurations x N seeds).

The six configurations compare single-stage training, same-domain fine-tuning,
and cross-domain sequential fine-tuning between raw and preprocessed images:

    ID              Stage 1     Stage 2     Evaluated on
    single_raw      raw         -           raw
    ft_raw_raw      raw         raw         raw
    single_proc     processed   -           processed
    ft_proc_proc    processed   processed   processed
    cross_raw_proc  raw         processed   processed   (Raw to Processed)
    cross_proc_raw  processed   raw         raw         (Processed to Raw)

Training settings (manuscript Table 9):
    Stage 1 / single-stage : 100 epochs, AdamW, lr0 = 0.002
    Stage 2 (fine-tuning)  : 50 epochs,  AdamW, lr0 = 0.0003, freeze = 0
    Image size 640, batch size 16, deterministic=True, seeds 0, 1, 2.

Stage-1 models are shared within a seed, so each seed requires six training
runs (two stage-1 runs and four stage-2 runs) instead of eight.

Important: ``optimizer="AdamW"`` is passed explicitly to every ``YOLO.train()``
call. With ``optimizer="auto"``, Ultralytics prints
"optimizer=auto found, ignoring lr0=..." and the requested learning rate is not
used. After each run, ``args.yaml`` is read back to verify the optimizer and
lr0 that were actually used; the result is stored in ``lr_verification.json``.
A run whose verification fails is not marked as completed.

Model selection in the manuscript is based on the validation subset only, so
the default evaluation split is ``val``.

Usage examples
--------------
    # train seeds 0, 1 and 2 and summarise
    python ablation_study.py --seeds 0 1 2

    # summarise existing weights without training
    python ablation_study.py --mode aggregate

    # Google Colab: copy datasets from Drive to local disk first and back up
    # results to Drive so that interrupted sessions can resume
    python ablation_study.py --seeds 0 1 2 \
        --dataset-source /content/drive/MyDrive/dataset \
        --backup-dir /content/drive/MyDrive/ablation_backup

Expected dataset layout (YOLO segmentation format, one data.yaml per dataset):
    <data-root>/Dataset_Roboflow/data.yaml     (raw images)
    <data-root>/Dataset_Gamma_New/data.yaml    (preprocessed images)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

BASE_MODEL = "yolo11n-seg.pt"

DATASET_FOLDERS = {
    "raw": "Dataset_Roboflow",
    "processed": "Dataset_Gamma_New",
}

STAGE1_CFG = dict(
    epochs=100,
    lr0=0.002,
    optimizer="AdamW",  # must be explicit; "auto" ignores lr0
    imgsz=640,
    batch=16,
)

STAGE2_CFG = dict(
    epochs=50,
    lr0=0.0003,
    optimizer="AdamW",  # must be explicit; "auto" ignores lr0
    freeze=0,
    imgsz=640,
    batch=16,
)

# (config_id, label, stage-1 dataset, stage-2 dataset or None, evaluation dataset)
SCENARIOS = [
    ("single_raw",     "Single-stage (Raw)",             "raw",       None,        "raw"),
    ("ft_raw_raw",     "Fine-tuning (Raw -> Raw)",       "raw",       "raw",       "raw"),
    ("single_proc",    "Single-stage (Processed)",       "processed", None,        "processed"),
    ("ft_proc_proc",   "Fine-tuning (Proc -> Proc)",     "processed", "processed", "processed"),
    ("cross_raw_proc", "Cross-dataset FT (Raw -> Proc)", "raw",       "processed", "processed"),
    ("cross_proc_raw", "Cross-dataset FT (Proc -> Raw)", "processed", "raw",       "raw"),
]

METRIC_KEYS = [
    "mAP50_mask", "mAP50_95_mask", "P_mask", "R_mask",
    "mAP50_box", "mAP50_95_box", "P_box", "R_box",
    "preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "FPS",
]

# Optional: map run folders produced by an earlier version of this script to
# the seed-0 naming scheme. Those runs were trained with optimizer="auto"
# (lr0 ignored) and must NOT be mixed with AdamW runs; disabled by default.
LEGACY_RUN_MAP = {
    "single_stage_raw":          "stage1_raw_s0",
    "single_stage_processed":    "stage1_processed_s0",
    "finetune_stage2_raw":       "stage2_raw_to_raw_s0",
    "finetune_stage2_processed": "stage2_processed_to_processed_s0",
    "cross_raw_to_processed":    "stage2_raw_to_processed_s0",
    "cross_processed_to_raw":    "stage2_processed_to_raw_s0",
}

# Output file names (written inside --output-dir)
RUNS_CSV = "ablation_runs.csv"
RUNS_JSON = "ablation_runs.json"
SUMMARY_CSV = "ablation_summary.csv"
TABLE_MD = "ablation_table.md"
PROGRESS_FILE = "training_progress.json"
LR_VERIFY_FILE = "lr_verification.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_std(values):
    """Return (mean, sample SD) of the non-None values; SD = 0 for one value."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def count_images(folder: Path) -> int:
    return sum(len(list(folder.rglob(ext))) for ext in ("*.jpg", "*.jpeg", "*.png"))


def stage_datasets(source_root: Path, data_root: Path) -> None:
    """Copy dataset folders from a (slow) source such as Google Drive to local disk.

    Folders that already exist locally are skipped. Stale Ultralytics label
    caches (*.cache), which may point to the source paths, are removed.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    for folder in DATASET_FOLDERS.values():
        src, dst = source_root / folder, data_root / folder
        if dst.is_symlink():  # remove a symlink left by a previous session
            dst.unlink()
        if dst.exists():
            print(f"[SKIP] {folder} already present locally ({count_images(dst)} images)")
            continue
        if not src.exists():
            print(f"[ERROR] {src} not found")
            continue
        print(f"[COPY] {folder} ... ", end="", flush=True)
        t0 = time.time()
        shutil.copytree(src, dst)
        print(f"{count_images(dst)} images, {time.time() - t0:.0f} s")

    for cache in data_root.rglob("*.cache"):
        cache.unlink()
        print(f"[CLEAN] removed {cache.name}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AblationStudy:

    def __init__(self, args: argparse.Namespace):
        self.mode = args.mode
        self.seeds = list(args.seeds)
        self.eval_split = args.eval_split
        self.eval_cross_domain = not args.no_cross_domain
        self.migrate_legacy = args.migrate_legacy_runs
        self.data_root = Path(args.data_root).resolve()
        self.output_dir = Path(args.output_dir).resolve()
        self.runs_dir = self.output_dir / "runs"
        self.backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None

        self.datasets: dict[str, str] = {}
        self.runs: list[dict] = []
        self.progress = {"completed_runs": [], "last_update": None}
        self.lr_verify: dict[str, dict] = {}

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Dataset root : {self.data_root}")
        print(f"[INFO] Output dir   : {self.output_dir}")
        print(f"[INFO] Backup dir   : {self.backup_dir or '(none)'}")
        print(f"[INFO] Mode         : {self.mode}")
        print(f"[INFO] Seeds        : {self.seeds if self.seeds else '(no training)'}")
        print(f"[INFO] Eval split   : {self.eval_split}")
        print(f"[INFO] Stage 1      : optimizer={STAGE1_CFG['optimizer']}, "
              f"lr0={STAGE1_CFG['lr0']}, epochs={STAGE1_CFG['epochs']}")
        print(f"[INFO] Stage 2      : optimizer={STAGE2_CFG['optimizer']}, "
              f"lr0={STAGE2_CFG['lr0']}, epochs={STAGE2_CFG['epochs']}, "
              f"freeze={STAGE2_CFG['freeze']}")

    # ---------------- paths ----------------

    def _out(self, name: str) -> Path:
        return self.output_dir / name

    # ---------------- progress ----------------

    def _load_progress(self):
        p = self._out(PROGRESS_FILE)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            if "completed_configs" in data and "completed_runs" not in data:
                data["completed_runs"] = data.pop("completed_configs")
            data.setdefault("completed_runs", [])
            self.progress = data
        except Exception as e:
            print(f"[WARNING] Could not read {p}: {e}")

    def _save_progress(self):
        self.progress["last_update"] = datetime.now().isoformat()
        self._out(PROGRESS_FILE).write_text(json.dumps(self.progress, indent=2))
        self._backup(self._out(PROGRESS_FILE), PROGRESS_FILE)

    def _mark_done(self, run_name):
        if run_name not in self.progress["completed_runs"]:
            self.progress["completed_runs"].append(run_name)
        self._save_progress()

    def _is_done(self, run_name):
        return run_name in self.progress["completed_runs"]

    # ---------------- learning-rate verification ----------------

    def _load_lr_verify(self):
        p = self._out(LR_VERIFY_FILE)
        if p.exists():
            try:
                self.lr_verify = json.loads(p.read_text())
            except Exception:
                self.lr_verify = {}

    def _save_lr_verify(self):
        self._out(LR_VERIFY_FILE).write_text(json.dumps(self.lr_verify, indent=2))
        self._backup(self._out(LR_VERIFY_FILE), LR_VERIFY_FILE)

    @staticmethod
    def verify_lr_from_run(run_dir: Path, expected_optimizer: str, expected_lr0: float) -> dict:
        """Read the Ultralytics ``args.yaml`` of a run and check optimizer and lr0."""
        info = {
            "run_dir": str(run_dir),
            "expected_optimizer": expected_optimizer,
            "expected_lr0": expected_lr0,
            "found_optimizer": None,
            "found_lr0": None,
            "optimizer_ok": False,
            "lr0_ok": False,
            "ok": False,
            "note": "",
        }
        args_yaml = run_dir / "args.yaml"
        if not args_yaml.exists():
            info["note"] = f"args.yaml not found in {run_dir}"
            return info
        try:
            cfg = yaml.safe_load(args_yaml.read_text())
        except Exception as e:
            info["note"] = f"could not read args.yaml: {e}"
            return info

        info["found_optimizer"] = cfg.get("optimizer")
        info["found_lr0"] = cfg.get("lr0")
        notes = []

        found_opt = info["found_optimizer"]
        if isinstance(found_opt, str) and found_opt.lower() == expected_optimizer.lower():
            info["optimizer_ok"] = True
        else:
            notes.append(f"stored optimizer is '{found_opt}', not '{expected_optimizer}' "
                         f"(Ultralytics may have used optimizer=auto)")

        try:
            if info["found_lr0"] is not None and \
                    abs(float(info["found_lr0"]) - float(expected_lr0)) < 1e-9:
                info["lr0_ok"] = True
            else:
                notes.append(f"stored lr0 is {info['found_lr0']}, not {expected_lr0}")
        except Exception as e:
            notes.append(f"could not parse lr0: {e}")

        info["note"] = " | ".join(notes)
        info["ok"] = info["optimizer_ok"] and info["lr0_ok"]
        return info

    # ---------------- backup / restore ----------------

    def _backup(self, local_path: Path, rel_path: str):
        if not self.backup_dir:
            return
        try:
            src, dst = Path(local_path), self.backup_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as e:
            print(f"[WARNING] Backup failed ({local_path}): {e}")

    def restore_from_backup(self):
        """Restore progress, results and run folders from --backup-dir (if any)."""
        self._load_progress()
        self._load_lr_verify()
        if not self.backup_dir or not self.backup_dir.exists():
            print("[INFO] No backup directory -> starting from local state.")
        else:
            print(f"\n[RESTORE] from {self.backup_dir}")
            for fname in (PROGRESS_FILE, RUNS_CSV, RUNS_JSON, LR_VERIFY_FILE):
                f = self.backup_dir / fname
                if f.exists():
                    shutil.copy2(f, self._out(fname))
                    print(f"  [OK] {fname}")
            self._load_progress()
            self._load_lr_verify()

            backup_runs = self.backup_dir / "runs"
            if backup_runs.exists():
                for folder in sorted(backup_runs.iterdir()):
                    if not folder.is_dir():
                        continue
                    dst = self.runs_dir / folder.name
                    try:
                        shutil.copytree(folder, dst, dirs_exist_ok=True)
                        has_best = (dst / "weights" / "best.pt").exists()
                        print(f"  [OK] run {folder.name} (best.pt={has_best})")
                    except Exception as e:
                        print(f"  [WARNING] {folder.name}: {e}")

        runs_json = self._out(RUNS_JSON)
        if runs_json.exists():
            try:
                old = json.loads(runs_json.read_text())
                if isinstance(old, list):
                    self.runs = old
                    print(f"  [OK] loaded {len(self.runs)} previous results.")
            except Exception as e:
                print(f"  [WARNING] Could not load {runs_json}: {e}")

    # ---------------- legacy runs (optional) ----------------

    def migrate_legacy_runs(self):
        if not self.migrate_legacy:
            return
        print("\n[MIGRATE] Looking for legacy runs to map to seed 0 ...")
        n = 0
        for old_name, new_name in LEGACY_RUN_MAP.items():
            old_dir, new_dir = self.runs_dir / old_name, self.runs_dir / new_name
            if (new_dir / "weights" / "best.pt").exists():
                continue
            if not (old_dir / "weights" / "best.pt").exists():
                continue
            try:
                shutil.copytree(old_dir, new_dir, dirs_exist_ok=True)
                self._mark_done(new_name)
                self._backup(new_dir, f"runs/{new_name}")
                print(f"  [OK] {old_name} -> {new_name}")
                n += 1
            except Exception as e:
                print(f"  [WARNING] {old_name}: {e}")
        if n:
            print(f"[MIGRATE] {n} runs mapped to seed 0.")
            print("          WARNING: legacy runs used optimizer=auto (lr0 ignored) and")
            print("          must not be combined with runs that use explicit AdamW + lr0.")
        else:
            print("[MIGRATE] No matching legacy runs found.")

    # ---------------- datasets ----------------

    def find_datasets(self):
        print()
        for key, folder in DATASET_FOLDERS.items():
            local = self.data_root / folder
            if not local.exists():
                print(f"[SKIP] Folder '{folder}' not found in {self.data_root}.")
                continue
            yamls = sorted(local.rglob("data.yaml"))
            if not yamls:
                print(f"[WARNING] No data.yaml found in {local}")
                continue
            self.datasets[key] = str(yamls[0])
            print(f"[OK] dataset '{key}' -> {self.datasets[key]}")

    def check_eval_split(self):
        """Fall back to 'val' if the requested split is missing from a data.yaml."""
        if self.eval_split == "val":
            return
        for key, yaml_path in self.datasets.items():
            try:
                cfg = yaml.safe_load(Path(yaml_path).read_text())
            except Exception as e:
                print(f"[WARNING] Could not read {yaml_path}: {e}")
                continue
            if not cfg.get(self.eval_split):
                print("\n" + "!" * 72)
                print(f"[IMPORTANT] data.yaml of dataset '{key}' has no '{self.eval_split}' key.")
                print("            Evaluation falls back to the validation split.")
                print("!" * 72 + "\n")
                self.eval_split = "val"
                return

    # ---------------- training ----------------

    def _train(self, start_weights, data_yaml, run_name, cfg, seed):
        from ultralytics import YOLO

        run_dir = self.runs_dir / run_name
        best_pt, last_pt = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"

        if self._is_done(run_name) and best_pt.exists():
            print(f"[SKIP] {run_name} already completed.")
            if run_name not in self.lr_verify:
                self.lr_verify[run_name] = self.verify_lr_from_run(
                    run_dir, cfg["optimizer"], cfg["lr0"])
                self._save_lr_verify()
            return str(best_pt)

        if last_pt.exists():
            print(f"[RESUME] {run_name} from the last checkpoint ...")
            try:
                YOLO(str(last_pt)).train(resume=True)
            except Exception as e:
                msg = str(e).lower()
                if any(w in msg for w in ("already", "finished", "complete")):
                    print(f"[INFO] {run_name} had already finished.")
                else:
                    raise
        else:
            print(f"[TRAIN] {run_name} (seed={seed}, epochs={cfg['epochs']}, "
                  f"optimizer={cfg['optimizer']}, lr0={cfg['lr0']})")
            kw = dict(
                data=data_yaml,
                epochs=cfg["epochs"],
                optimizer=cfg["optimizer"],  # required: prevents optimizer=auto
                lr0=cfg["lr0"],
                imgsz=cfg["imgsz"],
                batch=cfg["batch"],
                seed=seed,
                deterministic=True,
                project=str(self.runs_dir),
                name=run_name,
                exist_ok=True,
            )
            if cfg.get("freeze") is not None:
                kw["freeze"] = cfg["freeze"]
            YOLO(start_weights).train(**kw)

        if not best_pt.exists():
            raise FileNotFoundError(f"best.pt not found: {best_pt}")

        v = self.verify_lr_from_run(run_dir, cfg["optimizer"], cfg["lr0"])
        self.lr_verify[run_name] = v
        self._save_lr_verify()
        print(f"  [LR-VERIFY] {run_name}: {'OK' if v['ok'] else 'FAILED'} "
              f"(stored optimizer='{v['found_optimizer']}', stored lr0={v['found_lr0']})")

        self._backup(run_dir, f"runs/{run_name}")
        if v["ok"]:
            self._mark_done(run_name)
        else:
            print(f"  [LR-VERIFY] Details: {v['note']}")
            print("  [LR-VERIFY] The run is NOT marked as completed and should be retrained.")
        return str(best_pt)

    # ---------------- evaluation ----------------

    def _evaluate(self, weights, data_yaml, split=None):
        from ultralytics import YOLO

        split = split or self.eval_split
        m = YOLO(str(weights)).val(data=data_yaml, split=split, imgsz=640, batch=16,
                                   verbose=False)
        pre = m.speed.get("preprocess", 0.0)
        inf = m.speed.get("inference", 0.0)
        post = m.speed.get("postprocess", 0.0)
        total = pre + inf + post
        return {
            "mAP50_mask":    round(float(m.seg.map50), 4),
            "mAP50_95_mask": round(float(m.seg.map), 4),
            "P_mask":        round(float(m.seg.mp), 4),
            "R_mask":        round(float(m.seg.mr), 4),
            "mAP50_box":     round(float(m.box.map50), 4),
            "mAP50_95_box":  round(float(m.box.map), 4),
            "P_box":         round(float(m.box.mp), 4),
            "R_box":         round(float(m.box.mr), 4),
            "preprocess_ms":  round(pre, 2),
            "inference_ms":   round(inf, 2),
            "postprocess_ms": round(post, 2),
            "total_ms":       round(total, 2),
            "FPS":            round(1000 / total, 2) if total > 0 else 0.0,
        }

    # ---------------- one (configuration, seed) pair ----------------

    def run_scenario(self, scenario, seed, train_allowed=True):
        cid, label, ds1, ds2, ds_eval = scenario
        print(f"\n{'=' * 72}\n  {label}   [seed={seed}]\n{'=' * 72}")

        s1_run = f"stage1_{ds1}_s{seed}"
        s1_best = self.runs_dir / s1_run / "weights" / "best.pt"
        if train_allowed:
            s1_weights = self._train(BASE_MODEL, self.datasets[ds1], s1_run, STAGE1_CFG, seed)
        else:
            if not s1_best.exists():
                print(f"[SKIP] {s1_run} not found (aggregate mode, no training).")
                return None
            s1_weights = str(s1_best)
            print(f"[USE] {s1_run}")

        s2_run = f"stage2_{ds1}_to_{ds2}_s{seed}" if ds2 else None
        if ds2 is None:
            final_weights = s1_weights
        elif train_allowed:
            final_weights = self._train(s1_weights, self.datasets[ds2], s2_run, STAGE2_CFG, seed)
        else:
            s2_best = self.runs_dir / s2_run / "weights" / "best.pt"
            if not s2_best.exists():
                print(f"[SKIP] {s2_run} not found (aggregate mode).")
                return None
            final_weights = str(s2_best)
            print(f"[USE] {s2_run}")

        print(f"[EVAL] on dataset '{ds_eval}' (split={self.eval_split})")
        met = self._evaluate(final_weights, self.datasets[ds_eval])

        s1_info = self.lr_verify.get(s1_run, {})
        s2_info = self.lr_verify.get(s2_run, {}) if s2_run else {}

        row = {
            "config_id": cid, "label": label, "seed": seed,
            "stage1": ds1, "stage2": ds2 if ds2 else "-",
            "eval_on": ds_eval, "eval_split": self.eval_split,
            "stage1_optimizer": s1_info.get("found_optimizer"),
            "stage1_lr0":       s1_info.get("found_lr0"),
            "stage1_lr_ok":     s1_info.get("ok"),
            "stage2_optimizer": s2_info.get("found_optimizer"),
            "stage2_lr0":       s2_info.get("found_lr0"),
            "stage2_lr_ok":     s2_info.get("ok"),
            **met,
        }

        print(f"  MASK  mAP50={met['mAP50_mask']:.4f}  mAP50-95={met['mAP50_95_mask']:.4f}  "
              f"P={met['P_mask']:.4f}  R={met['R_mask']:.4f}")
        print(f"  BOX   mAP50={met['mAP50_box']:.4f}  mAP50-95={met['mAP50_95_box']:.4f}")
        print(f"  TIME  {met['total_ms']:.2f} ms (pre {met['preprocess_ms']:.1f} / "
              f"inf {met['inference_ms']:.1f} / post {met['postprocess_ms']:.1f}) "
              f"= {met['FPS']:.1f} FPS")
        if s1_info:
            print(f"  LR    Stage 1: optimizer={s1_info.get('found_optimizer')} "
                  f"lr0={s1_info.get('found_lr0')} ok={s1_info.get('ok')}")
        if s2_info:
            print(f"  LR    Stage 2: optimizer={s2_info.get('found_optimizer')} "
                  f"lr0={s2_info.get('found_lr0')} ok={s2_info.get('ok')}")

        if self.eval_cross_domain:
            other = "processed" if ds_eval == "raw" else "raw"
            if other in self.datasets:
                x = self._evaluate(final_weights, self.datasets[other])
                row["xdomain_eval_on"] = other
                row["xdomain_mAP50_mask"] = x["mAP50_mask"]
                row["xdomain_mAP50_95_mask"] = x["mAP50_95_mask"]
                print(f"  XDOM  mask mAP50-95 on '{other}': {x['mAP50_95_mask']:.4f}")

        return row

    # ---------------- saving ----------------

    def save_runs(self):
        if not self.runs:
            return
        dedup = {(r["config_id"], r["seed"]): r for r in self.runs}
        self.runs = [dedup[k] for k in sorted(dedup)]

        fields = []
        for r in self.runs:
            for k in r:
                if k not in fields:
                    fields.append(k)

        with open(self._out(RUNS_CSV), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.runs:
                w.writerow({k: r.get(k, "") for k in fields})

        self._out(RUNS_JSON).write_text(json.dumps(self.runs, indent=2))
        self._backup(self._out(RUNS_CSV), RUNS_CSV)
        self._backup(self._out(RUNS_JSON), RUNS_JSON)

    # ---------------- mean +/- SD summary ----------------

    def aggregate(self):
        if not self.runs:
            print("\nNo results to summarise yet.")
            return

        by_cfg: dict[str, list] = {}
        for r in self.runs:
            by_cfg.setdefault(r["config_id"], []).append(r)

        summary = []
        for cid, label, ds1, ds2, ds_eval in SCENARIOS:
            rows = by_cfg.get(cid, [])
            if not rows:
                continue
            e = {
                "config_id": cid, "label": label,
                "stage1": ds1, "stage2": ds2 if ds2 else "-", "eval_on": ds_eval,
                "n_seeds": len(rows),
                "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"])),
            }
            for k in METRIC_KEYS:
                mu, sd = mean_std([r.get(k) for r in rows])
                e[f"{k}_mean"] = round(mu, 4) if mu is not None else None
                e[f"{k}_std"] = round(sd, 4) if sd is not None else None
            summary.append(e)

        if summary:
            with open(self._out(SUMMARY_CSV), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                w.writeheader()
                w.writerows(summary)
            self._backup(self._out(SUMMARY_CSV), SUMMARY_CSV)

        self._print_summary(summary)
        self._print_lr_verification()
        self._write_markdown(summary)

    def _print_summary(self, summary):
        width = 104
        print("\n" + "=" * width)
        print("SUMMARY - MASK METRICS (mean +/- SD)".center(width))
        print("=" * width)
        print(f"{'Configuration':<34} {'n':>2}  {'mAP@0.5':>17} {'mAP@0.5:0.95':>17} "
              f"{'P':>17} {'R':>17}")
        print("-" * width)
        for s in summary:
            def fmt(k):
                m, d = s.get(f"{k}_mean"), s.get(f"{k}_std")
                return f"{m:.4f} ± {d:.4f}" if m is not None else "-"
            print(f"{s['label']:<34} {s['n_seeds']:>2}  {fmt('mAP50_mask'):>17} "
                  f"{fmt('mAP50_95_mask'):>17} {fmt('P_mask'):>17} {fmt('R_mask'):>17}")
        print("=" * width)

        if summary and min(s["n_seeds"] for s in summary) < 3:
            print("\n[NOTE] Some configurations have fewer than 3 seeds;")
            print("       the standard deviation cannot yet be interpreted.")

        # Descriptive comparison only; formal tests (ANOVA, Friedman) are run separately.
        if len(summary) >= 2:
            ranked = sorted(summary, key=lambda s: s.get("mAP50_95_mask_mean") or 0, reverse=True)
            best, runner = ranked[0], ranked[1]
            gap = (best["mAP50_95_mask_mean"] or 0) - (runner["mAP50_95_mask_mean"] or 0)
            pooled = ((best.get("mAP50_95_mask_std") or 0) +
                      (runner.get("mAP50_95_mask_std") or 0)) / 2
            print(f"\n[DESCRIPTIVE] Highest mean : {best['label']}")
            print(f"              Gap to 2nd ({runner['label']}): {gap:+.4f}")
            print(f"              Mean SD of the two configurations : {pooled:.4f}")
            if pooled <= 0:
                print("              -> SD not available (at least 2 seeds required).")
            elif gap < pooled:
                print("              -> The gap is smaller than the seed-to-seed variation.")
            elif gap < 2 * pooled:
                print("              -> The gap is only slightly larger than the seed-to-seed variation.")
            else:
                print("              -> The gap is clearly larger than the seed-to-seed variation.")

    def _print_lr_verification(self):
        if not self.lr_verify:
            return
        width = 104
        print("\n" + "=" * width)
        print("LEARNING-RATE VERIFICATION (read from Ultralytics args.yaml)".center(width))
        print("=" * width)
        print(f"{'Run':<34} {'optimizer':<12} {'lr0':<12} {'status':<10} note")
        print("-" * width)
        all_ok = True
        for name in sorted(self.lr_verify):
            v = self.lr_verify[name]
            ok = bool(v.get("ok"))
            all_ok &= ok
            print(f"{name:<34} {str(v.get('found_optimizer')):<12} "
                  f"{str(v.get('found_lr0')):<12} {'OK' if ok else 'FAILED':<10} {v.get('note', '')}")
        print("=" * width)
        if all_ok:
            print("[LR-VERIFY] All runs used the requested optimizer and lr0.")
        else:
            print("[LR-VERIFY] Some runs did not use the requested settings. Check the training")
            print("            log for 'optimizer=auto found, ignoring lr0'; such runs must not")
            print("            be reported.")

    def _write_markdown(self, summary):
        seeds_used = sorted({r["seed"] for r in self.runs})
        lines = [
            "**Validation performance of different training strategies and dataset "
            "configurations**\n",
            f"*Values are mean ± SD over {len(seeds_used)} independent runs "
            f"(seeds {', '.join(map(str, seeds_used))}). All metrics are mask metrics on the "
            f"{self.eval_split} split. All configurations used identical hyperparameters, "
            f"dataset partitions, and pretrained initialization; only the random seed and the "
            f"data condition varied.*\n",
            "| Configuration | Stage 1 | Stage 2 | mAP@0.5 | mAP@0.5:0.95 | P | R |",
            "|---|---|---|---|---|---|---|",
        ]

        best_id = max(summary, key=lambda s: s.get("mAP50_95_mask_mean") or 0)["config_id"] \
            if summary else None
        for s in summary:
            def fmt(k):
                m, d = s.get(f"{k}_mean"), s.get(f"{k}_std")
                return f"{m:.4f} ± {d:.4f}" if m is not None else "—"
            b = "**" if s["config_id"] == best_id else ""
            name = s["label"].split(" (")[0]
            lines.append(
                f"| {b}{name}{b} | {b}{s['stage1'].capitalize()}{b} | "
                f"{b}{s['stage2'].capitalize()}{b} | {b}{fmt('mAP50_mask')}{b} | "
                f"{b}{fmt('mAP50_95_mask')}{b} | {b}{fmt('P_mask')}{b} | {b}{fmt('R_mask')}{b} |")

        ms = [r["total_ms"] for r in self.runs if r.get("total_ms")]
        if ms:
            mu, sd = mean_std(ms)
            lines.append(
                f"\n*Ultralytics validation timing across all runs: {mu:.2f} ± {sd:.2f} ms per "
                f"image (Ultralytics preprocessing, inference, and post-processing at batch size "
                f"16). This excludes the adaptive preprocessing pipeline and is not the latency "
                f"benchmark reported in the manuscript.*")

        text = "\n".join(lines)
        self._out(TABLE_MD).write_text(text, encoding="utf-8")
        self._backup(self._out(TABLE_MD), TABLE_MD)
        print(f"\n[OK] Markdown table written to {self._out(TABLE_MD)}")
        print("\n" + "-" * 72 + "\n" + text + "\n" + "-" * 72)

    # ---------------- orchestration ----------------

    def run(self):
        print("\n" + "=" * 72)
        print("ABLATION STUDY - YOLO11n-seg - 6 CONFIGURATIONS")
        print("=" * 72)

        self.restore_from_backup()
        self.migrate_legacy_runs()
        self.find_datasets()

        if len(self.datasets) < 2:
            print(f"\n[FATAL] Two datasets (raw and processed) are required; found: "
                  f"{list(self.datasets)}")
            for k, v in DATASET_FOLDERS.items():
                print(f"        expected {self.data_root / v / 'data.yaml'}  (for '{k}')")
            return

        self.check_eval_split()

        train_allowed = self.mode == "run"
        seeds = self.seeds if train_allowed else sorted({r["seed"] for r in self.runs} | {0})
        if not seeds:
            seeds, train_allowed = [0], False
            print("\n[INFO] No seeds given -> re-evaluating existing weights only.")

        total, done = len(seeds) * len(SCENARIOS), 0
        for seed in seeds:
            print(f"\n{'#' * 72}\n#  SEED {seed}\n{'#' * 72}")
            for sc in SCENARIOS:
                done += 1
                print(f"\n--- [{done}/{total}] ---")
                try:
                    row = self.run_scenario(sc, seed, train_allowed=train_allowed)
                    if row:
                        self.runs.append(row)
                        self.save_runs()
                except Exception as e:
                    print(f"[ERROR] {sc[0]} seed={seed}: {e}")
                    traceback.print_exc()

        print("\n" + "=" * 72 + "\nDONE\n" + "=" * 72)
        self.save_runs()
        self.aggregate()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Six-configuration, multi-seed ablation study of YOLO11n-seg training "
                    "strategies on raw and preprocessed rice-leaf images.")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2],
                   help="Random seeds to train (default: 0 1 2).")
    p.add_argument("--mode", choices=["run", "aggregate"], default="run",
                   help="'run' = train and summarise; 'aggregate' = summarise existing weights.")
    p.add_argument("--data-root", default="datasets",
                   help="Folder containing the dataset folders (default: ./datasets).")
    p.add_argument("--output-dir", default="runs/ablation",
                   help="Folder for run folders and result files (default: ./runs/ablation).")
    p.add_argument("--dataset-source", default=None,
                   help="Optional source folder (e.g. Google Drive) to copy missing datasets "
                        "from into --data-root before training.")
    p.add_argument("--backup-dir", default=None,
                   help="Optional folder (e.g. on Google Drive) for backup and resume.")
    p.add_argument("--eval-split", choices=["val", "test"], default="val",
                   help="Split used for evaluation (default: val, as in the manuscript).")
    p.add_argument("--no-cross-domain", action="store_true",
                   help="Skip the additional evaluation on the other image domain.")
    p.add_argument("--migrate-legacy-runs", action="store_true",
                   help="Map run folders from an earlier script version to seed 0 "
                        "(those runs used optimizer=auto; not recommended).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dataset_source:
        stage_datasets(Path(args.dataset_source), Path(args.data_root))
    AblationStudy(args).run()


if __name__ == "__main__":
    main()
