# ============================================================
# 01_baseline_training.py   (original name: yolo_baseline_train_multiarch_val.py)
#
# Baseline sweep used in Section 4.1 of the paper.
# Models    : YOLOv8n-seg, YOLOv9c-seg, YOLO11n-seg
# LR        : 0.0010, 0.0015, 0.0020
# Epochs    : 100 (early-stopping patience 20)
# Optimizer : AdamW, seed 0
#
# Workflow:
# 1. Train every architecture / learning-rate combination for 100 epochs
# 2. Evaluate the best checkpoint on the VALIDATION set
# 3. Save the results (no test-set evaluation)
#
# Run on Google Colab (GPU). Install the dependency first:  pip install ultralytics
# Some comments and console messages are in Indonesian.
# ============================================================

import os
import time
import random
import csv
import zipfile
import shutil
import numpy as np
import torch

# In Google Colab run this once in a separate cell:
# !pip install ultralytics

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


# ============================================================
# 1. KUNCI SEMUA SUMBER RANDOMNESS
# ============================================================

SEED = 0

def set_full_determinism(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

set_full_determinism(SEED)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATASET_ZIP_PATH = "/content/Dataset_Roboflow.zip"
DATASET_ROOT = "/content/Dataset_Roboflow"
RUN_EXTRACT = True

SPLITS_EXPECTED = ["train", "valid", "test"]
DATA_YAML = "/content/Dataset_Roboflow/data.yaml"

PROJECT_DIR = "/content/runs_baseline_multiarch"

IMG_SIZE = 640
BATCH = 16
WORKERS = 0
DEVICE = 0

# ============================================================
# MODEL (V8, V9, V11)
# ============================================================
# Catatan:
# - YOLOv8: yolo8n-seg.pt (nano)
# - YOLOv9: yolo9c-seg.pt (compact) atau yolo9t-seg.pt (tiny).
#           Jika weight yolo9c-seg tidak tersedia di ultralytics, gunakan yolo9c.pt (detection)
#           atau yolo8n-seg.pt sebagai fallback. (Disini saya gunakan yolo9c-seg.pt)
# - YOLOv11: yolo11n-seg.pt (nano)

ARCHS = {
    "YOLOv8":  "yolov8n-seg.pt",
    "YOLOv9":  "yolov9c-seg.pt",
    "YOLOv11": "yolo11n-seg.pt",
}


# ============================================================
# LEARNING RATE
# ============================================================

LR_SWEEP = [
    0.0010,
    0.0015,
    0.0020
]


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SWEEP_EPOCHS = 100
SWEEP_PATIENCE = 20


# ============================================================
# AUGMENTATION
# ============================================================

AUG_OVERRIDES = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.3,
    fliplr=0.0,
    flipud=0.0,
    mosaic=1.0,
    mixup=0.0,
)


# ============================================================
# UTIL: COUNT NUMBER OF IMAGES
# ============================================================

def count_images(data_yaml_path, split_dir_name, dataset_root=DATASET_ROOT):
    possible_paths = [
        os.path.join(dataset_root, split_dir_name, "images"),
        os.path.join(dataset_root, split_dir_name),
        os.path.join(dataset_root, split_dir_name, "img"),
    ]

    for img_dir in possible_paths:
        if os.path.isdir(img_dir):
            images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                return len(images)

    try:
        import yaml
        with open(data_yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)
        base_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        split_path = data_cfg.get(split_dir_name)
        if split_path is None:
            return 0
        full_path = (split_path if os.path.isabs(split_path) else os.path.join(base_dir, split_path))
        if not os.path.isdir(full_path):
            return 0
        return len([f for f in os.listdir(full_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    except Exception:
        return 0


# ============================================================
# PRINT SEGMENTATION METRICS
# ============================================================

def print_segmentation_metrics(metrics, label="", split_name=""):
    print("\n" + "=" * 60)
    print(f"SEGMENTATION METRICS {label} {split_name}")
    print("=" * 60)

    if hasattr(metrics, "seg"):
        seg = metrics.seg
        map50 = getattr(seg, "map50", 0.0)
        map50_95 = getattr(seg, "map", 0.0)
        precision = getattr(seg, "mp", 0.0)
        recall = getattr(seg, "mr", 0.0)

        print("\nSEGMENTATION METRICS:")
        print(f"mAP50                  : {map50:.4f}")
        print(f"mAP50-95               : {map50_95:.4f}")
        print(f"Precision              : {precision:.4f}")
        print(f"Recall                 : {recall:.4f}")

        if hasattr(seg, "ap_class_index") and hasattr(seg, "ap"):
            print("\nPer-Class mAP50-95:")
            for i, ap in enumerate(seg.ap):
                if i < len(seg.ap_class_index):
                    print(f"Class {seg.ap_class_index[i]}: {ap:.4f}")

        if hasattr(seg, "ap50"):
            print("\nPer-Class mAP50:")
            for i, ap50 in enumerate(seg.ap50):
                if i < len(seg.ap_class_index):
                    print(f"Class {seg.ap_class_index[i]}: {ap50:.4f}")
    else:
        print("WARNING: Segmentation metrics unavailable.")
    print("=" * 60)


# ============================================================
# MODEL EFFICIENCY
# ============================================================

def report_model_efficiency(model, metrics, weight_path, n_images, imgsz=IMG_SIZE, label=""):
    try:
        gflops = get_flops(model.model, imgsz=imgsz)
    except Exception as e:
        print(f"[WARNING] Failed to calculate GFLOPs: {e}")
        gflops = None

    speed = getattr(metrics, "speed", {})
    ms_preprocess = speed.get("preprocess", 0.0)
    ms_inference = speed.get("inference", 0.0)
    ms_postprocess = speed.get("postprocess", 0.0)

    ms_per_image_total = ms_preprocess + ms_inference + ms_postprocess
    total_time_s = (ms_per_image_total * n_images) / 1000.0

    size_mb = (os.path.getsize(weight_path) / (1024 * 1024)) if (weight_path and os.path.exists(weight_path)) else None

    print(f"\n--- MODEL EFFICIENCY {label} ---")
    print(f"GFLOPs                  : {gflops:.2f}" if gflops is not None else "GFLOPs                  : N/A")
    print(f"Preprocess/image        : {ms_preprocess:.2f} ms")
    print(f"Inference/image         : {ms_inference:.2f} ms")
    print(f"Postprocess/image       : {ms_postprocess:.2f} ms")
    print(f"Total/image             : {ms_per_image_total:.2f} ms")
    print(f"Number of images        : {n_images}")
    print(f"Estimated total time    : {total_time_s:.2f} seconds")
    print(f"Model size              : {size_mb:.2f} MB" if size_mb is not None else "Model size              : N/A")

    return {
        "gflops": gflops,
        "ms_per_image": ms_per_image_total,
        "total_time_s": total_time_s,
        "size_mb": size_mb,
        "n_images": n_images,
    }


# ============================================================
# CHECK LAST EPOCH
# ============================================================

def print_last_epoch_reached(run_name, project_dir=PROJECT_DIR, budget=SWEEP_EPOCHS):
    results_csv = os.path.join(project_dir, run_name, "results.csv")
    if not os.path.exists(results_csv):
        print(f"[WARNING] results.csv not found: {run_name}")
        return None

    with open(results_csv, "r") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        return None

    last_epoch = int(float(rows[-1][0]))
    stopped_early = (last_epoch < budget)
    status = "EARLY-STOPPED (possible plateau)" if stopped_early else "FULL BUDGET REACHED"

    print(f"  -> {run_name}: last epoch = {last_epoch}/{budget} [{status}]")
    return last_epoch


# ============================================================
# EXTRACT DATASET ZIP
# ============================================================

def extract_dataset_zip(zip_path=DATASET_ZIP_PATH, extract_to=DATASET_ROOT, splits=SPLITS_EXPECTED, overwrite=False):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    if os.path.exists(extract_to):
        if overwrite:
            shutil.rmtree(extract_to)
        else:
            print(f"[SKIP] {extract_to} already exists.")
            return extract_to

    tmp_extract = extract_to + "_tmp_extract"
    if os.path.exists(tmp_extract):
        shutil.rmtree(tmp_extract)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupt ZIP: {bad_file}")
        zf.extractall(tmp_extract)

    top_items = [i for i in os.listdir(tmp_extract) if not i.startswith("__MACOSX")]
    has_split_at_root = any(s in top_items for s in splits)

    if has_split_at_root:
        os.rename(tmp_extract, extract_to)
    else:
        found = False
        for item in top_items:
            candidate = os.path.join(tmp_extract, item)
            if os.path.isdir(candidate):
                sub_items = os.listdir(candidate)
                if any(s in sub_items for s in splits):
                    shutil.move(candidate, extract_to)
                    found = True
                    break
        shutil.rmtree(tmp_extract, ignore_errors=True)
        if not found:
            raise RuntimeError("Dataset structure not recognized. ZIP must contain train/valid/test.")

    print(f"Dataset extracted -> {extract_to}")
    return extract_to


# ============================================================
# 3. BASELINE TRAINING SWEEP
# ============================================================

def run_sweep():
    results = []

    for arch_name, weight in ARCHS.items():
        for lr0 in LR_SWEEP:
            set_full_determinism(SEED)

            # Nama eksperimen (Membedakan arsitektur dan LR)
            run_name = f"sweep_{arch_name}_lr{lr0:.4f}"

            print("\n" + "=" * 70)
            print(f"BASELINE TRAINING")
            print(f"Model          : {arch_name}")
            print(f"Weight         : {weight}")
            print(f"Learning Rate  : {lr0:.4f}")
            print(f"Epochs         : {SWEEP_EPOCHS}")
            print(f"Run Name       : {run_name}")
            print("=" * 70)

            # Inisialisasi model
            try:
                model = YOLO(weight)
            except Exception as e:
                print(f"[ERROR] Gagal memuat model {weight}: {e}")
                print(f"[SKIP] Melewati {arch_name} LR={lr0}")
                continue  # Lanjut ke LR berikutnya jika model gagal dimuat

            # Training
            model.train(
                data=DATA_YAML,
                epochs=SWEEP_EPOCHS,
                patience=SWEEP_PATIENCE,
                imgsz=IMG_SIZE,
                batch=BATCH,
                workers=WORKERS,
                device=DEVICE,
                project=PROJECT_DIR,
                name=run_name,
                seed=SEED,
                deterministic=True,
                lr0=lr0,
                optimizer="AdamW",
                exist_ok=True,
                **AUG_OVERRIDES,
            )

            print_last_epoch_reached(run_name, budget=SWEEP_EPOCHS)

            # Load Best Checkpoint
            best_weight = os.path.join(PROJECT_DIR, run_name, "weights", "best.pt")
            best_model = YOLO(best_weight)

            # Evaluasi pada Validation Set
            metrics = best_model.val(data=DATA_YAML, split="val")
            print_segmentation_metrics(metrics, label=f"{arch_name} LR={lr0:.4f}", split_name="(Validation)")

            n_val_images = count_images(DATA_YAML, "valid")
            eff = report_model_efficiency(
                best_model, metrics, weight_path=best_weight, n_images=n_val_images, label=f"{arch_name} LR={lr0:.4f}"
            )

            # Simpan metrik ke list
            seg_metrics = getattr(metrics, "seg", None)
            results.append({
                "arch": arch_name,
                "lr0": lr0,
                "map50": getattr(seg_metrics, "map50", 0.0),
                "map50_95": getattr(seg_metrics, "map", 0.0),
                "precision": getattr(seg_metrics, "mp", 0.0),
                "recall": getattr(seg_metrics, "mr", 0.0),
                "gflops": eff["gflops"],
                "ms_per_image": eff["ms_per_image"],
                "total_time_s": eff["total_time_s"],
                "size_mb": eff["size_mb"],
                "n_images": eff["n_images"],
                "run_name": run_name,
                "best_weight_path": best_weight,
            })

    # ========================================================
    # SAVE RESULTS
    # ========================================================
    csv_path = os.path.join(PROJECT_DIR, "Baseline_Sweep_MultiArch_results.csv")
    os.makedirs(PROJECT_DIR, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "arch", "lr0", "map50", "map50_95", "precision", "recall",
            "gflops", "ms_per_image", "total_time_s", "size_mb", "n_images",
            "run_name", "best_weight_path",
        ])
        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================
    print("\nRESULT SUMMARY (VALIDATION SET)")
    print(f"{'Model':<12}{'LR':<10}{'mAP50':<10}{'mAP50-95':<12}{'Precision':<12}{'Recall':<12}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x["map50_95"], reverse=True):
        print(f"{r['arch']:<12}{r['lr0']:<10.4f}{r['map50']:<10.4f}{r['map50_95']:<12.4f}{r['precision']:<12.4f}{r['recall']:<12.4f}")

    print(f"\nResults saved -> {csv_path}")
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("YOLOv8, YOLOv9, YOLOv11 BASELINE EXPERIMENT (VAL SET ONLY)")
    print("=" * 70)
    print("Models        : YOLOv8-seg, YOLOv9-seg, YOLOv11-seg")
    print("Learning Rate : 0.0010, 0.0015, 0.0020")
    print("Epochs        : 100")
    print("Optimizer     : AdamW")
    print("Seed          : 0")
    print("=" * 70)

    if RUN_EXTRACT:
        extract_dataset_zip(DATASET_ZIP_PATH, DATASET_ROOT, SPLITS_EXPECTED, overwrite=False)

    # Jalankan proses sweep (training & evaluasi val)
    sweep_results = run_sweep()

    # ========================================================
    # PILIH CHECKPOINT TERBAIK KESELURUHAN
    # ========================================================
    if sweep_results:
        best = max(sweep_results, key=lambda x: x["map50_95"])

        print("\n" + "=" * 70)
        print("BEST VALIDATION CHECKPOINT OVERALL")
        print("=" * 70)
        print(f"Model         : {best['arch']}")
        print(f"Learning Rate : {best['lr0']:.4f}")
        print(f"mAP50         : {best['map50']:.4f}")
        print(f"mAP50-95      : {best['map50_95']:.4f}")
        print(f"Precision     : {best['precision']:.4f}")
        print(f"Recall        : {best['recall']:.4f}")
        print(f"Best weight   : {best['best_weight_path']}")
        print("=" * 70)
    else:
        print("\n[WARNING] Tidak ada model yang berhasil ditraining.")
