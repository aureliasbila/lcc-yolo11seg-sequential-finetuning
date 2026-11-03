# NOTE: some comments and console messages in this script are written in Indonesian.
# Edit the CONFIG block (paths) before running. Designed for Google Colab with Google Drive mounted.
# -*- coding: utf-8 -*-
"""
latency_benchmark_4_7.py  -  Section 4.7 Computational Efficiency (R2.6)
=======================================================================
Mengukur Params, GFLOPs, waktu adaptive preprocessing, preprocess / inference /
postprocess Ultralytics, total end-to-end (ms) dan FPS. Batch 1, imgsz 640.

PRINSIP INPUT:
  * SEMUA model diberi input CITRA RAW/ORIGINAL (internal test set), karena di
    lapangan kamera menghasilkan citra raw.
  * Model yang tahap akhirnya dilatih pada data processed (Single Processed,
    Proc->Proc, Raw->Proc) WAJIB menjalankan adaptive preprocessing (gamma + edge
    + HSV) lebih dulu  -> waktunya ikut dihitung di kolom adapt_ms.
  * File di Dataset_Gamma_New TIDAK dipakai sebagai input timing (itu akan
    menyembunyikan biaya preprocessing); hanya dipakai untuk cek bahwa fungsi
    preprocessing di sini identik dengan yang membangun dataset processed.

Jalankan di Google Colab (Runtime GPU). Hasil disimpan ke Drive:
  MyDrive/ablation_backup/latency/
"""
import json, os, platform, subprocess, time, warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import ultralytics

warnings.filterwarnings("ignore")

# ==========================================================================
#  KONFIGURASI - EDIT BAGIAN INI
# ==========================================================================
DRIVE       = Path("/content/drive/MyDrive")
DATASET_DIR = DRIVE / "dataset"
RAW_DS      = DATASET_DIR / "Dataset_Roboflow"       # dataset raw (input timing)
PROC_DS     = DATASET_DIR / "Dataset_Gamma_New"      # hanya untuk cek kesamaan pipeline
SPLIT_NAMES = ["test"]                               # internal test set (67 citra)
RUNS_ROOTS  = [Path("/content/runs/ablation"), DRIVE / "ablation_backup" / "runs"]
OUTPUT_DIR  = DRIVE / "ablation_backup" / "latency"

# Bobot baseline (Section 4.1). Isi path yang benar; yang tidak ada akan dilewati.
BASELINE_WEIGHTS = {
    "YOLOv8-seg (baseline)": DRIVE / "ablation_backup" / "baseline" / "Baseline8_best.pt",
    "YOLOv9-seg (baseline)": DRIVE / "ablation_backup" / "baseline" / "Baseline9_best.pt",
    "YOLO11n-seg (baseline)": DRIVE / "ablation_backup" / "baseline" / "Baseline_best.pt",
}

# Konfigurasi ablation: (config_id, label, stage1, stage2, domain input akhir)
SCENARIOS = [
    ("single_raw",     "Single-stage Raw",       "raw",       None,        "raw"),
    ("ft_raw_raw",     "Raw → Raw",              "raw",       "raw",       "raw"),
    ("single_proc",    "Single-stage Processed", "processed", None,        "processed"),
    ("ft_proc_proc",   "Processed → Processed",  "processed", "processed", "processed"),
    ("cross_raw_proc", "Raw → Processed",        "raw",       "processed", "processed"),
    ("cross_proc_raw", "Processed → Raw",        "processed", "raw",       "raw"),
]
TIMING_SEED = 0       # arsitektur identik antar-seed -> cukup satu seed per konfigurasi

# Parameter adaptive preprocessing terpilih (Table 5)
PREP = dict(gamma_min=0.7825, gamma_max=1.2026, gamma_target_mean=0.4835,
            edge_blur_sigma=0.6074, edge_strength=0.0469, hsv_saturation_gain=1.0171)

IMGSZ, CONF, NMS_IOU = 640, 0.25, 0.45
WARMUP     = 20           # iterasi warm-up per model (tidak dicatat)
REPEATS    = {"cuda": 3, "cpu": 1}       # berapa kali seluruh citra diulang
MAX_IMAGES = {"cuda": None, "cpu": 30}   # None = semua citra test
DEVICES    = ["cuda", "cpu"]             # hapus "cpu" bila tidak perlu
USE_HALF   = False                       # FP32 agar sebanding dengan training/eval
# ==========================================================================


def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print(f"[INFO] Drive tidak di-mount ({e})")


def find_split(ds_root):
    for n in SPLIT_NAMES:
        if (ds_root / n / "images").exists():
            return ds_root / n
    for p in ds_root.rglob("images"):
        if p.parent.name in SPLIT_NAMES:
            return p.parent
    raise FileNotFoundError(f"Folder {SPLIT_NAMES} tidak ditemukan di {ds_root}")


def list_images(split_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in exts)


def resolve_weights(ds1, ds2, seed):
    run = f"stage1_{ds1}_s{seed}" if ds2 is None else f"stage2_{ds1}_to_{ds2}_s{seed}"
    for root in RUNS_ROOTS:
        w = root / run / "weights" / "best.pt"
        if w.exists():
            return w
    return None


# ---- adaptive preprocessing (SAMA dengan pipeline BO; ganti bila fungsi asli Anda berbeda)
def adaptive_preprocess(img_bgr, p=PREP):
    x = img_bgr.astype(np.float32) / 255.0
    g = np.clip(np.log(p["gamma_target_mean"]) / np.log(max(float(x.mean()), 1e-6)),
                p["gamma_min"], p["gamma_max"])
    x = np.clip(np.power(x, g) * 255, 0, 255).astype(np.uint8)
    bl = cv2.GaussianBlur(x, (0, 0), sigmaX=p["edge_blur_sigma"])
    x = np.clip(cv2.addWeighted(x, 1 + p["edge_strength"], bl, -p["edge_strength"], 0),
                0, 255).astype(np.uint8)
    h = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 1] = np.clip(h[:, :, 1] * p["hsv_saturation_gain"], 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)


def check_pipeline(raw_paths, n=10):
    """Bandingkan output adaptive_preprocess() dengan file Dataset_Gamma_New."""
    try:
        proc_dir = find_split(PROC_DS) / "images"
    except FileNotFoundError:
        print("[WARNING] Dataset processed tidak ditemukan -> cek pipeline dilewati")
        return None
    proc = {p.stem: p for p in proc_dir.iterdir()}
    maes = []
    for rp in raw_paths[:n]:
        pp = proc.get(rp.stem)
        if pp is None:
            continue
        a = adaptive_preprocess(cv2.imread(str(rp))).astype(np.float32)
        b = cv2.imread(str(pp)).astype(np.float32)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        maes.append(float(np.abs(a - b).mean()))
    if not maes:
        print("[WARNING] Nama file raw/processed tidak cocok -> cek pipeline dilewati")
        return None
    m = float(np.mean(maes))
    print(f"[CHECK] MAE pipeline vs Dataset_Gamma_New ({len(maes)} citra): {m:.2f} (0-255)")
    if m > 2.0:
        print("  [WARNING] Selisih besar: fungsi adaptive_preprocess() mungkin tidak sama dengan "
              "pipeline asli Anda (atau file processed telah di-resize/kompresi JPEG ulang).")
    return m


def model_complexity(weights):
    m = YOLO(str(weights))
    params = sum(p.numel() for p in m.model.parameters()) / 1e6
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = float(get_flops(m.model, IMGSZ))
    except Exception:
        info = m.info(verbose=False)
        gflops = float(info[-1]) if info else float("nan")
    return params, gflops


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def benchmark(weights, domain, images, dev):
    model = YOLO(str(weights))
    kw = dict(imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, device=0 if dev == "cuda" else "cpu",
              half=USE_HALF and dev == "cuda", verbose=False)
    # warm-up
    for k in range(WARMUP):
        im = images[k % len(images)]
        model.predict(adaptive_preprocess(im) if domain == "processed" else im, **kw)
    rows = []
    for rep in range(REPEATS[dev]):
        for i, im in enumerate(images):
            sync(dev); t0 = time.perf_counter()
            x = adaptive_preprocess(im) if domain == "processed" else im
            t1 = time.perf_counter()
            r = model.predict(x, **kw)[0]
            sync(dev); t2 = time.perf_counter()
            rows.append(dict(rep=rep, img=i,
                             adapt_ms=(t1 - t0) * 1e3,
                             pre_ms=r.speed["preprocess"], inf_ms=r.speed["inference"],
                             post_ms=r.speed["postprocess"],
                             e2e_ms=(t2 - t0) * 1e3))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def hardware_info():
    info = dict(python=platform.python_version(), torch=torch.__version__,
                ultralytics=ultralytics.__version__, opencv=cv2.__version__)
    try:
        cpu = [l for l in open("/proc/cpuinfo") if "model name" in l][0].split(":")[1].strip()
    except Exception:
        cpu = platform.processor()
    info["cpu"] = f"{cpu} ({os.cpu_count()} threads)"
    try:
        ram = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]
        info["ram_gb"] = round(int(ram) / 1024 ** 2, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        try:
            info["gpu_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().strip()
        except Exception:
            pass
    return info


def main():
    mount_drive()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hw = hardware_info()
    print(json.dumps(hw, indent=2))
    devices = [d for d in DEVICES if d != "cuda" or torch.cuda.is_available()]
    if "cuda" not in devices:
        print("[WARNING] GPU tidak tersedia -> hanya CPU (Runtime > Change runtime type > GPU)")

    raw_paths = list_images(find_split(RAW_DS))
    print(f"[OK] {len(raw_paths)} citra raw dari {find_split(RAW_DS)}")
    pipe_mae = check_pipeline(raw_paths)
    raw_imgs = [cv2.imread(str(p)) for p in raw_paths]      # dimuat ke RAM: I/O disk tidak dihitung
    sizes = pd.Series([f"{im.shape[1]}x{im.shape[0]}" for im in raw_imgs]).value_counts()
    print("[INFO] Resolusi citra input:\n" + sizes.to_string())

    models = [(name, Path(w), "raw", "baseline") for name, w in BASELINE_WEIGHTS.items()]
    for cid, label, d1, d2, dom in SCENARIOS:
        w = resolve_weights(d1, d2, TIMING_SEED)
        models.append((label, w, dom, cid))

    per_img, summary = [], []
    for name, w, dom, tag in models:
        if w is None or not Path(w).exists():
            print(f"[SKIP] {name}: bobot tidak ditemukan ({w})")
            continue
        params, gflops = model_complexity(w)
        for dev in devices:
            imgs = raw_imgs[:MAX_IMAGES[dev]] if MAX_IMAGES[dev] else raw_imgs
            print(f"[RUN] {name:<26} {dev:<4} input={dom:<9} n={len(imgs)}x{REPEATS[dev]}")
            df = benchmark(w, dom, imgs, dev)
            df.insert(0, "device", dev); df.insert(0, "model", name)
            per_img.append(df)
            s = dict(model=name, tag=tag, input=dom, device=dev, weights=str(w),
                     params_M=round(params, 2), GFLOPs=round(gflops, 2), n=len(df))
            for c in ["adapt_ms", "pre_ms", "inf_ms", "post_ms", "e2e_ms"]:
                s[c + "_mean"] = df[c].mean(); s[c + "_sd"] = df[c].std(ddof=1)
            s["e2e_ms_median"] = df.e2e_ms.median()
            s["e2e_ms_p95"] = df.e2e_ms.quantile(0.95)
            s["FPS"] = 1000.0 / df.e2e_ms.mean()
            summary.append(s)

    per_img = pd.concat(per_img, ignore_index=True)
    summary = pd.DataFrame(summary)
    per_img.to_csv(OUTPUT_DIR / "latency_per_image.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "latency_summary.csv", index=False)
    hw["pipeline_mae_vs_processed_dataset"] = pipe_mae
    hw.update(imgsz=IMGSZ, conf=CONF, nms_iou=NMS_IOU, batch=1, warmup=WARMUP,
              repeats=REPEATS, fp16=USE_HALF, timing_seed=TIMING_SEED)
    json.dump(hw, open(OUTPUT_DIR / "hardware_and_settings.json", "w"), indent=2)

    # ---- tabel siap tempel (Table [X] Section 4.7)
    f = lambda m, s: f"{m:.2f} ± {s:.2f}"
    lines = []
    for dev in devices:
        g = summary[summary.device == dev]
        lines += [f"\n### Device: {dev.upper()} ({hw.get('gpu') if dev == 'cuda' else hw['cpu']})\n",
                  "| Model / configuration | Input | Params (M) | GFLOPs | Adaptive prep. (ms) | "
                  "Pre (ms) | Inference (ms) | Post (ms) | End-to-end (ms) | FPS |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            adapt = f(r.adapt_ms_mean, r.adapt_ms_sd) if r.input == "processed" else "–"
            lines.append(f"| {r.model} | {r.input} | {r.params_M:.2f} | {r.GFLOPs:.2f} | {adapt} | "
                         f"{f(r.pre_ms_mean, r.pre_ms_sd)} | {f(r.inf_ms_mean, r.inf_ms_sd)} | "
                         f"{f(r.post_ms_mean, r.post_ms_sd)} | {f(r.e2e_ms_mean, r.e2e_ms_sd)} | {r.FPS:.1f} |")
    lines.append(f"\nNote: batch size 1, input {IMGSZ}×{IMGSZ}, FP{'16' if USE_HALF else '32'}, "
                 f"{WARMUP} warm-up iterations; raw internal-test images held in memory "
                 f"(disk I/O excluded). Adaptive preprocessing is applied only to configurations "
                 f"whose final stage was trained on processed images. End-to-end = adaptive "
                 f"preprocessing + Ultralytics preprocess + inference + postprocess, measured with "
                 f"CUDA synchronisation. Values are mean ± SD over images × repeats.")
    md = "\n".join(lines)
    (OUTPUT_DIR / "latency_table.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[DONE] Output di {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
