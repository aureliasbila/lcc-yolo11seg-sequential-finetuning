# NOTE: some comments and console messages in this script are written in Indonesian.
# Edit the CONFIG block (paths) before running. Designed for Google Colab with Google Drive mounted.
# ==========================================================================
#  PREDICTION COMPARISON - 6 KONFIGURASI ABLATION (REVISI UNTUK REVIEWER)
#  YOLO11n-seg | panel ground truth + prediksi | SATU CELL COLAB
# ==========================================================================
#
#  PERUBAHAN DARI VERSI SEBELUMNYA
#  1. BUG WARNA INPUT: Ultralytics menganggap array numpy sebagai BGR.
#     Versi lama mengirim gambar RGB -> kanal merah/biru tertukar saat
#     inferensi. Untuk tugas berbasis warna (LCC) ini mengubah prediksi.
#     Sekarang model menerima BGR; RGB hanya dipakai untuk menggambar.
#  2. SEED dipilih dari VALIDATION (median mAP@0.5:0.95 validation per
#     konfigurasi, dari ablation_validation_runs.csv) atau seed tetap.
#     Test tidak dipakai sama sekali untuk memilih apa pun.
#  3. Panel GROUND TRUTH ditambahkan (dari file label YOLO-seg gambar).
#  4. Setiap deteksi dicocokkan ke ground truth (mask IoU >= 0.5):
#     correct / wrong class / false positive; GT yang tidak terdeteksi
#     dicatat sebagai missed. Hasilnya disimpan ke CSV untuk narasi.
#  5. Label konfigurasi baru (Raw -> Processed, dst.), tanpa "cross-dataset".
#  6. Tidak ada waktu inferensi di gambar maupun tabel (bukan pengukuran
#     latency yang sahih).
#  7. Figure dibuat pada UKURAN CETAK (lebar 6.7 in) dengan font 10 pt dan
#     600 dpi, sesuai format IJIES (huruf >= 10 pt, tanpa (a)(b), tanpa
#     judul di dalam gambar). Warna kelas di gambar = warna di legend.
#  8. Preprocessing memakai implementasi yang sama dengan program BO
#     (langkah uint8). Lebih aman lagi: isi PROCESSED_IMAGE_PATH dengan
#     file gambar yang sama dari Dataset_Gamma_New.
# ==========================================================================

# (Colab) !pip install -q ultralytics

import shutil, warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ==========================================================================
#  KONFIGURASI - EDIT BAGIAN INI
# ==========================================================================
DRIVE_BACKUP = Path("/content/drive/MyDrive/ablation_backup")
LOCAL_RUNS   = Path("/content/runs/ablation")

# hasil re-evaluasi validation (output program validation re-evaluation)
VAL_RUNS_CANDIDATES = [Path("ablation_validation_runs.csv"),
                       Path("/content/ablation_validation_runs.csv"),
                       DRIVE_BACKUP / "ablation_validation_runs.csv"]

IMAGE_PATH = "/content/daun malam.png"      # gambar RAW (dari internal test subset)
LABEL_PATH = "/content/daun malam.txt"      # label YOLO-seg gambar tsb; None jika tidak ada
PROCESSED_IMAGE_PATH = None                 # gambar yang sama dari Dataset_Gamma_New (disarankan)
IMAGE_SOURCE_TEXT = "the internal test subset"

OUTPUT_DIR   = Path("/content/prediction_comparison")
DRIVE_OUTPUT = DRIVE_BACKUP / "comparison_results"

SEED_RULE  = "median_val"     # "median_val" (disarankan) | "fixed"
FIXED_SEED = 0

CONF, NMS_IOU, IMGSZ = 0.25, 0.45, 640    # 0.25 = threshold standar Ultralytics
MATCH_IOU = 0.50                          # kriteria deteksi benar (mask IoU)

# parameter preprocessing terpilih (Table 5 naskah) - nama sama dengan kode BO
PREP = dict(gamma_min=0.7825, gamma_max=1.2026, gamma_target_mean=0.4835,
            edge_blur_sigma=0.6074, edge_strength=0.0469, hsv_saturation_gain=1.0171)

# (config_id, label di figure, stage1, stage2, domain input)
SCENARIOS = [
    ("single_raw",     "Single-stage Raw",       "raw",       None,        "raw"),
    ("ft_raw_raw",     "Raw → Raw",               "raw",       "raw",       "raw"),
    ("cross_proc_raw", "Processed → Raw",         "processed", "raw",       "raw"),
    ("single_proc",    "Single-stage Processed", "processed", None,        "processed"),
    ("ft_proc_proc",   "Processed → Processed",   "processed", "processed", "processed"),
    ("cross_raw_proc", "Raw → Processed",         "raw",       "processed", "processed"),
]

CLASS_COLORS  = {"N23": "#d62728", "N34": "#1f77b4", "N45": "#ff7f0e"}   # merah, biru, oranye
DEFAULT_COLOR = "#9467bd"
STATUS_STYLE  = {"correct": "-", "wrong class": ":", "false positive": "--", "gt": "-"}

FIG_WIDTH_IN, FONT_PT, DPI = 6.7, 10, 600
plt.rcParams.update({"font.family": "serif", "font.size": FONT_PT})
# ==========================================================================


def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return True
    except Exception as e:
        print(f"[INFO] Drive tidak tersedia ({e})")
        return False


# ---------------- preprocessing (sama dengan program BO) ----------------
def preprocess_bgr(img, p=PREP):
    x = img.astype(np.float32) / 255.0
    g = np.clip(np.log(p["gamma_target_mean"]) / np.log(max(float(x.mean()), 1e-6)),
                p["gamma_min"], p["gamma_max"])
    x = np.clip(np.power(x, g) * 255, 0, 255).astype(np.uint8)
    bl = cv2.GaussianBlur(x, (0, 0), sigmaX=p["edge_blur_sigma"])
    x = np.clip(cv2.addWeighted(x, 1 + p["edge_strength"], bl, -p["edge_strength"], 0),
                0, 255).astype(np.uint8)
    h = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 1] = np.clip(h[:, :, 1] * p["hsv_saturation_gain"], 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR), float(g)


# ---------------- seed dari validation ----------------
def pick_seeds():
    if SEED_RULE == "fixed":
        return {cid: (FIXED_SEED, None, f"fixed seed {FIXED_SEED}") for cid, *_ in SCENARIOS}
    path = next((c for c in VAL_RUNS_CANDIDATES if c.exists()), None)
    if path is None:
        print("[WARNING] ablation_validation_runs.csv tidak ditemukan -> seed tetap", FIXED_SEED)
        return {cid: (FIXED_SEED, None, "fallback fixed seed") for cid, *_ in SCENARIOS}
    df = pd.read_csv(path)
    if "eval_split" in df:
        df = df[df.eval_split.astype(str).str.lower().isin(["val", "validation"])]
    print(f"[OK] Seed dipilih dari {path} (validation, {len(df)} run)")
    out = {}
    for cid, *_ in SCENARIOS:
        g = df[df.config_id == cid].sort_values("seed")
        if g.empty:
            out[cid] = (FIXED_SEED, None, "tidak ada di CSV -> seed tetap")
            continue
        v = g.mAP50_95_mask.values
        i = int(np.argmin(np.abs(v - np.median(v))))
        out[cid] = (int(g.seed.values[i]), float(v[i]), f"median validation of {len(v)} seeds")
    return out


def resolve_weights(ds1, ds2, seed):
    run = f"stage1_{ds1}_s{seed}" if ds2 is None else f"stage2_{ds1}_to_{ds2}_s{seed}"
    for root in (LOCAL_RUNS, DRIVE_BACKUP / "runs"):
        w = root / run / "weights" / "best.pt"
        if w.exists():
            return str(w), run
    return None, run


# ---------------- ground truth & pencocokan ----------------
def load_gt(path, W, H, names):
    if not path or not Path(path).exists():
        return None
    gts = []
    for line in open(path):
        v = line.split()
        if len(v) < 7:
            continue
        poly = np.array(v[1:], dtype=float).reshape(-1, 2) * [W, H]
        cid = int(v[0])
        gts.append(dict(class_id=cid, class_name=names.get(cid, str(cid)), polygon=poly,
                        bbox=np.array([*poly.min(0), *poly.max(0)]), confidence=None,
                        status="gt"))
    return gts


def poly_to_mask(poly, H, W):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def match(dets, gts, H, W):
    """Greedy berdasarkan confidence: mask IoU >= MATCH_IOU."""
    if gts is None:
        for d in dets:
            d.update(status="no GT", gt_index=None, gt_class=None, mask_iou=None)
        return dets, []
    gt_masks = [poly_to_mask(g["polygon"], H, W) for g in gts]
    used = set()
    for d in sorted(dets, key=lambda x: -x["confidence"]):
        best, best_iou = None, 0.0
        for j, gm in enumerate(gt_masks):
            if j in used:
                continue
            inter = np.logical_and(d["mask"], gm).sum()
            union = np.logical_or(d["mask"], gm).sum()
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou = j, iou
        if best is not None and best_iou >= MATCH_IOU:
            used.add(best)
            d.update(gt_index=best, gt_class=gts[best]["class_name"], mask_iou=round(best_iou, 3),
                     status="correct" if d["class_name"] == gts[best]["class_name"] else "wrong class")
        else:
            d.update(gt_index=None, gt_class=None, mask_iou=round(best_iou, 3), status="false positive")
    missed = [j for j in range(len(gts)) if j not in used]
    return dets, missed


def extract(res, names, H, W):
    dets = []
    if res.boxes is None or len(res.boxes) == 0:
        return dets
    md = res.masks.data.cpu().numpy() if res.masks is not None else None
    for i in range(len(res.boxes)):
        cid = int(res.boxes.cls[i])
        mask = None
        if md is not None:
            mask = md[i] > 0.5
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        poly = res.masks.xy[i] if res.masks is not None else None
        dets.append(dict(class_id=cid, class_name=names.get(cid, str(cid)),
                         confidence=float(res.boxes.conf[i]),
                         bbox=res.boxes.xyxy[i].cpu().numpy(), polygon=poly, mask=mask))
    return dets


# ---------------- menggambar panel ----------------
def draw_panel(ax, img_rgb, items, title):
    ax.imshow(img_rgb)
    ax.set_axis_off()
    ax.set_title(title, fontsize=FONT_PT, pad=3)
    for it in items:
        col = CLASS_COLORS.get(it["class_name"], DEFAULT_COLOR)
        if it.get("polygon") is not None and len(it["polygon"]) > 2:
            ax.add_patch(Polygon(it["polygon"], closed=True, facecolor=col, alpha=0.35, edgecolor="none"))
        x1, y1, x2, y2 = it["bbox"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col,
                               lw=1.4, ls=STATUS_STYLE.get(it["status"], "-")))
        txt = it["class_name"] if it["confidence"] is None else f'{it["class_name"]} {it["confidence"]:.2f}'
        H, W = img_rgb.shape[:2]
        right = x1 > 0.55 * W                     # label rata kanan agar tidak terpotong tepi
        top_ok = y1 > 0.12 * H                    # label di atas kotak bila ada ruang
        ax.text(x2 if right else x1, y1 if top_ok else y1 + 2, txt, fontsize=FONT_PT, color="white",
                va="bottom" if top_ok else "top", ha="right" if right else "left", clip_on=True,
                bbox=dict(facecolor=col, edgecolor="none", pad=1.0, alpha=0.9))


def main():
    use_drive = mount_drive()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    raw_bgr = cv2.imread(IMAGE_PATH)
    if raw_bgr is None:
        print(f"[FATAL] Gambar tidak ditemukan: {IMAGE_PATH}")
        return
    if PROCESSED_IMAGE_PATH and Path(PROCESSED_IMAGE_PATH).exists():
        proc_bgr, gamma_used = cv2.imread(PROCESSED_IMAGE_PATH), None
        print("[PREP] Memakai gambar processed dari dataset.")
    else:
        proc_bgr, gamma_used = preprocess_bgr(raw_bgr)
        print(f"[PREP] Gambar processed dihitung ulang (gamma adaptif = {gamma_used:.4f}).")
    H, W = raw_bgr.shape[:2]
    imgs_bgr = {"raw": raw_bgr, "processed": proc_bgr}
    imgs_rgb = {k: cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for k, v in imgs_bgr.items()}
    stem = Path(IMAGE_PATH).stem

    seeds = pick_seeds()
    det_rows, sum_rows, panel_items, names = [], [], {}, None

    for cid, label, ds1, ds2, dom in SCENARIOS:
        seed, vmap, note = seeds[cid]
        wpath, run = resolve_weights(ds1, ds2, seed)
        if wpath is None:
            print(f"[SKIP] {label}: best.pt tidak ada untuk {run}")
            continue
        model = YOLO(wpath)
        names = names or model.names
        # PENTING: input BGR (konvensi Ultralytics untuk array numpy)
        res = model.predict(source=imgs_bgr[dom], imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                            device=device, verbose=False, retina_masks=True)[0]
        dets = extract(res, model.names, H, W)
        gts = load_gt(LABEL_PATH, W, H, model.names)
        dets, missed = match(dets, gts, H, W)
        panel_items[cid] = (label, seed, dets)

        for k, d in enumerate(sorted(dets, key=lambda x: -x["confidence"]), 1):
            det_rows.append(dict(configuration=label, seed=seed, input=dom, detection=k,
                                 pred_class=d["class_name"], confidence=round(d["confidence"], 4),
                                 status=d["status"], gt_class=d["gt_class"], mask_iou=d["mask_iou"]))
        st = [d["status"] for d in dets]
        fp_conf = [d["confidence"] for d in dets if d["status"] == "false positive"]
        tp_conf = [d["confidence"] for d in dets if d["status"] in ("correct", "wrong class")]
        sum_rows.append(dict(configuration=label, seed=seed, seed_rule=note,
                             val_mAP50_95_of_seed=None if vmap is None else round(vmap, 4),
                             input=dom, detections=len(dets), correct=st.count("correct"),
                             wrong_class=st.count("wrong class"), false_positive=st.count("false positive"),
                             missed_gt=len(missed) if gts is not None else None,
                             conf_on_leaf=", ".join(f"{c:.4f}" for c in tp_conf),
                             max_conf_false_positive=round(max(fp_conf), 4) if fp_conf else None))

    if not panel_items:
        print("[FATAL] Tidak ada model yang dimuat.")
        return

    # ---------------- figure: baris 1 input raw, baris 2 input processed ----------------
    gts_raw = load_gt(LABEL_PATH, W, H, names)
    layout = [("gt", "raw"), "single_raw", "ft_raw_raw", "cross_proc_raw",
              ("gt", "processed"), "single_proc", "ft_proc_proc", "cross_raw_proc"]
    pw = FIG_WIDTH_IN / 4
    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, 2 * (pw * H / W + 0.65) + 0.5))
    for ax, key in zip(axes.ravel(), layout):
        if isinstance(key, tuple):
            dom = key[1]
            title = ("Ground truth" if gts_raw is not None else "Input") + f"\n{dom} input\n "
            draw_panel(ax, imgs_rgb[dom], gts_raw or [], title)
        elif key in panel_items:
            label, seed, dets = panel_items[key]
            n = len(dets)
            dom = "processed" if key in ("single_proc", "ft_proc_proc", "cross_raw_proc") else "raw"
            name = label.replace("Single-stage ", "Single-stage\n").replace(" \u2192 ", " \u2192\n")
            draw_panel(ax, imgs_rgb[dom], dets, f"{name}\n{n} detection{'s' if n != 1 else ''}")
        else:
            ax.set_axis_off()
    handles = [Patch(facecolor=c, alpha=0.6, label=k) for k, c in CLASS_COLORS.items()]
    handles += [Line2D([0], [0], color="black", lw=1.4, ls="-", label="Correct / ground truth"),
                Line2D([0], [0], color="black", lw=1.4, ls=":", label="Wrong class"),
                Line2D([0], [0], color="black", lw=1.4, ls="--", label="False positive")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FONT_PT)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.4, w_pad=0.3)

    out_png = OUTPUT_DIR / f"{stem}_prediction_comparison.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()

    det_df, sum_df = pd.DataFrame(det_rows), pd.DataFrame(sum_rows)
    det_csv = OUTPUT_DIR / f"{stem}_detections.csv"
    sum_csv = OUTPUT_DIR / f"{stem}_summary.csv"
    det_df.to_csv(det_csv, index=False)
    sum_df.to_csv(sum_csv, index=False)
    print("\nRINGKASAN PER KONFIGURASI\n" + sum_df.to_string(index=False))
    print("\nDETAIL PER DETEKSI\n" + det_df.to_string(index=False))

    if gts_raw is not None:
        gt_txt = ", ".join(g["class_name"] for g in gts_raw)
        print(f"\nGround truth: {len(gts_raw)} leaf instance(s): {gt_txt}")

    if use_drive:
        DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)
        for f in (out_png, det_csv, sum_csv):
            shutil.copy2(f, DRIVE_OUTPUT / f.name)
        print(f"[OK] Backup -> {DRIVE_OUTPUT}")

    seed_rule = ("the seed whose validation mAP@0.5:0.95 was the median of the three seeds"
                 if SEED_RULE == "median_val" else f"seed {FIXED_SEED}")
    print("\nCAPTION (sesuaikan nomor figure):")
    print(f"Fig. X. Ground truth and predictions of the six training configurations on an example "
          f"nighttime image from {IMAGE_SOURCE_TEXT}. The top row uses the original image and the "
          f"bottom row the preprocessed image, according to the final training domain of each "
          f"configuration. For each configuration, {seed_rule} is shown. Predictions with confidence "
          f">= {CONF:.2f} are displayed; solid, dotted, and dashed boxes denote correct detections, "
          f"wrong-class detections, and false positives (mask IoU threshold {MATCH_IOU:.2f}).")


main()
