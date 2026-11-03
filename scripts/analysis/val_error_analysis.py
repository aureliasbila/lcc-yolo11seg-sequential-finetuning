# NOTE: some comments and console messages in this script are written in Indonesian.
# Edit the CONFIG block (paths) before running. Designed for Google Colab with Google Drive mounted.
# ==========================================================================
#  VALIDATION ERROR ANALYSIS + FIG. 15 - 6 KONFIGURASI x 3 SEED (18 MODEL)
#  YOLO11n-seg | SATU CELL COLAB | dataset & bobot dari Google Drive
# ==========================================================================
#
#  TUJUAN
#  1. Menghitung jenis kesalahan setiap konfigurasi pada SELURUH validation
#     set (deskriptif, melengkapi Table 11):
#        correct      : mask IoU >= 0.5 dengan GT dan kelas benar
#        wrong class  : mask IoU >= 0.5 dengan GT tetapi kelas salah
#        false pos.   : tidak cocok dengan GT mana pun (termasuk deteksi ganda)
#        missed       : GT yang tidak terdeteksi
#     Dihitung untuk ke-3 seed -> total dan mean +/- SD per seed.
#  2. Confusion (GT class -> predicted class / missed) per konfigurasi.
#  3. Memilih gambar untuk Fig. 15 dengan ATURAN TERTULIS (bukan manual),
#     memakai model seed-median-validation tiap konfigurasi:
#        - "disagreement": gambar dengan jumlah hasil BERBEDA terbanyak
#          di antara 6 konfigurasi (tie-break: total kesalahan terbanyak,
#          lalu nama file).
#        - "typical": dipilih acak (seed 42) dari gambar yang diprediksi
#          benar sempurna oleh ke-6 konfigurasi.
#  4. Menggambar Fig. 15 (format IJIES: 6.7 in, font 10 pt, 600 dpi).
#
#  TEST SET TIDAK DISENTUH. Gambar dikirim ke model sebagai PATH FILE
#  (Ultralytics membacanya sendiri dalam BGR -> tidak ada bug warna).
# ==========================================================================

# (Colab) !pip install -q ultralytics

import json, shutil, warnings
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
DRIVE        = Path("/content/drive/MyDrive")
DATASET_DIR  = DRIVE / "dataset"
RAW_DS       = DATASET_DIR / "Dataset_Roboflow"      # dataset raw
PROC_DS      = DATASET_DIR / "Dataset_Gamma_New"     # dataset processed
RUNS_ROOTS   = [Path("/content/runs/ablation"), DRIVE / "ablation_backup" / "runs"]
VAL_RUNS_CANDIDATES = [Path("/content/ablation_validation_runs.csv"),
                       Path("ablation_validation_runs.csv"),
                       DRIVE / "ablation_backup" / "ablation_validation_runs.csv"]
OUTPUT_DIR   = DRIVE / "ablation_backup" / "val_error_analysis"

SEEDS = [0, 1, 2]
CONF, NMS_IOU, IMGSZ = 0.25, 0.45, 640    # threshold operasional (bukan threshold mAP)
MATCH_IOU = 0.50
RANDOM_SEED = 42                          # untuk memilih gambar "typical"
SPLIT_NAMES = ["valid", "val", "validation"]

SCENARIOS = [   # urutan sama dengan Table 11
    ("single_raw",     "Single-stage Raw",       "raw",       None,        "raw"),
    ("ft_raw_raw",     "Raw → Raw",               "raw",       "raw",       "raw"),
    ("single_proc",    "Single-stage Processed", "processed", None,        "processed"),
    ("ft_proc_proc",   "Processed → Processed",   "processed", "processed", "processed"),
    ("cross_raw_proc", "Raw → Processed",         "raw",       "processed", "processed"),
    ("cross_proc_raw", "Processed → Raw",         "processed", "raw",       "raw"),
]
CLASS_ORDER   = ["N23", "N34", "N45"]
CLASS_COLORS  = {"N23": "#d62728", "N34": "#1f77b4", "N45": "#ff7f0e"}
DEFAULT_COLOR = "#9467bd"
STATUS_STYLE  = {"correct": "-", "wrong class": ":", "false positive": "--", "gt": "-"}
FIG_WIDTH_IN, FONT_PT, DPI = 6.7, 10, 600
plt.rcParams.update({"font.family": "serif", "font.size": FONT_PT})
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
    for p in ds_root.rglob("images"):                     # dataset di subfolder
        if p.parent.name in SPLIT_NAMES:
            return p.parent
    raise FileNotFoundError(f"Folder validation tidak ditemukan di {ds_root}")


def list_images(split_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return {p.stem: p for p in sorted((split_dir / "images").iterdir()) if p.suffix.lower() in exts}


def resolve_weights(ds1, ds2, seed):
    run = f"stage1_{ds1}_s{seed}" if ds2 is None else f"stage2_{ds1}_to_{ds2}_s{seed}"
    for root in RUNS_ROOTS:
        w = root / run / "weights" / "best.pt"
        if w.exists():
            return str(w), run
    return None, run


def median_val_seeds():
    path = next((c for c in VAL_RUNS_CANDIDATES if c.exists()), None)
    if path is None:
        print("[WARNING] ablation_validation_runs.csv tidak ditemukan -> seed 0 untuk figure")
        return {cid: 0 for cid, *_ in SCENARIOS}
    df = pd.read_csv(path)
    out = {}
    for cid, *_ in SCENARIOS:
        g = df[df.config_id == cid].sort_values("seed")
        v = g.mAP50_95_mask.values
        out[cid] = int(g.seed.values[int(np.argmin(np.abs(v - np.median(v))))]) if len(v) else 0
    print(f"[OK] Seed median validation dari {path}: {out}")
    return out


def load_gt(label_path, W, H, names):
    gts = []
    if not label_path.exists():
        return gts
    for line in open(label_path):
        v = line.split()
        if len(v) < 7:
            continue
        poly = np.array(v[1:], dtype=float).reshape(-1, 2) * [W, H]
        cid = int(v[0])
        gts.append(dict(class_name=names.get(cid, str(cid)), polygon=poly,
                        bbox=np.array([*poly.min(0), *poly.max(0)]), confidence=None, status="gt"))
    return gts


def poly_to_mask(poly, H, W):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def predict(model, img_path, H, W, device):
    res = model.predict(source=str(img_path), imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                        device=device, verbose=False, retina_masks=True)[0]
    dets = []
    if res.boxes is None or len(res.boxes) == 0:
        return dets
    md = res.masks.data.cpu().numpy() if res.masks is not None else None
    for i in range(len(res.boxes)):
        mask = None
        if md is not None:
            mask = md[i] > 0.5
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        cid = int(res.boxes.cls[i])
        dets.append(dict(class_name=model.names.get(cid, str(cid)), confidence=float(res.boxes.conf[i]),
                         bbox=res.boxes.xyxy[i].cpu().numpy(),
                         polygon=res.masks.xy[i] if res.masks is not None else None, mask=mask))
    return dets


def match(dets, gts, gt_masks):
    used, pairs = set(), []
    for d in sorted(dets, key=lambda x: -x["confidence"]):
        best, best_iou = None, 0.0
        if d["mask"] is not None:
            for j, gm in enumerate(gt_masks):
                if j in used:
                    continue
                union = np.logical_or(d["mask"], gm).sum()
                iou = np.logical_and(d["mask"], gm).sum() / union if union else 0.0
                if iou > best_iou:
                    best, best_iou = j, iou
        if best is not None and best_iou >= MATCH_IOU:
            used.add(best)
            ok = d["class_name"] == gts[best]["class_name"]
            d["status"] = "correct" if ok else "wrong class"
            pairs.append((gts[best]["class_name"], d["class_name"]))
        else:
            d["status"] = "false positive"
    missed = [gts[j]["class_name"] for j in range(len(gts)) if j not in used]
    for c in missed:
        pairs.append((c, "missed"))
    return dets, pairs, missed


def draw_panel(ax, img_rgb, items, title):
    ax.imshow(img_rgb)
    ax.set_axis_off()
    ax.set_title(title, fontsize=FONT_PT, pad=3)
    H, W = img_rgb.shape[:2]
    for it in items:
        col = CLASS_COLORS.get(it["class_name"], DEFAULT_COLOR)
        if it.get("polygon") is not None and len(it["polygon"]) > 2:
            ax.add_patch(Polygon(it["polygon"], closed=True, facecolor=col, alpha=0.35, edgecolor="none"))
        x1, y1, x2, y2 = it["bbox"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=1.4,
                               ls=STATUS_STYLE.get(it.get("status", "gt"), "-")))
        txt = it["class_name"] if it["confidence"] is None else f'{it["class_name"]} {it["confidence"]:.2f}'
        right, top_ok = x1 > 0.55 * W, y1 > 0.12 * H
        ax.text(x2 if right else x1, y1 if top_ok else y1 + 2, txt, fontsize=FONT_PT, color="white",
                va="bottom" if top_ok else "top", ha="right" if right else "left", clip_on=True,
                bbox=dict(facecolor=col, edgecolor="none", pad=1.0, alpha=0.9))


def make_figure(stem, raw_imgs, proc_imgs, raw_lbl, models_fig, names, device, out_png):
    raw_p, proc_p = raw_imgs[stem], proc_imgs[stem]
    rgb = {"raw": cv2.cvtColor(cv2.imread(str(raw_p)), cv2.COLOR_BGR2RGB),
           "processed": cv2.cvtColor(cv2.imread(str(proc_p)), cv2.COLOR_BGR2RGB)}
    H, W = rgb["raw"].shape[:2]
    gts = load_gt(raw_lbl / f"{stem}.txt", W, H, names)
    gt_masks = [poly_to_mask(g["polygon"], H, W) for g in gts]
    layout = [("gt", "raw"), "single_raw", "ft_raw_raw", "cross_proc_raw",
              ("gt", "processed"), "single_proc", "ft_proc_proc", "cross_raw_proc"]
    pw = FIG_WIDTH_IN / 4
    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, 2 * (pw * H / W + 0.75) + 0.5))
    lines = []
    for ax, key in zip(axes.ravel(), layout):
        if isinstance(key, tuple):
            dom = key[1]
            draw_panel(ax, rgb[dom], gts, f"Ground truth\n{'original' if dom == 'raw' else 'preprocessed'} input\n ")
            continue
        label, dom, model = models_fig[key]
        path = raw_p if dom == "raw" else proc_p
        dets, _, missed = match(predict(model, path, H, W, device), gts, gt_masks)
        n = len(dets)
        name = label.replace("Single-stage ", "Single-stage\n").replace(" → ", " →\n")
        draw_panel(ax, rgb[dom], dets, f"{name}\n{n} detection{'s' if n != 1 else ''}")
        lines.append(f"  {label:<24} " + "; ".join(f"{d['class_name']} {d['confidence']:.3f} ({d['status']})"
                                                 for d in sorted(dets, key=lambda x: -x['confidence']))
                     + (f"; missed: {', '.join(missed)}" if missed else ""))
    handles = [Patch(facecolor=CLASS_COLORS[c], alpha=0.6, label=c) for c in CLASS_ORDER]
    handles += [Line2D([0], [0], color="black", lw=1.4, ls="-", label="Correct / ground truth"),
                Line2D([0], [0], color="black", lw=1.4, ls=":", label="Wrong class"),
                Line2D([0], [0], color="black", lw=1.4, ls="--", label="False positive")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FONT_PT)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.2, w_pad=0.3)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    gt_txt = ", ".join(g["class_name"] for g in gts) or "none"
    print(f"\n{out_png.name}: {stem}  | GT: {len(gts)} leaf ({gt_txt})\n" + "\n".join(lines))


def main():
    mount_drive()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    raw_split, proc_split = find_split(RAW_DS), find_split(PROC_DS)
    raw_imgs, proc_imgs = list_images(raw_split), list_images(proc_split)
    raw_lbl = raw_split / "labels"
    proc_lbl = proc_split / "labels" if (proc_split / "labels").exists() else raw_lbl
    stems = sorted(set(raw_imgs) & set(proc_imgs))
    print(f"[DATA] raw val: {len(raw_imgs)} | processed val: {len(proc_imgs)} | pasangan: {len(stems)}")
    if len(stems) < len(raw_imgs):
        print(f"[WARNING] {len(raw_imgs) - len(stems)} gambar raw tanpa pasangan processed (dilewati).")

    # ukuran gambar & GT disimpan sekali per domain
    cache = {}
    names = None
    rows, confusion = [], {}

    for cid, label, ds1, ds2, dom in SCENARIOS:
        for seed in SEEDS:
            wpath, run = resolve_weights(ds1, ds2, seed)
            if wpath is None:
                print(f"[SKIP] {label} seed {seed}: {run} tidak ditemukan")
                continue
            model = YOLO(wpath)
            names = names or model.names
            imgs, lbl = (raw_imgs, raw_lbl) if dom == "raw" else (proc_imgs, proc_lbl)
            print(f"[RUN] {label:<24} seed {seed}  ({len(stems)} gambar, input {dom})")
            for stem in stems:
                key = (dom, stem)
                if key not in cache:
                    im = cv2.imread(str(imgs[stem]))
                    H, W = im.shape[:2]
                    gts = load_gt(lbl / f"{stem}.txt", W, H, model.names)
                    cache[key] = (H, W, gts, [poly_to_mask(g["polygon"], H, W) for g in gts])
                H, W, gts, gt_masks = cache[key]
                dets, pairs, missed = match(predict(model, imgs[stem], H, W, device), gts, gt_masks)
                st = [d["status"] for d in dets]
                rows.append(dict(config_id=cid, configuration=label, seed=seed, image=stem,
                                 n_gt=len(gts), correct=st.count("correct"),
                                 wrong_class=st.count("wrong class"),
                                 false_positive=st.count("false positive"), missed=len(missed)))
                for g, p in pairs:
                    confusion[(label, g, p)] = confusion.get((label, g, p), 0) + 1

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "val_errors_per_image.csv", index=False)
    cols = ["n_gt", "correct", "wrong_class", "false_positive", "missed"]

    per_seed = df.groupby(["configuration", "seed"], sort=False)[cols].sum().reset_index()
    per_seed.to_csv(OUTPUT_DIR / "val_errors_per_seed.csv", index=False)
    order = [s[1] for s in SCENARIOS]
    summ = []
    for lab in order:
        g = per_seed[per_seed.configuration == lab]
        if g.empty:
            continue
        r = dict(configuration=lab, n_seeds=len(g))
        for c in cols:
            r[f"{c}_total"] = int(g[c].sum())
            r[f"{c}_mean"] = round(g[c].mean(), 2)
            r[f"{c}_sd"] = round(g[c].std(ddof=1), 2) if len(g) > 1 else 0.0
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUTPUT_DIR / "val_errors_summary.csv", index=False)

    conf_rows = []
    for lab in order:
        for g in CLASS_ORDER:
            r = dict(configuration=lab, gt_class=g)
            for p in CLASS_ORDER + ["missed"]:
                r[p] = confusion.get((lab, g, p), 0)
            conf_rows.append(r)
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(OUTPUT_DIR / "val_confusion_by_config.csv", index=False)

    # tabel siap tempel: mean +/- SD per seed
    md = ["| Configuration | Correct | Wrong class | False positive | Missed |", "|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        md.append(f"| {r.configuration} | " + " | ".join(
            f"{r[c + '_mean']:.1f} ± {r[c + '_sd']:.1f}" for c in ["correct", "wrong_class", "false_positive", "missed"]) + " |")
    n_gt = int(summ.n_gt_mean.iloc[0]) if len(summ) else 0
    md.append(f"\n*Counts per model on the validation subset ({len(stems)} images, {n_gt} annotated instances), "
              f"mean ± SD over three seeds; confidence threshold {CONF}, mask IoU ≥ {MATCH_IOU}.*")
    (OUTPUT_DIR / "val_errors_table.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 90 + "\nRINGKASAN KESALAHAN DI VALIDATION (mean ± SD per seed)\n" + "=" * 90)
    print("\n".join(md))
    print("\nCONFUSION (dijumlah 3 seed): GT class -> predicted class / missed")
    print(conf_df.to_string(index=False))

    # ---------------- pemilihan gambar Fig. 15 (model seed-median) ----------------
    med = median_val_seeds()
    sel = df[[df.loc[i, "seed"] == med[df.loc[i, "config_id"]] for i in df.index]]
    sig = sel.assign(sig=sel[["correct", "wrong_class", "false_positive", "missed"]].astype(str).agg("-".join, axis=1),
                     err=sel.wrong_class + sel.false_positive + sel.missed)
    per_img = sig.groupby("image").agg(n_configs=("sig", "size"), distinct=("sig", "nunique"),
                                       total_err=("err", "sum")).reset_index()
    per_img = per_img[per_img.n_configs == len(SCENARIOS)]
    hardest = per_img.sort_values(["distinct", "total_err", "image"], ascending=[False, False, True]).iloc[0]
    perfect = per_img[per_img.total_err == 0].image.tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    typical = rng.choice(perfect) if perfect else per_img.sort_values(["total_err", "image"]).image.iloc[0]
    per_img.to_csv(OUTPUT_DIR / "val_image_disagreement.csv", index=False)
    print(f"\n[FIG] disagreement image: {hardest.image} (distinct outcomes = {hardest.distinct}, "
          f"total errors = {hardest.total_err})")
    print(f"[FIG] typical image     : {typical} (dipilih acak dari {len(perfect)} gambar tanpa kesalahan)")

    models_fig = {}
    for cid, label, ds1, ds2, dom in SCENARIOS:
        w, _ = resolve_weights(ds1, ds2, med[cid])
        if w:
            models_fig[cid] = (label, dom, YOLO(w))
    make_figure(hardest.image, raw_imgs, proc_imgs, raw_lbl, models_fig, names, device,
                OUTPUT_DIR / "fig15_disagreement.png")
    make_figure(typical, raw_imgs, proc_imgs, raw_lbl, models_fig, names, device,
                OUTPUT_DIR / "fig15_typical.png")

    json.dump(dict(conf=CONF, match_iou=MATCH_IOU, median_val_seeds=med,
                   disagreement_image=str(hardest.image), typical_image=str(typical),
                   n_val_images=len(stems)),
              open(OUTPUT_DIR / "val_error_analysis_config.json", "w"), indent=2)
    print(f"\n[OK] Semua output di {OUTPUT_DIR}")
    print("\nCAPTION (Fig. 15): Ground truth and predictions of the six training configurations on the "
          "validation image with the largest disagreement among configurations (top row: original input; "
          "bottom row: preprocessed input). For each configuration, the seed with the median validation "
          f"mAP@0.5:0.95 is shown; predictions with confidence ≥ {CONF} are displayed; solid, dotted, and "
          "dashed boxes denote correct detections, wrong-class detections, and false positives.")


main()
