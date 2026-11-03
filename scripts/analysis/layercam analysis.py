# -*- coding: utf-8 -*-
"""
Grad-CAM YOLO11n-seg — Fig. 15 (selected configuration: Raw → Processed)
=======================================================================
Row layout:
  1–3 : internal test (Dataset_Gamma_New/test), one image per class N23/N34/N45,
        chosen RANDOMLY (fixed seed) from images whose GT contains that class — without looking at predictions.
  4   : external Sidoarjo, an N34 leaf WRONGLY predicted as N23
        (rule: the wrong case with the highest N23 prediction confidence)
  5   : external Sidoarjo, an N34 leaf CORRECTLY predicted
        (rule: the correct case with the highest confidence)
Columns: Model input | Ground-truth mask | YOLO segmentation | Grad-CAM

External: /content/New_data.zip -> New_Sda/test (the same zip as Table 16/17)
Model   : stage2_raw_to_processed_s{seed}/weights/best.pt
Seed    : selected AUTOMATICALLY = the seed whose VALIDATION mAP@0.5:0.95 is closest to the median of the 3 seeds
Layer   : default "head_cls" = the classification-branch feature of the head at the scale (P3/P4/P5) where the
          anchor of the leaf being explained is located; the "auto" option = the old program's layer-selection mechanism.
Input   : the model was trained on processed images -> internal uses Dataset_Gamma_New,
          external Sidoarjo is given the same BO preprocessing (Table 6) before prediction.
"""
!pip install -q ultralytics

import json, random, warnings
from pathlib import Path
import cv2, numpy as np, pandas as pd, yaml, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURATION
# =========================================================
DRIVE       = Path("/content/drive/MyDrive")
DATASET_DIR = DRIVE / "dataset"
PROC_DS     = DATASET_DIR / "Dataset_Gamma_New"          # internal (processed) train/valid/test
EXTERNAL_ZIP = Path("/content/New_data.zip")             # external zip used for Table 16/17
EXTERNAL_SRC = Path("/content/tmp_gradcam_external")     # extraction destination
EXT_SITE     = "New_Sda"                                  # Sidoarjo folder inside the zip (New_Sda/test/images, labels)
EXT_SDA      = None                                       # filled automatically after extraction
RUNS_ROOTS  = [Path("/content/runs/ablation"), DRIVE / "ablation_backup" / "runs"]
RUN_TEMPLATE = "stage2_raw_to_processed_s{seed}"         # selected configuration
SEEDS       = [0, 1, 2]
OUTPUT_DIR  = DRIVE / "ablation_backup" / "gradcam_fig15"

CLASS_NAMES = ["N23", "N34", "N45"]
IMG_SIZE    = 640
CONF_OP, NMS_IOU_OP, MATCH_IOU = 0.25, 0.45, 0.50       # same as external evaluation
SAMPLE_SEED = 42                                         # seed for random selection of internal images
PREP = dict(gamma_min=0.7825, gamma_max=1.2026, gamma_target_mean=0.4835,
            edge_blur_sigma=0.6074, edge_strength=0.0469, hsv_saturation_gain=1.0171)

CAM_LAYER          = "head_cls"   # "head_cls" = head classification branch (the feature that directly produces class scores),
                                  #   scale P3/P4/P5 is chosen automatically according to the anchor being explained.
                                  # Enter a number (e.g. 10) or "auto" to use a backbone/neck layer like the old program.
EXPERIMENT_LAYERS  = [6, 8, 10, 13, 16, 19, 22]          # used only if CAM_LAYER = "auto"
FIXED_LAYER        = None      # ("auto" mode) enter a number to force a specific layer
N_LAYER_SELECT_IMG = 20        # number of VALIDATION images for layer selection
GRADIENT_THRESHOLD = 1e-8
GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA = (15, 15), 5
TOPK_ANCHORS        = 10      # number of anchors (inside the leaf mask) used as Grad-CAM targets
CAM_NORM_PERCENTILE = 99.0    # normalize the heatmap at this percentile (not the maximum value)
FONT_PT = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
plt.rcParams.update({"font.family": "serif", "font.size": FONT_PT})


# =========================================================
# 2. DATA UTILITIES
# =========================================================
def mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print(f"[INFO] Drive not mounted ({e})")


def extract_external():
    """Extract the external zip then return the <EXT_SITE>/test folder (containing images/ and labels/)."""
    import shutil, zipfile
    zp = EXTERNAL_ZIP
    if not zp.exists():
        alt = sorted(Path("/content").glob("New*ata*.zip"))
        if not alt:
            raise FileNotFoundError(f"External zip not found: {EXTERNAL_ZIP}")
        zp = alt[0]
        print(f"[ZIP] Using {zp.name}")
    if EXTERNAL_SRC.exists():
        shutil.rmtree(EXTERNAL_SRC)
    EXTERNAL_SRC.mkdir(parents=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(EXTERNAL_SRC)
    cands = sorted((p for p in EXTERNAL_SRC.rglob(EXT_SITE) if p.is_dir()), key=lambda p: len(p.parts))
    if not cands:
        raise FileNotFoundError(f"Folder {EXT_SITE} not found in {zp.name}")
    site = cands[0]
    root = site / "test" if (site / "test" / "images").exists() else site
    n_img = len([p for p in (root / "images").iterdir() if p.suffix.lower() in IMG_EXT])
    print(f"[ZIP] {zp.name} -> {root} ({n_img} images)")
    return root


def adaptive_preprocess(img, p=PREP):
    x = img.astype(np.float32) / 255.0
    g = np.clip(np.log(p["gamma_target_mean"]) / np.log(max(float(x.mean()), 1e-6)),
                p["gamma_min"], p["gamma_max"])
    x = np.clip(np.power(x, g) * 255, 0, 255).astype(np.uint8)
    bl = cv2.GaussianBlur(x, (0, 0), sigmaX=p["edge_blur_sigma"])
    x = np.clip(cv2.addWeighted(x, 1 + p["edge_strength"], bl, -p["edge_strength"], 0), 0, 255).astype(np.uint8)
    h = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    h[:, :, 1] = np.clip(h[:, :, 1] * p["hsv_saturation_gain"], 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)


def split_dir(root, names):
    for n in names:
        if (root / n / "images").exists():
            return root / n
    raise FileNotFoundError(f"split {names} not found in {root}")


def images_with_labels(root):
    """All images under root (recursive) together with their paired label files (.../labels/<stem>.txt)."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.parent.name == "images":
            lp = p.parent.parent / "labels" / (p.stem + ".txt")
            if lp.exists():
                out.append((p, lp))
    return out


def read_gt(lp, H, W):
    """List of (class_id, bool mask) from a YOLO-seg label."""
    gts = []
    for line in lp.read_text().splitlines():
        v = line.split()
        if len(v) < 7 or (len(v) - 1) % 2:
            continue
        pts = np.clip(np.array(v[1:], np.float32).reshape(-1, 2) * [W, H], 0, [W - 1, H - 1]).astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        gts.append((int(float(v[0])), m.astype(bool)))
    return gts


# =========================================================
# 3. MODEL & SEED (validation median)
# =========================================================
def resolve_weights(seed):
    run = RUN_TEMPLATE.format(seed=seed)
    for r in RUNS_ROOTS:
        w = r / run / "weights" / "best.pt"
        if w.exists():
            return w
    raise FileNotFoundError(f"{run}/weights/best.pt not found in {RUNS_ROOTS}")


def select_median_seed():
    val_dir = split_dir(PROC_DS, ("valid", "val"))
    yml = Path("/content/tmp_gradcam_val.yaml")
    yaml.safe_dump({"path": "/", "train": str(val_dir / "images"), "val": str(val_dir / "images"),
                    "names": {i: n for i, n in enumerate(CLASS_NAMES)}}, open(yml, "w"))
    rows = []
    for s in SEEDS:
        m = YOLO(str(resolve_weights(s)), task="segment").val(
            data=str(yml), split="val", imgsz=IMG_SIZE, batch=16, plots=False, verbose=False,
            project="/content/tmp_gradcam", name=f"val_s{s}", exist_ok=True)
        rows.append(dict(seed=s, mAP50=float(m.seg.map50), mAP50_95=float(m.seg.map)))
    df = pd.DataFrame(rows)
    med = df.mAP50_95.median()
    seed = int(df.iloc[(df.mAP50_95 - med).abs().argsort().iloc[0]].seed)
    print("\n[SEED] Validation per seed (processed validation):\n" + df.to_string(index=False))
    print(f"[SEED] Median mAP@0.5:0.95 = {med:.4f} -> selected seed = {seed}")
    return seed, df


# =========================================================
# 4. PREDICTION & MATCHING (same rules as external evaluation)
# =========================================================
def predict(model, img):
    r = model.predict(img, imgsz=IMG_SIZE, conf=CONF_OP, iou=NMS_IOU_OP,
                      retina_masks=True, verbose=False, device=DEVICE)[0]
    H, W = img.shape[:2]
    if r.masks is None or len(r.boxes) == 0:
        return np.zeros((0, H, W), bool), np.zeros(0, int), np.zeros(0)
    pm = r.masks.data.cpu().numpy() > 0.5
    if pm.shape[1:] != (H, W):
        pm = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in pm])
    return pm, r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()


def match(pm, pc, ps, gts):
    """Greedy matching (decreasing confidence, mask IoU >= MATCH_IOU). Return dict gt_idx -> (pred_idx, iou)."""
    used, res = set(), {}
    for i in np.argsort(-ps):
        best, bj = 0.0, -1
        for j, (_, g) in enumerate(gts):
            if j in used:
                continue
            iou = np.logical_and(pm[i], g).sum() / max(np.logical_or(pm[i], g).sum(), 1)
            if iou > best:
                best, bj = iou, j
        if best >= MATCH_IOU:
            used.add(bj); res[bj] = (int(i), float(best))
    return res


# =========================================================
# 5. GRAD-CAM (core identical to the old program)
# =========================================================
def preprocess_tensor(img_bgr):
    x = cv2.cvtColor(cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


class YOLOGradCAM:
    def __init__(self, model, layers):
        self.net = model.model
        self.head_mode = layers == "head_cls"
        if self.head_mode:
            head = self.net.model[-1]
            self.modules = {f"P{3 + i}": head.cv3[i][1] for i in range(len(head.cv3))}   # class features per scale
        else:
            ls = [layers] if isinstance(layers, int) else list(layers)
            self.modules = {i: self.net.model[i] for i in ls}
        self.layers = list(self.modules)
        self.act, self.grad, self.gmag, self.hooks = {}, {}, {}, []
        self.last_idx = None
        for p in self.net.parameters():
            p.requires_grad_(True)
        for key, mod in self.modules.items():
            self.hooks.append(mod.register_forward_hook(self._fa(key)))
            self.hooks.append(mod.register_full_backward_hook(self._fg(key)))

    def _fa(self, idx):
        def h(m, i, o):
            self.act[idx] = o.detach()
        return h

    def _fg(self, idx):
        def h(m, gi, go):
            g = go[0]
            self.grad[idx] = None if g is None else g.detach()
            self.gmag[idx] = 0.0 if g is None else float(g.abs().mean())
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()

    @staticmethod
    def _tensors(o):
        if isinstance(o, torch.Tensor):
            return [o]
        if isinstance(o, (list, tuple)):
            return [t for x in o for t in YOLOGradCAM._tensors(x)]
        if isinstance(o, dict):
            return [t for x in o.values() for t in YOLOGradCAM._tensors(x)]
        return []

    def _class_score(self, out, cls, mask=None):
        """Instance-specific target for Grad-CAM.
        - The prediction tensor is taken EXPLICITLY: (1, 4+nc+32, N) = [xywh pixels | class scores | mask coeffs].
          (Ultralytics 8.4 also outputs a 64-channel 'boxes' tensor; picking the tensor with the
           most channels — like the old program — wrongly grabs the box distribution, not class scores.)
        - Class scores use raw logits (the 'scores' dict) when available, so the gradient is not lost
          to a saturated sigmoid; otherwise logits are computed from the probabilities.
        - Only the TOP-K anchors whose box centers lie inside the explained leaf prediction mask are used."""
        k = len(CLASS_NAMES)
        ts = [t for t in self._tensors(out) if t.dim() == 3]
        exact = [t for t in ts if t.shape[1] == 4 + k + 32]
        pred = (exact or [t for t in ts if t.shape[1] >= 4 + k and t.shape[1] != 64])[0][0]
        c = min(cls, k - 1)
        logits = None
        if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], dict) \
                and "scores" in out[1] and out[1]["scores"].shape[1] == k:
            logits = out[1]["scores"][0, c]                       # raw logit
        if logits is None:
            p = pred[4 + c].clamp(1e-6, 1 - 1e-6)
            logits = torch.log(p / (1 - p))
        idx = None
        if mask is not None:
            m = torch.from_numpy(cv2.resize(mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                                            interpolation=cv2.INTER_NEAREST) > 0).to(logits.device)
            cx = pred[0].detach().long().clamp(0, IMG_SIZE - 1)
            cy = pred[1].detach().long().clamp(0, IMG_SIZE - 1)
            idx = torch.nonzero(m[cy, cx]).squeeze(1)
        pool = idx if idx is not None and idx.numel() > 0 else torch.arange(logits.numel(), device=logits.device)
        top = logits[pool].topk(min(TOPK_ANCHORS, pool.numel()))
        self.last_idx = pool[top.indices].detach().cpu().numpy()      # global anchor indices being explained
        return top.values.sum()

    def __call__(self, img_bgr, cls, mask=None):
        self.net.eval(); self.net.zero_grad()
        self.act.clear(); self.grad.clear(); self.gmag.clear()
        x = preprocess_tensor(img_bgr).requires_grad_(True)
        self._class_score(self.net(x), cls, mask).backward()
        H, W = img_bgr.shape[:2]
        res = {}
        for idx in self.layers:
            a, g = self.act.get(idx), self.grad.get(idx)
            if a is None or g is None:
                continue
            cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True)).squeeze().cpu().numpy()
            cam = cv2.GaussianBlur(cv2.resize(np.atleast_2d(cam), (W, H)), GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
            hi = np.percentile(cam, CAM_NORM_PERCENTILE)
            cam = np.clip(cam / hi, 0, 1) if hi > 0 else cam      # percentile normalization, not a single-pixel max
            res[idx] = dict(cam=cam, grad=self.gmag.get(idx, 0.0),
                            valid=self.gmag.get(idx, 0.0) > GRADIENT_THRESHOLD)
        if self.head_mode and res:
            # anchors are ordered P3 -> P4 -> P5; pick the scale where the majority of explained anchors lie
            sizes = [self.act[k].shape[2] * self.act[k].shape[3] for k in self.layers]
            bounds = np.cumsum(sizes)
            scale = np.bincount(np.searchsorted(bounds, self.last_idx, side="right"),
                                minlength=len(sizes)).argmax()
            key = self.layers[int(scale)]
            res["head_cls"] = dict(res[key], scale=key)
        return res


def read_pred_mask(model, img, i):
    pm, _, _ = predict(model, img)
    return pm[i]


def cam_metrics(cam, mask):
    if mask is None or mask.sum() == 0:
        return None
    m = mask.astype(bool)
    inside, outside = cam[m], cam[~m]
    gx, gy = cv2.Sobel(cam, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(cam, cv2.CV_64F, 0, 1, ksize=3)
    return dict(mask_coverage=float(inside.sum() / (cam.sum() + 1e-8)),
                activation_ratio=float(inside.mean() / (outside.mean() + 1e-8)) if outside.size else np.nan,
                smoothness=float(1 / (1 + np.sqrt(gx ** 2 + gy ** 2).mean() * 10)),
                spread=float(cam.var()))


def select_layer(model):
    """Old program mechanism, run on VALIDATION images (not the figure images)."""
    if CAM_LAYER != "auto":
        print(f"[LAYER] Using CAM_LAYER = {CAM_LAYER}")
        return CAM_LAYER, None
    if FIXED_LAYER is not None:
        print(f"[LAYER] Using FIXED_LAYER = {FIXED_LAYER}")
        return FIXED_LAYER, None
    val = images_with_labels(split_dir(PROC_DS, ("valid", "val")))
    random.Random(SAMPLE_SEED).shuffle(val)
    gc = YOLOGradCAM(model, EXPERIMENT_LAYERS)
    stats = {i: {k: [] for k in ["grad", "mask_coverage", "activation_ratio", "smoothness", "spread"]}
             for i in EXPERIMENT_LAYERS}
    n = 0
    for p, _ in val:
        if n >= N_LAYER_SELECT_IMG:
            break
        img = cv2.imread(str(p))
        pm, pc, ps = predict(model, img)
        if len(ps) == 0:
            continue
        top = int(ps.argmax())
        for idx, d in gc(img, int(pc[top]), pm[top]).items():
            if not d["valid"]:
                continue
            stats[idx]["grad"].append(d["grad"])
            m = cam_metrics(d["cam"], pm[top])
            if m:
                for k, v in m.items():
                    stats[idx][k].append(v)
        n += 1
    gc.remove()
    weights = dict(grad=0.20, mask_coverage=0.25, activation_ratio=0.15, smoothness=0.25, spread=0.15)
    means = {k: {i: np.mean(s[k]) for i, s in stats.items() if s[k]} for k in weights}
    score = {i: 0.0 for i in EXPERIMENT_LAYERS if stats[i]["grad"]}
    for k, w in weights.items():
        vals = means[k]
        if not vals:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        for i in score:
            if i in vals:
                score[i] += w * (vals[i] - lo) / (hi - lo if hi > lo else 1.0)
    best = max(score, key=score.get) if score else EXPERIMENT_LAYERS[2]
    print("[LAYER] Score per layer (validation):", {k: round(v, 3) for k, v in score.items()})
    print(f"[LAYER] Selected layer = {best}")
    return best, score


# =========================================================
# 6. FIGURE SAMPLE SELECTION
# =========================================================
def pick_internal(model):
    test = images_with_labels(split_dir(PROC_DS, ("test",)))
    rng = random.Random(SAMPLE_SEED)
    rows = []
    for c, name in enumerate(CLASS_NAMES):
        cands = [(p, lp) for p, lp in test
                 if any(int(float(l.split()[0])) == c for l in lp.read_text().splitlines() if l.strip())]
        if not cands:
            print(f"[WARN] internal test without class {name}"); continue
        p, lp = rng.choice(sorted(cands))
        img = cv2.imread(str(p))
        rows.append(dict(set="Internal test", rule=f"random {name} (seed {SAMPLE_SEED})",
                         path=p, label=lp, img=img, gt_cls=c))
    return rows


def pick_sidoarjo(model):
    """All Sidoarjo N34 instances -> outcome; pick one wrong->N23 (highest conf) & correct (highest conf)."""
    n34 = CLASS_NAMES.index("N34"); n23 = CLASS_NAMES.index("N23")
    recs = []
    for p, lp in images_with_labels(EXT_SDA):
        raw = cv2.imread(str(p)); img = adaptive_preprocess(raw)
        H, W = img.shape[:2]
        gts = read_gt(lp, H, W)
        if not any(c == n34 for c, _ in gts):
            continue
        pm, pc, ps = predict(model, img)
        mt = match(pm, pc, ps, gts)
        for j, (c, g) in enumerate(gts):
            if c != n34:
                continue
            if j in mt:
                i, iou = mt[j]
                outcome = "correct" if pc[i] == n34 else f"wrong->{CLASS_NAMES[pc[i]]}"
                recs.append(dict(path=p, label=lp, img=img, gt_idx=j, pred_idx=i, pred_cls=int(pc[i]),
                                 conf=float(ps[i]), iou=iou, outcome=outcome))
            else:
                recs.append(dict(path=p, label=lp, img=img, gt_idx=j, pred_idx=None, pred_cls=None,
                                 conf=np.nan, iou=np.nan, outcome="missed"))
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "img"} for r in recs])
    print("\n[SDA] Sidoarjo N34 instance outcomes:\n" + df.outcome.value_counts().to_string())
    rows = []
    for want, rule in [("wrong->N23", "N34 misclassified as N23, highest confidence"),
                       ("correct", "N34 correctly classified, highest confidence")]:
        sub = [r for r in recs if r["outcome"] == want]
        if not sub:
            print(f"[WARN] No '{want}' cases in Sidoarjo"); continue
        r = max(sub, key=lambda z: z["conf"])
        rows.append(dict(set="External – Sidoarjo", rule=rule, path=r["path"], label=r["label"],
                         img=r["img"], gt_cls=n34, forced_pred=r["pred_idx"], forced_gt=r["gt_idx"]))
    return rows, recs, df


# =========================================================
# 7. VISUAL
# =========================================================
def overlay_seg(img, mask, text):
    vis = img.copy()
    if mask is not None:
        m8 = mask.astype(np.uint8) * 255
        col = np.zeros_like(img); col[:, :, 1] = m8
        vis = cv2.addWeighted(vis, 0.7, col, 0.3, 0)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), max(2, img.shape[1] // 300))
    s = img.shape[1] / 640
    cv2.putText(vis, text, (int(15 * s), int(40 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s,
                (255, 255, 255), max(2, int(3 * s)), cv2.LINE_AA)
    return vis


def overlay_cam(img, cam):
    hm = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.5, hm, 0.5, 0)


# =========================================================
# 8. MAIN
# =========================================================
def main():
    global EXT_SDA
    mount_drive()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXT_SDA = extract_external()

    seed, seed_df = select_median_seed()
    weights = resolve_weights(seed)
    model = YOLO(str(weights), task="segment"); model.to(DEVICE)
    assert [model.names[i] for i in range(len(CLASS_NAMES))] == CLASS_NAMES, model.names
    print(f"[MODEL] {weights}")

    layer, layer_scores = select_layer(model)

    rows = pick_internal(model)
    sda_rows, sda_recs, sda_df = pick_sidoarjo(model)
    rows += sda_rows

    gc = YOLOGradCAM(model, layer)
    table = []
    for r in rows:
        img = r["img"]; H, W = img.shape[:2]
        gts = read_gt(r["label"], H, W)
        pm, pc, ps = predict(model, img)
        mt = match(pm, pc, ps, gts)
        if r.get("forced_pred") is not None:                       # external row: prediction matching GT N34
            i, j = r["forced_pred"], r["forced_gt"]
        else:                                                       # internal row: first GT of the target class
            j = next(k for k, (c, _) in enumerate(gts) if c == r["gt_cls"])
            i = mt[j][0] if j in mt else (int(ps.argmax()) if len(ps) else None)
        gt_mask = gts[j][1]
        if i is None:
            print(f"[WARN] {r['path'].name}: no prediction"); continue
        pred_mask, pred_cls, conf = pm[i], int(pc[i]), float(ps[i])
        iou = float(np.logical_and(pred_mask, gt_mask).sum() / max(np.logical_or(pred_mask, gt_mask).sum(), 1))
        cres = gc(img, pred_cls, pred_mask)[layer]                 # target = predicted class, this instance
        cam = cres["cam"]
        mg, mp = cam_metrics(cam, gt_mask), cam_metrics(cam, pred_mask)
        r.update(gt_mask=gt_mask, pred_mask=pred_mask, cam=cam, pred_cls=pred_cls, conf=conf, iou=iou)
        table.append(dict(row=len(table) + 1, set=r["set"], rule=r["rule"], file=r["path"].name,
                          gt=CLASS_NAMES[r["gt_cls"]], pred=CLASS_NAMES[pred_cls], conf=round(conf, 3),
                          cam_scale=cres.get("scale", str(layer)), leaf_frac=round(float(gt_mask.mean()), 3),
                          mask_iou=round(iou, 3),
                          cam_in_gt_leaf=round(mg["mask_coverage"], 3),
                          cam_ratio_in_out_gt=round(mg["activation_ratio"], 2)))
    gc.remove()
    rows = [r for r in rows if "cam" in r]

    # ---- CAM statistics for all Sidoarjo N34 (correct vs wrong), quantitative complement ----
    gc = YOLOGradCAM(model, layer)
    agg = []
    for rec in sda_recs:
        if rec["pred_idx"] is None:
            continue
        img = rec["img"]; H, W = img.shape[:2]
        gt_mask = read_gt(rec["label"], H, W)[rec["gt_idx"]][1]
        cam = gc(img, rec["pred_cls"], read_pred_mask(model, img, rec["pred_idx"]))[layer]["cam"]
        agg.append(dict(file=rec["path"].name, outcome=rec["outcome"], conf=rec["conf"],
                        cam_in_gt_leaf=cam_metrics(cam, gt_mask)["mask_coverage"]))
    gc.remove()
    agg = pd.DataFrame(agg)
    if len(agg):
        print("\n[SDA] Proportion of Grad-CAM activation inside the leaf (GT), per outcome:")
        print(agg.groupby("outcome").cam_in_gt_leaf.agg(["count", "mean", "std"]).round(3).to_string())

    # ---- figure 5 rows x 4 columns ----
    titles = ["Model input", "Ground-truth mask", "YOLO segmentation", f"Grad-CAM"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.75 * len(rows)))
    axes = np.atleast_2d(axes)
    for k, r in enumerate(rows):
        img = r["img"]
        gt_vis = cv2.cvtColor((r["gt_mask"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        seg = overlay_seg(img, r["pred_mask"], f"{CLASS_NAMES[r['pred_cls']]} {r['conf']:.2f}")
        panels = [img, gt_vis, seg, overlay_cam(img, r["cam"])]
        for c, pnl in enumerate(panels):
            ax = axes[k, c]
            ax.imshow(cv2.cvtColor(cv2.resize(pnl, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if k == 0:
                ax.set_title(titles[c], fontsize=FONT_PT)
        if r["set"].startswith("Internal"):
            lab = f"Internal test\nGT {CLASS_NAMES[r['gt_cls']]}"
        else:
            ok = "correct" if r["pred_cls"] == r["gt_cls"] else "misclassified"
            lab = f"Sidoarjo\nGT {CLASS_NAMES[r['gt_cls']]} ({ok})"
        axes[k, 0].set_ylabel(lab, fontsize=FONT_PT)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUTPUT_DIR / "fig15_gradcam.png", dpi=600)
    plt.close(fig)

    # ---- save details ----
    pd.DataFrame(table).to_csv(OUTPUT_DIR / "fig15_rows.csv", index=False)
    seed_df.to_csv(OUTPUT_DIR / "seed_validation.csv", index=False)
    sda_df.to_csv(OUTPUT_DIR / "sidoarjo_n34_outcomes.csv", index=False)
    if len(agg):
        agg.to_csv(OUTPUT_DIR / "sidoarjo_n34_cam_coverage.csv", index=False)
    json.dump(dict(weights=str(weights), seed=seed, layer=str(layer),
                   layer_scores={str(k): v for k, v in (layer_scores or {}).items()},
                   sample_seed=SAMPLE_SEED, conf=CONF_OP, nms_iou=NMS_IOU_OP, match_iou=MATCH_IOU,
                   cam_smoothing=dict(kernel=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA),
                   cam_target=f"sum of raw predicted-class logits of top-{TOPK_ANCHORS} anchors inside the explained predicted mask",
                   cam_norm_percentile=CAM_NORM_PERCENTILE),
              open(OUTPUT_DIR / "settings.json", "w"), indent=2)

    print("\n=== Fig. 15 rows ===")
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\n[DONE] Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()