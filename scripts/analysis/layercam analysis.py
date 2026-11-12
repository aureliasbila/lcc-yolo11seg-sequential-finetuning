# -*- coding: utf-8 -*-
"""
LayerCAM for YOLO11n-seg - Fig. 15 (selected configuration: Raw -> Processed)
============================================================================
Rows of the figure:
  1-3 : internal test (Dataset_Gamma_New/test), one instance per class (N23 / N34 / N45)
  4   : external Sidoarjo, N34 leaf misclassified as N23
  5   : external Sidoarjo, N34 leaf correctly classified
  (SELECTION_MODE="median": the example is the instance whose activation-in-leaf proportion is closest
   to the MEDIAN of its group, computed over ALL detected instances - not the best-looking one.)
Columns: model input | ground-truth mask | YOLO segmentation | LayerCAM

COVERAGE
  A CAM explains an existing detection, so maps are generated for every ground-truth instance that the
  model detected (greedy matching, mask IoU >= MATCH_IOU). Ground-truth instances without a matching
  prediction have nothing to explain; they are counted and listed ([COVERAGE] lines and
  internal_instances_not_detected.csv), so that the number of maps can be compared with the number of
  annotated instances (internal test: 67 images, 68 instances).

OUTPUT : saved locally (OUTPUT_DIR) and as <OUTPUT_DIR>.zip.
DATA   : processed internal dataset (Dataset_Gamma_New) and the external zip (Sidoarjo folder).

CAM METHOD - instance-specific LayerCAM for YOLO-seg
  - Layer  : neck P3 output (layer 16 in YOLO11-seg, stride 8), the highest-resolution feature map used by
             all output branches (class, mask coefficients, mask prototypes). The layer index is read
             automatically from the head (head.f[0]).
  - Target : (a) class : sum_a sigmoid(z_a) * z_a over the candidate anchors that greedy NMS merges into the
                         explained detection (replica of Ultralytics' greedy NMS; z = logit of the predicted class)
             (b) mask  : mean instance-mask logit (prototypes x coefficients of the anchor kept by NMS)
                         inside the predicted mask (Seg-Grad-CAM style, Vinogradova et al., 2020)
             Maps (a) and (b) are normalised and averaged ("cls+seg"); "cls" alone can also be chosen.
  - Weights: LayerCAM (Jiang et al., 2021): sum_k ReLU(dS/dA_k(x,y)) * A_k(x,y) per location, so that every
             part of the leaf that raises the score is highlighted (not only the neighbourhood of the anchor).
  - Input  : letterbox identical to model.predict; consistency is checked ([CHECK] box IoU).
  - The heat map is not masked with the leaf mask: any activation outside the leaf is a genuine signal.

Model  : stage2_raw_to_processed_s{seed}/weights/best.pt
Seed   : chosen automatically = the seed whose VALIDATION mAP@0.5:0.95 is closest to the median of 3 seeds
Input  : the models were trained on processed images -> the internal set uses Dataset_Gamma_New; the external
         Sidoarjo images receive the same BO preprocessing (Table 6) before prediction.

Requirements: pip install ultralytics  (the versions used are written to settings.json)
"""

import json, random, shutil, warnings
from pathlib import Path
import cv2, numpy as np, pandas as pd, yaml, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURATION (edit the paths for your environment)
# =========================================================
DRIVE        = Path("/content/drive/MyDrive")
DATASET_DIR  = DRIVE / "dataset"
PROC_DS      = DATASET_DIR / "Dataset_Gamma_New"           # internal (processed) train/valid/test
EXTERNAL_ZIP = Path("/content/New_Data_External.zip")     # external zip used for Tables 16/17
EXTERNAL_SRC = Path("/content/tmp_layercam_external")       # extraction target
EXT_SITE     = "New_Sda"                                   # Sidoarjo folder inside the zip
EXT_SDA      = None                                        # set automatically after extraction
RUNS_ROOTS   = [Path("/content/runs/ablation"), DRIVE / "ablation_backup" / "runs"]
RUN_TEMPLATE = "stage2_raw_to_processed_s{seed}"           # selected configuration
SEEDS        = [0, 1, 2]
OUTPUT_DIR   = Path("/content/layercam_fig15")             # local output folder

CLASS_NAMES = ["N23", "N34", "N45"]
IMG_SIZE    = 640
STRIDE      = 32
LETTERBOX_AUTO = True        # same as model.predict (rect=True, single image) -> minimum rectangle
CONF_OP, NMS_IOU_OP, MATCH_IOU = 0.25, 0.45, 0.50         # same as the external evaluation
SAMPLE_SEED = 42
SELECTION_MODE = "median"    # "median" (representative) | "random" (random internal, highest-confidence Sidoarjo)
PREP = dict(gamma_min=0.7825, gamma_max=1.2026, gamma_target_mean=0.4835,
            edge_blur_sigma=0.6074, edge_strength=0.0469, hsv_saturation_gain=1.0171)

# ---- CAM ----
CAM_LAYERS    = "auto"       # "auto" = neck P3 output (head.f[0], layer 16 in YOLO11-seg); or a list, e.g. [16] / [13, 16]
CAM_TARGET    = "cls+seg"    # "cls+seg" (class + instance mask) | "cls" (class only) | "seg" (mask only)
CAM_WEIGHTING = "layercam"   # "layercam" (default) | "hires" (signed gradient x activation) | "gradcam" (global-average weights)
ANCHOR_IOU    = NMS_IOU_OP   # IoU threshold of the NMS replica for the anchors that belong to a detection
CAM_BLUR_SIGMA = 4.0         # Gaussian blur (pixels in the 640 input space); 0 = no blur
CAM_NORM_PERCENTILE = 99.0   # heat map is normalised at this percentile (not at the maximum)
CAM_ALPHA     = 0.5          # overlay transparency of the heat map

FONT_PT = 10
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
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
    """Extract the external zip and return the <EXT_SITE>/test folder (containing images/ and labels/)."""
    import zipfile
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


def load_image(path, external):
    """Model input image: the internal set is already processed; external images receive the BO preprocessing."""
    img = cv2.imread(str(path))
    return adaptive_preprocess(img) if external else img


def split_dir(root, names):
    for n in names:
        if (root / n / "images").exists():
            return root / n
    raise FileNotFoundError(f"split {names} not found in {root}")


def images_with_labels(root):
    """All images below root (recursive) with their label file (.../labels/<stem>.txt)."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.parent.name == "images":
            lp = p.parent.parent / "labels" / (p.stem + ".txt")
            if lp.exists():
                out.append((p, lp))
    return out


def read_gt(lp, H, W):
    """List of (class_id, boolean mask) read from a YOLO-seg label file."""
    gts = []
    for line in Path(lp).read_text().splitlines():
        v = line.split()
        if len(v) < 7 or (len(v) - 1) % 2:
            continue
        pts = np.clip(np.array(v[1:], np.float32).reshape(-1, 2) * [W, H], 0, [W - 1, H - 1]).astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        gts.append((int(float(v[0])), m.astype(bool)))
    return gts


# =========================================================
# 3. MODEL AND SEED (median validation score)
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
    yml = Path("/content/tmp_layercam_val.yaml")
    yaml.safe_dump({"path": "/", "train": str(val_dir / "images"), "val": str(val_dir / "images"),
                    "names": {i: n for i, n in enumerate(CLASS_NAMES)}}, open(yml, "w"))
    rows = []
    for s in SEEDS:
        m = YOLO(str(resolve_weights(s)), task="segment").val(
            data=str(yml), split="val", imgsz=IMG_SIZE, batch=16, plots=False, verbose=False,
            project="/content/tmp_layercam", name=f"val_s{s}", exist_ok=True)
        rows.append(dict(seed=s, mAP50=float(m.seg.map50), mAP50_95=float(m.seg.map)))
    df = pd.DataFrame(rows)
    med = df.mAP50_95.median()
    seed = int(df.iloc[(df.mAP50_95 - med).abs().argsort().iloc[0]].seed)
    print("\n[SEED] Validation per seed (processed validation set):\n" + df.to_string(index=False))
    print(f"[SEED] Median mAP@0.5:0.95 = {med:.4f} -> selected seed = {seed}")
    return seed, df


# =========================================================
# 4. PREDICTION AND MATCHING (same rules as the external evaluation)
# =========================================================
def predict(model, img):
    """Return (masks [n,H,W] bool, boxes xyxy [n,4] in original coordinates, classes [n], confidences [n])."""
    r = model.predict(img, imgsz=IMG_SIZE, conf=CONF_OP, iou=NMS_IOU_OP,
                      retina_masks=True, verbose=False, device=DEVICE)[0]
    H, W = img.shape[:2]
    if r.masks is None or len(r.boxes) == 0:
        return np.zeros((0, H, W), bool), np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
    pm = r.masks.data.cpu().numpy() > 0.5
    if pm.shape[1:] != (H, W):
        pm = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for m in pm])
    return (pm, r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy())


def match(pm, pc, ps, gts):
    """Greedy matching (confidence descending, mask IoU >= MATCH_IOU); the class is NOT used for matching.
    Returns {gt_index: (pred_index, iou)}; ground-truth instances without a match are absent."""
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
# 5. INSTANCE-SPECIFIC LAYERCAM (class + mask target)
# =========================================================
def letterbox(img):
    """Replica of the Ultralytics LetterBox used by model.predict. Returns the image and the inverse parameters."""
    h, w = img.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = IMG_SIZE - nw, IMG_SIZE - nh
    if LETTERBOX_AUTO:
        dw, dh = dw % STRIDE, dh % STRIDE
    dw, dh = dw / 2, dh / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, dict(r=r, left=left, top=top, nw=nw, nh=nh)


def box_iou_1toN(box, boxes):
    """IoU of one xyxy box against N xyxy boxes (numpy)."""
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def normalize_cam(cam):
    hi = np.percentile(cam, CAM_NORM_PERCENTILE)
    hi = hi if hi > 0 else cam.max()
    return np.clip(cam / hi, 0, 1) if hi > 0 else cam


class InstanceCAM:
    """Instance-specific LayerCAM for YOLO-seg.

    Class target  S_cls = sum_a sigmoid(z_{c,a}) * z_{c,a}, over the pre-NMS candidates a that greedy NMS merges
                  into the explained detection (argmax class = c, conf >= CONF_OP, IoU >= ANCHOR_IOU, not
                  suppressed earlier by a stronger detection of the same class - replica of greedy NMS).
    Mask target   S_seg = mean instance-mask logit (prototypes x coefficients of the anchor kept by NMS)
                  over the pixels inside the predicted mask.
    Map           M(x,y) = sum_k ReLU(dS/dA_k(x,y)) * A_k(x,y)  (LayerCAM), ReLU, normalised per target,
                  then averaged over the targets.
    """

    def __init__(self, weights, layers=CAM_LAYERS, target=CAM_TARGET, weighting=CAM_WEIGHTING):
        self.net = YOLO(str(weights), task="segment").model.float().to(DEVICE).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.head = self.net.model[-1]
        self.nc, self.nm = int(self.head.nc), int(getattr(self.head, "nm", 32))
        self.layers = [int(self.head.f[0])] if layers == "auto" else ([layers] if isinstance(layers, int) else list(layers))
        self.targets = [t for t in ("cls", "seg") if t in target]
        self.weighting = weighting
        self.feats, self.logits, self.hooks = {}, {}, []
        for i, br in enumerate(self.head.cv3):
            self.hooks.append(br.register_forward_hook(self._save_logits(i)))
        for li in self.layers:
            self.hooks.append(self.net.model[li].register_forward_hook(self._save_out(li)))

    def _save_logits(self, i):
        def h(mod, inp, out):
            self.logits[i] = out
        return h

    def _save_out(self, key):
        def h(mod, inp, out):
            self.feats[key] = out if isinstance(out, torch.Tensor) else out[0]
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()

    @staticmethod
    def _tensors(o):
        if isinstance(o, torch.Tensor):
            return [o]
        if isinstance(o, (list, tuple)):
            return [t for x in o for t in InstanceCAM._tensors(x)]
        if isinstance(o, dict):
            return [t for x in o.values() for t in InstanceCAM._tensors(x)]
        return []

    def _owner(self, z, pred, P, pb, pc, ps, H, W):
        """Replica of per-class greedy NMS: owner[a] = index of the final detection that 'owns' anchor a."""
        prob_all = torch.sigmoid(z).detach().cpu().numpy()
        top_cls = prob_all.argmax(0)
        xywh = pred[:4].detach().cpu().numpy().T
        xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
        xyxy = (xyxy - [P["left"], P["top"], P["left"], P["top"]]) / P["r"]     # original coordinates, not clipped
        xyxy_clip = np.clip(xyxy, 0, [W, H, W, H])                              # Ultralytics clips the final boxes
        owner, kept = np.full(z.shape[1], -1), {}
        for c in np.unique(pc):
            prob = prob_all[c]
            cand = (top_cls == c) & (prob >= CONF_OP)
            for d in [d for d in np.argsort(-np.asarray(ps)) if int(pc[d]) == c]:
                iou_d = box_iou_1toN(np.asarray(pb[d], float), xyxy_clip)
                same = cand & (np.abs(prob - float(ps[d])) < 2e-3)
                k = int(np.argmax(np.where(same, iou_d, -1))) if same.any() else int(np.argmax(iou_d))
                kept[d] = (k, float(iou_d[k]))
                sup = cand & (owner < 0) & (prob <= prob[k] + 1e-6) & (box_iou_1toN(xyxy[k], xyxy) > ANCHOR_IOU)
                owner[sup] = d
                owner[k] = d
        return owner, kept

    def __call__(self, img_bgr, pb, pc, ps, i, mask):
        """img_bgr: model input image; pb/pc/ps: all final detections of this image (original xyxy boxes,
        classes, confidences); i: index of the detection to explain; mask: its predicted mask (original resolution)."""
        H, W = img_bgr.shape[:2]
        c = int(pc[i])
        lb, P = letterbox(img_bgr)
        x = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).contiguous().to(DEVICE).requires_grad_(True)
        Hin, Win = x.shape[2:]
        self.feats.clear(); self.logits.clear()
        out = self.net(x)

        ts = self._tensors(out)
        pred = [t for t in ts if t.dim() == 3 and t.shape[1] == 4 + self.nc + self.nm][0][0]   # (4+nc+nm, N)
        proto = [t for t in ts if t.dim() == 4 and t.shape[1] == self.nm][0][0]               # (nm, h/4, w/4)
        z = torch.cat([self.logits[k][0].flatten(1) for k in range(len(self.head.cv3))], 1)   # logits (nc, N)
        assert z.shape[1] == pred.shape[1], "anchor order of the logits does not match the predictions"

        owner, kept = self._owner(z, pred, P, pb, pc, ps, H, W)
        k_i, check_iou = kept[i]
        idx = torch.from_numpy(np.nonzero(owner == i)[0]).to(z.device)

        scores = {}
        if "cls" in self.targets:
            scores["cls"] = (torch.sigmoid(z[c, idx]).detach() * z[c, idx]).sum()
        if "seg" in self.targets:
            nm, ph, pw = proto.shape
            mlog = (pred[4 + self.nc:4 + self.nc + nm, k_i] @ proto.reshape(nm, -1)).reshape(ph, pw)
            m_lb = np.zeros((Hin, Win), np.float32)
            m_lb[P["top"]:P["top"] + P["nh"], P["left"]:P["left"] + P["nw"]] = \
                cv2.resize(mask.astype(np.float32), (P["nw"], P["nh"]), interpolation=cv2.INTER_LINEAR)
            m_p = cv2.resize(m_lb, (pw, ph), interpolation=cv2.INTER_AREA) > 0.5
            if m_p.any():
                scores["seg"] = mlog[torch.from_numpy(m_p).to(mlog.device)].mean()

        keys = list(self.feats)
        maps = {}
        for name, sc in scores.items():
            grads = torch.autograd.grad(sc, [self.feats[k] for k in keys], retain_graph=True)
            total = torch.zeros(Hin, Win, device=x.device)
            for k, g in zip(keys, grads):
                a = self.feats[k].detach()
                if self.weighting == "layercam":
                    m = (g.clamp(min=0) * a).sum(1, keepdim=True)
                elif self.weighting == "hires":
                    m = (g * a).sum(1, keepdim=True)
                else:                                                            # classic Grad-CAM
                    m = (g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True)
                s = Hin / a.shape[2]
                total += F.interpolate(m, size=(Hin, Win), mode="bilinear", align_corners=False)[0, 0] / s ** 2
            cam = F.relu(total).cpu().numpy().astype(np.float32)
            if CAM_BLUR_SIGMA > 0:
                cam = cv2.GaussianBlur(cam, (0, 0), CAM_BLUR_SIGMA)
            cam = cam[P["top"]:P["top"] + P["nh"], P["left"]:P["left"] + P["nw"]]   # remove the letterbox padding
            maps[name] = normalize_cam(cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR))
        cam = normalize_cam(np.mean(list(maps.values()), 0))
        return dict(cam=cam, maps=maps, n_anchors=int(len(idx)), check_iou=check_iou)


def cam_metrics(cam, mask):
    """mask_coverage = share of the CAM energy inside the mask (energy-based pointing game)."""
    if mask is None or mask.sum() == 0:
        return None
    m = mask.astype(bool)
    inside, outside = cam[m], cam[~m]
    return dict(mask_coverage=float(inside.sum() / (cam.sum() + 1e-8)),
                activation_ratio=float(inside.mean() / (outside.mean() + 1e-8)) if outside.size else np.nan)


# =========================================================
# 6. SELECTION OF THE FIGURE SAMPLES
# =========================================================
def collect_sidoarjo(model):
    """All N34 instances of Sidoarjo -> outcome (correct / wrong->X / missed)."""
    n34 = CLASS_NAMES.index("N34")
    recs = []
    for p, lp in images_with_labels(EXT_SDA):
        img = load_image(p, external=True)
        H, W = img.shape[:2]
        gts = read_gt(lp, H, W)
        if not any(c == n34 for c, _ in gts):
            continue
        pm, pb, pc, ps = predict(model, img)
        mt = match(pm, pc, ps, gts)
        for j, (c, _) in enumerate(gts):
            if c != n34:
                continue
            if j in mt:
                i, iou = mt[j]
                outcome = "correct" if pc[i] == n34 else f"wrong->{CLASS_NAMES[pc[i]]}"
                recs.append(dict(path=p, label=lp, gt_idx=j, pred_idx=i, pred_cls=int(pc[i]),
                                 conf=float(ps[i]), iou=iou, outcome=outcome))
            else:
                recs.append(dict(path=p, label=lp, gt_idx=j, pred_idx=None, pred_cls=None,
                                 conf=np.nan, iou=np.nan, outcome="missed"))
    df = pd.DataFrame(recs)
    print("\n[SDA] Outcome of the N34 instances of Sidoarjo:\n" + df.outcome.value_counts().to_string())
    return recs, df


def audit_all(model, cammer, sda_recs):
    """LayerCAM for ALL detected internal-test instances and ALL detected Sidoarjo N34 instances.
    Also counts the annotated instances and lists those without a matching prediction."""
    rows = []

    def one(set_name, group, p, lp, img, gts, j, det, i, gt_cls, outcome):
        pm, pb, pc, ps = det
        pred_cls, conf = int(pc[i]), float(ps[i])
        gt_mask = gts[j][1]
        res = cammer(img, pb, pc, ps, i, mask=pm[i])
        m = cam_metrics(res["cam"], gt_mask)
        rows.append(dict(set=set_name, group=group, file=p.name, path=str(p), label=str(lp), gt_idx=j,
                         pred_idx=i, gt=CLASS_NAMES[gt_cls], pred=CLASS_NAMES[pred_cls], conf=conf,
                         outcome=outcome, leaf_frac=float(gt_mask.mean()),
                         cam_in_leaf=m["mask_coverage"], cam_ratio=m["activation_ratio"],
                         **{f"cam_in_leaf_{k}": cam_metrics(v, gt_mask)["mask_coverage"] for k, v in res["maps"].items()},
                         n_anchors=res["n_anchors"], check_iou=res["check_iou"],
                         degenerate=bool(res["cam"].max() <= 0)))

    # ---- internal test ----
    test = images_with_labels(split_dir(PROC_DS, ("test",)))
    n_gt, not_detected = {c: 0 for c in CLASS_NAMES}, []
    for n, (p, lp) in enumerate(test, 1):
        img = load_image(p, external=False); H, W = img.shape[:2]
        gts = read_gt(lp, H, W)
        det = predict(model, img)
        pm, pb, pc, ps = det
        mt = match(pm, pc, ps, gts)
        for j, (c, _) in enumerate(gts):                       # count every annotated instance
            n_gt[CLASS_NAMES[c]] += 1
            if j not in mt:                                    # no prediction with mask IoU >= MATCH_IOU
                not_detected.append(dict(file=p.name, gt_idx=j, gt=CLASS_NAMES[c]))
        for j, (i, _) in mt.items():
            c = gts[j][0]
            one("Internal test", f"Internal-{CLASS_NAMES[c]}", p, lp, img, gts, j, det, i, c,
                "correct" if pc[i] == c else f"wrong->{CLASS_NAMES[pc[i]]}")
        if n % 25 == 0:
            print(f"  [AUDIT] internal {n}/{len(test)} images")
    print(f"\n[COVERAGE] internal-test annotated instances: {n_gt} "
          f"(total {sum(n_gt.values())} in {len(test)} images)")
    print(f"[COVERAGE] without a matching prediction (mask IoU >= {MATCH_IOU}): {len(not_detected)} -> {not_detected}")
    pd.DataFrame(not_detected, columns=["file", "gt_idx", "gt"]).to_csv(
        OUTPUT_DIR / "internal_instances_not_detected.csv", index=False)

    # ---- external Sidoarjo (N34) ----
    cur = None
    for r in sda_recs:
        if r["pred_idx"] is None:
            continue
        if cur != r["path"]:                                    # instances of one image are consecutive
            cur = r["path"]
            img = load_image(cur, external=True); H, W = img.shape[:2]
            gts = read_gt(r["label"], H, W)
            det = predict(model, img)
        one("External - Sidoarjo", f"Sidoarjo-N34-{r['outcome']}", r["path"], r["label"], img, gts,
            r["gt_idx"], det, r["pred_idx"], CLASS_NAMES.index("N34"), r["outcome"])

    df = pd.DataFrame(rows)
    df["cam_vs_area"] = df.cam_in_leaf / df.leaf_frac.clip(lower=1e-6)
    summ = df.groupby("group").agg(n=("file", "size"), leaf_frac=("leaf_frac", "mean"),
                                   cam_in_leaf_mean=("cam_in_leaf", "mean"), cam_in_leaf_sd=("cam_in_leaf", "std"),
                                   cam_in_leaf_median=("cam_in_leaf", "median"),
                                   cam_vs_area_median=("cam_vs_area", "median"),
                                   pct_degenerate=("degenerate", lambda s: 100 * s.mean())).round(3)
    print("\n[AUDIT] LayerCAM for ALL detected instances (cam_vs_area > 1 = leaning towards the leaf):\n" + summ.to_string())

    # coverage per internal class: annotated vs analysed
    cov = pd.DataFrame({"annotated": pd.Series(n_gt)})
    cov["analysed"] = [int(((df["set"] == "Internal test") & (df["gt"] == c)).sum()) for c in cov.index]
    cov["not_detected"] = cov.annotated - cov.analysed
    cov.loc["Total"] = cov.sum()
    print("\n[COVERAGE] internal test, per ground-truth class:\n" + cov.to_string())
    cov.to_csv(OUTPUT_DIR / "internal_coverage.csv")

    bad = df[df.check_iou < 0.9]
    if len(bad):
        print(f"[CHECK] {len(bad)} instances: anchor box of the CAM vs predicted box IoU < 0.9 "
              f"(check LETTERBOX_AUTO / ultralytics version). Median IoU = {df.check_iou.median():.3f}")
    else:
        print(f"[CHECK] CAM forward pass consistent with model.predict (median box IoU = {df.check_iou.median():.3f})")
    return df, summ


GROUP_ORDER = [f"Internal-{c}" for c in CLASS_NAMES] + ["Sidoarjo-N34-wrong->N23", "Sidoarjo-N34-correct"]


def pick_median(df):
    """One example per group: the instance whose cam_in_leaf is closest to the group median (all detected instances)."""
    rows = []
    for g in GROUP_ORDER:
        sub = df[df.group == g]
        if sub.empty:
            print(f"[WARN] group {g} is empty"); continue
        r = sub.iloc[(sub.cam_in_leaf - sub.cam_in_leaf.median()).abs().argsort().iloc[0]]
        rows.append(dict(set=r["set"], rule=f"{g}: closest to group median activation-in-leaf (n={len(sub)})",
                         path=Path(r["path"]), label=Path(r["label"]), gt_cls=CLASS_NAMES.index(r["gt"]),
                         forced_pred=int(r["pred_idx"]), forced_gt=int(r["gt_idx"])))
    return rows


def pick_random(df):
    """Alternative: internal = random instance per class (fixed seed); Sidoarjo = highest confidence."""
    rng = random.Random(SAMPLE_SEED)
    rows = []
    for g in GROUP_ORDER:
        sub = df[df.group == g]
        if sub.empty:
            print(f"[WARN] group {g} is empty"); continue
        if g.startswith("Internal"):
            r = sub.sort_values(["file", "gt_idx"]).iloc[rng.randrange(len(sub))]
            rule = f"{g}: random instance (seed {SAMPLE_SEED})"
        else:
            r = sub.loc[sub.conf.idxmax()]
            rule = f"{g}: highest confidence"
        rows.append(dict(set=r["set"], rule=rule, path=Path(r["path"]), label=Path(r["label"]),
                         gt_cls=CLASS_NAMES.index(r["gt"]), forced_pred=int(r["pred_idx"]),
                         forced_gt=int(r["gt_idx"])))
    return rows


# =========================================================
# 7. VISUALS
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
    return cv2.addWeighted(img, 1 - CAM_ALPHA, hm, CAM_ALPHA, 0)


# =========================================================
# 8. MAIN
# =========================================================
def main():
    global EXT_SDA
    mount_drive()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXT_SDA = extract_external()
    print(f"[ENV] ultralytics {ultralytics.__version__} | torch {torch.__version__} | device {DEVICE}")

    seed, seed_df = select_median_seed()
    weights = resolve_weights(seed)
    model = YOLO(str(weights), task="segment"); model.to(DEVICE)
    assert [model.names[i] for i in range(len(CLASS_NAMES))] == CLASS_NAMES, model.names
    print(f"[MODEL] {weights}")
    cammer = InstanceCAM(weights)
    print(f"[CAM] layer = {cammer.layers} | target = {CAM_TARGET} | weighting = {CAM_WEIGHTING}")
    sda_recs, sda_df = collect_sidoarjo(model)
    all_df, all_summ = audit_all(model, cammer, sda_recs)
    all_df.drop(columns=["path", "label"]).to_csv(OUTPUT_DIR / "cam_all_instances.csv", index=False)
    all_summ.to_csv(OUTPUT_DIR / "cam_summary_by_group.csv")

    rows = pick_median(all_df) if SELECTION_MODE == "median" else pick_random(all_df)
    table = []
    for r in rows:
        img = load_image(r["path"], external=not r["set"].startswith("Internal"))
        H, W = img.shape[:2]
        gts = read_gt(r["label"], H, W)
        pm, pb, pc, ps = predict(model, img)
        i, j = r["forced_pred"], r["forced_gt"]
        gt_mask = gts[j][1]
        pred_mask, pred_cls, conf = pm[i], int(pc[i]), float(ps[i])
        iou = float(np.logical_and(pred_mask, gt_mask).sum() / max(np.logical_or(pred_mask, gt_mask).sum(), 1))
        res = cammer(img, pb, pc, ps, i, mask=pred_mask)           # target = predicted class of this instance
        mg = cam_metrics(res["cam"], gt_mask)
        r.update(img=img, gt_mask=gt_mask, pred_mask=pred_mask, cam=res["cam"], pred_cls=pred_cls, conf=conf)
        table.append(dict(row=len(table) + 1, set=r["set"], rule=r["rule"], file=r["path"].name,
                          gt=CLASS_NAMES[r["gt_cls"]], pred=CLASS_NAMES[pred_cls], conf=round(conf, 3),
                          n_target_anchors=res["n_anchors"],
                          leaf_frac=round(float(gt_mask.mean()), 3), mask_iou=round(iou, 3),
                          cam_in_gt_leaf=round(mg["mask_coverage"], 3),
                          cam_ratio_in_out_gt=round(mg["activation_ratio"], 2),
                          **{f"cam_in_gt_leaf_{k}": round(cam_metrics(v, gt_mask)["mask_coverage"], 3)
                             for k, v in res["maps"].items()}))
    layers_used = cammer.layers
    cammer.remove()

    # ---- CAM statistics of all Sidoarjo N34 instances (correct vs wrong) ----
    sda_cam = all_df[all_df.set != "Internal test"][["file", "outcome", "conf", "cam_in_leaf"]]
    if len(sda_cam):
        print("\n[SDA] Share of LayerCAM activation inside the (ground-truth) leaf, per outcome:")
        print(sda_cam.groupby("outcome").cam_in_leaf.agg(["count", "mean", "std"]).round(3).to_string())

    # ---- figure: 5 rows x 4 columns ----
    titles = ["Model input", "Ground-truth mask", "YOLO segmentation", "LayerCAM"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.75 * len(rows)))
    axes = np.atleast_2d(axes)
    for k, r in enumerate(rows):
        img = r["img"]
        gt_vis = cv2.cvtColor((r["gt_mask"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        seg = overlay_seg(img, r["pred_mask"], CLASS_NAMES[r["pred_cls"]])     # class label only, no confidence
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
    fig.savefig(OUTPUT_DIR / "fig15_layercam.png", dpi=600)
    plt.close(fig)

    # ---- save details ----
    pd.DataFrame(table).to_csv(OUTPUT_DIR / "fig15_rows.csv", index=False)
    seed_df.to_csv(OUTPUT_DIR / "seed_validation.csv", index=False)
    sda_df.drop(columns=["path", "label"]).assign(file=[p.name for p in sda_df.path]) \
          .to_csv(OUTPUT_DIR / "sidoarjo_n34_outcomes.csv", index=False)
    if len(sda_cam):
        sda_cam.to_csv(OUTPUT_DIR / "sidoarjo_n34_cam_coverage.csv", index=False)
    json.dump(dict(weights=str(weights), seed=seed, cam_layers=layers_used, cam_weighting=CAM_WEIGHTING,
                   ultralytics=ultralytics.__version__, torch=torch.__version__,
                   cam_method="instance-specific LayerCAM (sum_k ReLU(dS/dA_k) * A_k per location) on the neck P3 "
                              "output; letterbox identical to model.predict; no mask applied to the CAM",
                   cam_target=f"{CAM_TARGET}: cls = sum over the pre-NMS candidates that greedy NMS merges into the "
                              f"explained detection (argmax class, conf>={CONF_OP}, box IoU>{ANCHOR_IOU}) of "
                              f"sigmoid(z)*z with z the raw class logit; seg = mean instance-mask logit "
                              f"(prototypes x coefficients of the NMS-kept anchor) inside the predicted mask; "
                              f"per-target maps percentile-normalised and averaged",
                   coverage="maps are generated for ground-truth instances matched to a prediction "
                            f"(mask IoU >= {MATCH_IOU}); see internal_coverage.csv and "
                            f"internal_instances_not_detected.csv",
                   cam_blur_sigma_px640=CAM_BLUR_SIGMA, cam_norm_percentile=CAM_NORM_PERCENTILE,
                   sample_seed=SAMPLE_SEED, selection_mode=SELECTION_MODE, conf=CONF_OP,
                   nms_iou=NMS_IOU_OP, match_iou=MATCH_IOU),
              open(OUTPUT_DIR / "settings.json", "w"), indent=2)

    print("\n=== Fig. 15 rows ===")
    print(pd.DataFrame(table).to_string(index=False))
    zp = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR)
    print(f"\n[DONE] Output in {OUTPUT_DIR}  |  archive: {zp}")
    try:
        from IPython.display import Image, display
        display(Image(filename=str(OUTPUT_DIR / "fig15_layercam.png"), width=700))
    except Exception:
        pass


if __name__ == "__main__":
    main()