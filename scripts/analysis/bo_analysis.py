# NOTE: some comments and console messages in this script are written in Indonesian.
# Edit the CONFIG block (paths) before running. Designed for Google Colab with Google Drive mounted.
# ==========================================================================
#  ANALISIS ULANG HASIL BAYESIAN OPTIMIZATION (R1.8 & R1.10) - TANPA TRAINING
#  Satu cell Colab. Membaca optuna_study.db + folder trial yang SUDAH ADA.
#
#  Menghasilkan:
#   1. bo_trials.csv               : catatan lengkap semua trial (untuk GitHub/R1.10)
#   2. Ringkasan protokol aktual   : sampler, trial acak vs TPE, jumlah trial selesai,
#                                    optimizer & learning rate yang BENAR-BENAR dipakai
#   3. Rentang mAP antar-trial vs variasi antar-seed (apakah BO memilih di atas noise?)
#   4. Importance parameter + STABILITAS (bootstrap) untuk J, mAP, dan Delta-E terpisah
#   5. Delta-E dihitung ulang untuk parameter terbaik pada SELURUH validation & per kelas
#   6. Figure siap jurnal (matplotlib, font >= 10 pt, satu panel, tanpa label (a)/(b))
#   7. Teks §3.2 dan tabel pengaturan BO dengan angka terisi otomatis
# ==========================================================================

# (Colab) !pip install -q optuna scikit-learn scikit-image pyyaml

import os, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import optuna
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------- KONFIGURASI ----------------
MOUNT_DRIVE   = True
DRIVE_ROOT    = Path("/content/drive/MyDrive/bo_preprocessing_yolo")
STUDY_NAME    = "bo_preprocessing_yolo11_seg"
RUNS_DIR      = DRIVE_ROOT / "runs"
DATASET_ROOT  = Path("/content/drive/MyDrive/dataset/Dataset_Roboflow")   # untuk Delta-E ulang
OUT_DIR       = DRIVE_ROOT / "analisis_revisi"

LAMBDA_COLOR      = 0.01
N_STARTUP_DEFAULT = 10      # default TPESampler Optuna (program BO tidak mengubahnya)
DELTAE_REF_N      = 30      # program BO memakai 30 gambar valid PERTAMA (urut abjad)
N_BOOT            = 200     # bootstrap untuk stabilitas importance
# Variasi antar-seed mAP@0.5:0.95 (validation) dari Table 9 Anda, sebagai patokan noise
SEED_SD_REF = {"Raw (Table 9)": 0.0024, "Proposed (Table 9)": 0.0010}
RECOMPUTE_DELTAE  = True
FULL_EPOCHS       = 60      # trial dgn epoch < ini (mis. uji TEST_MODE 5 epoch) dikeluarkan dari analisis
# ---------------------------------------------

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("[info]", e)
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 10, "legend.fontsize": 9.5,
                     "font.family": "serif", "savefig.dpi": 300})

PLAIN_LABEL = {"gamma_min": "gamma_min", "gamma_max": "gamma_max", "gamma_target_mean": "the target mean intensity",
               "edge_blur_sigma": "the Gaussian blur sigma", "edge_strength": "the edge strength",
               "hsv_saturation_gain": "the saturation gain"}
PARAM_LABEL = {
    "gamma_min": r"$\gamma_{min}$", "gamma_max": r"$\gamma_{max}$",
    "gamma_target_mean": "Target mean $T$", "edge_blur_sigma": r"Blur $\sigma$",
    "edge_strength": r"Edge strength $\alpha$", "hsv_saturation_gain": "Saturation gain $g_s$",
}

# ==========================================================================
# 1. Muat study & susun tabel trial
# ==========================================================================
storage = f"sqlite:///{DRIVE_ROOT / 'optuna_study.db'}"
study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
print(f"Optuna terpasang sekarang: {optuna.__version__}")

def read_trial_run(num):
    d = RUNS_DIR / f"trial_{num}"
    info = {}
    a = d / "args.yaml"
    if a.exists():
        cfg = yaml.safe_load(open(a))
        for k in ["optimizer", "lr0", "epochs", "patience", "batch", "imgsz", "seed",
                  "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "degrees", "mosaic", "scale", "translate"]:
            info[f"arg_{k}"] = cfg.get(k)
    r = d / "results.csv"
    if r.exists():
        res = pd.read_csv(r); res.columns = [c.strip() for c in res.columns]
        info["epochs_run"] = int(res["epoch"].max()) if "epoch" in res else len(res)
        if "lr/pg0" in res:
            info["lr_actual_max"] = float(res["lr/pg0"].max())     # puncak lr = lr0 efektif
        for col in res.columns:
            if col.startswith("metrics/mAP50-95(M)"):
                info["best_epoch"] = int(res.loc[res[col].idxmax(), "epoch"]) if "epoch" in res else None
    f = d / "final_result.json"
    if f.exists():
        info.update({f"json_{k}": v for k, v in json.load(open(f)).items() if k != "parameters"})
    return info

rows = []
_trials_sorted = sorted(study.trials, key=lambda tt: tt.number)
def _phase(t):
    # TPE memakai sampling acak selama jumlah trial COMPLETE yang sudah ada < n_startup_trials
    n_prev = sum(1 for u in _trials_sorted if u.number < t.number and u.state.name == "COMPLETE"
                 and u.datetime_complete is not None and t.datetime_start is not None
                 and u.datetime_complete <= t.datetime_start)
    return "random" if n_prev < N_STARTUP_DEFAULT else "TPE"
for t in study.trials:
    row = dict(trial=t.number, state=t.state.name, phase=_phase(t),
               J=t.value, map50_95=t.user_attrs.get("map50_95"), deltaE=t.user_attrs.get("color_shift_deltaE"),
               start=t.datetime_start, end=t.datetime_complete)
    row.update(t.params)
    row.update(read_trial_run(t.number))
    if row.get("map50_95") is None and row.get("json_map50_95") is not None:
        row["map50_95"] = row["json_map50_95"]; row["deltaE"] = row.get("json_color_shift_deltaE")
    rows.append(row)
trials = pd.DataFrame(rows).sort_values("trial")
trials["duration_min"] = (pd.to_datetime(trials["end"]) - pd.to_datetime(trials["start"])).dt.total_seconds() / 60
trials.to_csv(OUT_DIR / "bo_trials.csv", index=False)
done_all = trials[trials.state == "COMPLETE"].copy()
short = done_all[done_all.get("epochs_run", pd.Series(FULL_EPOCHS, index=done_all.index)).fillna(FULL_EPOCHS) < FULL_EPOCHS]
done = done_all.drop(short.index).copy()
# tandai konfigurasi ulangan (parameter identik dengan trial selesai sebelumnya)
_pcols = [c for c in study.best_trial.params]
_key = done[_pcols].round(10).astype(str).agg("|".join, axis=1)
done["repeat_of"] = [int(done.trial[_key == k].iloc[0]) if (_key == k).sum() > 1 and
                     int(done.trial[_key == k].iloc[0]) != int(tr) else None
                     for k, tr in zip(_key, done.trial)]
print("Konfigurasi ulangan:", {int(r.trial): int(r.repeat_of) for _, r in done.iterrows() if pd.notna(r.repeat_of)},
      f"-> konfigurasi unik: {_key.nunique()}")

print("\n" + "=" * 78 + "\n1. PROTOKOL BO YANG SEBENARNYA TERJADI\n" + "=" * 78)
print(f"Trial tercatat     : {len(trials)}  states={trials.state.value_counts().to_dict()}")
print(f"COMPLETE           : {len(done_all)}  -> dikeluarkan (epoch < {FULL_EPOCHS}): "
      f"{short.trial.astype(int).tolist()}  -> dianalisis: {len(done)}")
print("Ringkasan per trial (cek manual):")
cols=[c for c in ["trial","state","epochs_run","map50_95","deltaE","J","lr_actual_max","duration_min"] if c in trials]
print(trials[cols].round(4).to_string(index=False))
print(f"Trial acak (awal)  : {int((done.phase=='random').sum())} | dipandu TPE: {int((done.phase=='TPE').sum())}")
if len(done) < 20:
    print("  [PERHATIAN] < 20 trial selesai -> kemungkinan early stopping tingkat-study (patience 10) "
          "atau trial terputus. Naskah menulis '20 trials': sesuaikan.")
best = done.loc[done.J.idxmax()]
best_map = done.loc[done.map50_95.idxmax()]
print(f"Trial terpilih (J max)  : #{int(best.trial)}  J={best.J:.4f}  mAP50-95={best.map50_95:.4f}  dE={best.deltaE:.3f}")
print(f"Trial dgn mAP tertinggi : #{int(best_map.trial)}  J={best_map.J:.4f}  mAP50-95={best_map.map50_95:.4f}  dE={best_map.deltaE:.3f}")
print(f"Peringkat mAP trial terpilih: {int((done.map50_95 > best.map50_95).sum())+1} dari {len(done)}; "
      f"peringkat dE: {int((done.deltaE < best.deltaE).sum())+1} dari {len(done)}")
for c in ["arg_optimizer", "arg_lr0", "lr_actual_max", "arg_epochs", "epochs_run", "arg_patience",
          "arg_hsv_s", "arg_hsv_v", "arg_fliplr", "arg_mosaic"]:
    if c in done:
        vals = done[c].dropna().unique()
        print(f"  {c:<16}: {vals[:5]}")
print("  Catatan: program BO tidak menulis optimizer/lr0 -> Ultralytics memakai optimizer='auto' "
      "(AdamW, lr = 0.002*5/(4+nc) = 0.001429 untuk nc=3, karena 60 epoch x 88 batch < 10.000 iterasi). "
      "'lr_actual_max' dari results.csv adalah bukti angka yang benar-benar dipakai.")

# ==========================================================================
# 2. Apakah perbedaan antar-trial lebih besar dari noise antar-seed?
# ==========================================================================
print("\n" + "=" * 78 + "\n2. RENTANG ANTAR-TRIAL vs VARIASI ANTAR-SEED\n" + "=" * 78)
m = done.map50_95
print(f"mAP50-95 antar-trial: min {m.min():.4f}, median {m.median():.4f}, max {m.max():.4f}, "
      f"rentang {m.max()-m.min():.4f}, SD {m.std():.4f}")
print(f"Delta-E antar-trial : min {done.deltaE.min():.3f}, max {done.deltaE.max():.3f} "
      f"-> penalti lambda*dE berkisar {LAMBDA_COLOR*done.deltaE.min():.4f}..{LAMBDA_COLOR*done.deltaE.max():.4f}")
print(f"J antar-trial       : rentang {done.J.max()-done.J.min():.4f}")
gap = m.max() - best.map50_95   # selisih mAP antara trial ber-mAP tertinggi dan trial terpilih
for k, sd in SEED_SD_REF.items():
    print(f"  selisih (mAP tertinggi - mAP terpilih) = {gap:.4f}  vs SD antar-seed {k} = {sd:.4f}  "
          f"-> {gap/sd:.1f}x SD")
corr = np.corrcoef(done.J, done.deltaE)[0, 1], np.corrcoef(done.J, done.map50_95)[0, 1]
print(f"  korelasi J dgn Delta-E = {corr[0]:+.2f} | J dgn mAP = {corr[1]:+.2f}  "
      "(menunjukkan komponen mana yang paling menentukan peringkat trial)")

# ==========================================================================
# 3. Importance + stabilitas (bootstrap) untuk tiga target
# ==========================================================================
print("\n" + "=" * 78 + "\n3. IMPORTANCE PARAMETER & STABILITASNYA\n" + "=" * 78)
keep_nums = set(done.trial.astype(int))
completed_frozen = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.number in keep_nums]
params = list(completed_frozen[0].params.keys())
targets = {
    "J (objective)": lambda t: t.value,
    "mAP@0.5:0.95": lambda t: t.user_attrs.get("map50_95"),
    "Delta-E": lambda t: t.user_attrs.get("color_shift_deltaE"),
}
LEGEND_NAME = {"J (objective)": r"$J(\theta)$", "mAP@0.5:0.95": "mAP@0.5:0.95", "Delta-E": r"$\Delta E$"}

def importance(trial_list, target, seed):
    s = optuna.create_study(direction="maximize")
    s.add_trials(trial_list)
    ev = optuna.importance.FanovaImportanceEvaluator(seed=seed)
    return optuna.importance.get_param_importances(s, evaluator=ev, target=target, params=params)

rng = np.random.default_rng(0)
imp_rows = []
for tname, tfun in targets.items():
    if any(tfun(t) is None for t in completed_frozen):
        continue
    full = importance(completed_frozen, tfun, seed=0)
    boot = {p: [] for p in params}
    top1 = {p: 0 for p in params}
    for b in range(N_BOOT):
        idx = rng.integers(0, len(completed_frozen), len(completed_frozen))
        sample = [completed_frozen[i] for i in idx]
        try:
            imp = importance(sample, tfun, seed=int(b))
        except Exception:
            continue
        for p in params:
            boot[p].append(imp.get(p, 0.0))
        top1[max(imp, key=imp.get)] += 1
    nb = max(1, len(boot[params[0]]))
    for p in params:
        arr = np.array(boot[p]) if boot[p] else np.array([np.nan])
        imp_rows.append(dict(target=tname, param=p, importance=full.get(p, 0.0),
                             ci_low=np.nanpercentile(arr, 2.5), ci_high=np.nanpercentile(arr, 97.5),
                             rank1_freq=top1[p] / nb))
imp_df = pd.DataFrame(imp_rows)
imp_df.to_csv(OUT_DIR / "bo_importance_bootstrap.csv", index=False)
for tname, g in imp_df.groupby("target", sort=False):
    print(f"\nTarget: {tname}")
    for r in g.sort_values("importance", ascending=False).itertuples():
        print(f"  {r.param:<20} {r.importance:6.3f}  [95% bootstrap {r.ci_low:.3f}-{r.ci_high:.3f}]  "
              f"peringkat-1 pada {100*r.rank1_freq:4.0f}% bootstrap")

# ==========================================================================
# 4. Delta-E ulang untuk parameter terbaik: 30 gambar pertama vs seluruh valid
# ==========================================================================
de_summary = None
if RECOMPUTE_DELTAE and (DATASET_ROOT / "valid" / "images").exists():
    import cv2
    from skimage.color import rgb2lab, deltaE_ciede2000
    P = {k: best[k] for k in params}

    def pipeline(img, p):
        x = img.astype(np.float32) / 255.0
        g = np.clip(np.log(p["gamma_target_mean"]) / np.log(max(float(x.mean()), 1e-6)),
                    p["gamma_min"], p["gamma_max"])
        x = np.clip(np.power(x, g) * 255, 0, 255).astype(np.uint8)
        bl = cv2.GaussianBlur(x, (0, 0), sigmaX=p["edge_blur_sigma"])
        x = np.clip(cv2.addWeighted(x, 1 + p["edge_strength"], bl, -p["edge_strength"], 0), 0, 255).astype(np.uint8)
        h = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
        h[:, :, 1] = np.clip(h[:, :, 1] * p["hsv_saturation_gain"], 0, 255)
        return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def de(path):
        im = cv2.imread(path)
        a = rgb2lab(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
        b = rgb2lab(cv2.cvtColor(pipeline(im, P), cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
        return float(deltaE_ciede2000(a, b).mean())

    vals = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = sorted(set(sum([glob.glob(str(DATASET_ROOT / "valid" / "images" / e)) for e in exts], [])))
    for p in paths:
        import re as _re
        vals.append(dict(file=os.path.basename(p), cls=_re.match(r"(N\d\d)", os.path.basename(p)).group(1) if _re.match(r"(N\d\d)", os.path.basename(p)) else "?", dE=de(p)))
    dv = pd.DataFrame(vals)
    first = dv.head(DELTAE_REF_N)
    de_summary = dict(first30=first.dE.mean(), first30_classes=first.cls.value_counts().to_dict(),
                      all_valid=dv.dE.mean(), per_class=dv.groupby("cls").dE.mean().round(3).to_dict(),
                      n_valid=len(dv))
    dv.to_csv(OUT_DIR / "deltaE_best_params_valid.csv", index=False)
    print("\n" + "=" * 78 + "\n4. DELTA-E PARAMETER TERBAIK: SUBSET BO vs SELURUH VALIDATION\n" + "=" * 78)
    print(f"30 gambar pertama (dipakai BO): {de_summary['first30']:.3f}  kelas: {de_summary['first30_classes']}")
    print(f"Seluruh validation (n={len(dv)}) : {de_summary['all_valid']:.3f}")
    print(f"Per kelas                      : {de_summary['per_class']}")
    print(f"Nilai tersimpan di trial       : {best.deltaE:.3f}  (harus ~ sama dgn baris pertama)")

# ==========================================================================
# 5. Figure siap jurnal
# ==========================================================================
# 5a. Riwayat optimasi
fig, ax = plt.subplots(figsize=(3.5, 2.9))
rep = done.repeat_of.notna()
for ph, mk, lab in [("random", "o", "Random start-up"), ("TPE", "s", "TPE-guided")]:
    s = done[(done.phase == ph) & ~rep]
    ax.scatter(s.trial, s.J, marker=mk, s=22, facecolors="none" if ph == "random" else "black",
               edgecolors="black", linewidths=0.9, label=lab, zorder=3)
if rep.any():
    s = done[rep]
    ax.scatter(s.trial, s.J, marker="x", s=26, color="black", linewidths=1.0,
               label="Repeated configuration", zorder=3)
ax.plot(done.trial, done.J.cummax(), color="black", lw=1.2, label="Best so far", zorder=2, drawstyle="steps-post")
b = done.loc[done.J.idxmax()]
ax.annotate(f"trial {int(b.trial)}", (b.trial, b.J), xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=9)
ax.set_xlabel("Trial number"); ax.set_ylabel(r"Objective $J(\theta)$")
ax.set_xticks(range(0, int(done.trial.max()) + 2, 5))
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.8, handletextpad=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT_DIR / "fig_bo_history.png", bbox_inches="tight"); plt.close(fig)

# 5b. Importance dgn interval bootstrap (tiga target, satu panel)
order = (imp_df[imp_df.target == "J (objective)"].sort_values("importance").param.tolist()
         if "J (objective)" in imp_df.target.values else params)
tlist = [t for t in targets if t in imp_df.target.values]
fig, ax = plt.subplots(figsize=(3.5, 3.7))
h = 0.8 / len(tlist); shades = ["0.15", "0.55", "0.85"]
for k, tname in enumerate(tlist):
    g = imp_df[imp_df.target == tname].set_index("param").loc[order]
    y = np.arange(len(order)) + (k - (len(tlist) - 1) / 2) * h
    ax.barh(y, g.importance, height=h, color=shades[k], edgecolor="black", lw=0.5, label=LEGEND_NAME.get(tname, tname))
    ax.errorbar(g.importance, y, xerr=[np.clip(g.importance - g.ci_low, 0, None), np.clip(g.ci_high - g.importance, 0, None)],
                fmt="none", ecolor="black", elinewidth=0.7, capsize=1.5)
ax.set_yticks(range(len(order))); ax.set_yticklabels([PARAM_LABEL.get(p, p) for p in order])
ax.set_xlabel("Relative importance (fANOVA)"); ax.set_xlim(0, 1)
h_, l_ = ax.get_legend_handles_labels()
ax.legend(h_[::-1], l_[::-1], frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=3, columnspacing=0.6, handletextpad=0.3, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT_DIR / "fig_bo_importance_bootstrap.png", bbox_inches="tight"); plt.close(fig)

# ==========================================================================
# 6. Teks & tabel untuk naskah
# ==========================================================================
n_rand, n_tpe = int((done.phase == "random").sum()), int((done.phase == "TPE").sum())
lr_txt = "0.001429"   # lr0 hasil optimizer='auto' Ultralytics: 0.002*5/(4+3)
if "lr_actual_max" in done and done.lr_actual_max.notna().any():
    print(f"  Puncak lr/pg0 teramati (trial {FULL_EPOCHS} epoch): {sorted(done.lr_actual_max.dropna().round(6).unique())} "
          "(sedikit < 0.001429 karena warmup 3 epoch lalu decay linear)")
de_txt = (f" Recomputed over the whole validation subset, the ΔE of the selected configuration was "
          f"{de_summary['all_valid']:.2f} (vs {de_summary['first30']:.2f} on the reference subset)." if de_summary else "")
jt = imp_df[imp_df.target == "J (objective)"].sort_values("importance", ascending=False)
top = jt.iloc[0] if len(jt) else None

table = f"""**Table [X]. Bayesian optimization settings**

| Item | Setting |
|---|---|
| Library / sampler | Optuna [version], TPE sampler (univariate), seed 42 |
| Initialization | {n_rand} random trials, then {n_tpe} TPE-guided trials (24 EI candidates) |
| Trials completed | {len(done)} |
| Objective | J(θ) = mAP@0.5:0.95 (mask, validation) − 0.01·ΔE(θ), maximized |
| ΔE(θ) | Mean CIEDE2000 over all pixels, averaged over the first {DELTAE_REF_N} validation images |
| Per-trial training | YOLO11n-seg from COCO-pretrained weights, 60 epochs, batch 16, 640 × 640, seed 42, deterministic |
| Optimizer | Ultralytics automatic selection: AdamW, initial lr = {lr_txt} (linear decay), momentum 0.9 |
| Checkpoint evaluated | best.pt (highest Ultralytics validation fitness), re-evaluated on the validation subset |
| Data | Preprocessing applied to training and validation subsets; test subset not used |
| Selection rule | Trial with the highest J(θ) |
| Study-level stopping | Stop after 10 consecutive trials without improvement in J |
"""
text = (
    f"Bayesian Optimization was implemented with the univariate TPE sampler of Optuna (seed 42). "
    f"The first {n_rand} trials were sampled at random to initialize the density estimates and the remaining "
    f"{n_tpe} were proposed by the TPE expected-improvement criterion, giving {len(done)} completed trials. "
    f"In each trial, the training and validation images were preprocessed with the candidate parameters and "
    f"YOLO11n-seg was trained from the same COCO-pretrained weights for 60 epochs (batch 16, 640 × 640, "
    f"seed 42, deterministic mode), using the optimizer selected automatically by Ultralytics (AdamW, "
    f"initial lr = {lr_txt} with linear decay). The best checkpoint was evaluated on the validation subset, and mAP₅₀₋₉₅ in Eq. (8) denotes "
    f"the resulting mask mAP@0.5:0.95. ΔE(θ) is the CIEDE2000 colour difference between the original and "
    f"preprocessed images, averaged over all pixels of each image and then over the first {DELTAE_REF_N} images of "
    f"the validation subset.{de_txt} The trial with the highest J(θ) was selected. The same validation subset was "
    f"also used for the preprocessing ablation and for fine-tuning selection; the test subset was not used at any "
    f"stage of optimization or selection. Across the {len(done)} trials, validation mAP@0.5:0.95 ranged from "
    f"{m.min():.4f} to {m.max():.4f}; because each trial was trained with a single seed, "
    + ("this spread is comparable to the run-to-run variability reported in Table 9, so the selected configuration "
       "should be regarded as one well-performing setting rather than a uniquely optimal one."
       if gap < 2 * max(SEED_SD_REF.values()) else
       "the selected configuration should nevertheless be interpreted with the run-to-run variability reported "
       "in Table 9 in mind.")
    + (f" The parameter importance in Fig. [Y] was estimated with fANOVA and its uncertainty with {N_BOOT} bootstrap "
       f"resamples of the trials; {PLAIN_LABEL.get(top.param, top.param)} ranked first in "
       f"{100*top.rank1_freq:.0f}% of resamples, so the ranking is reported as indicative only."
       if top is not None else "")
)
(OUT_DIR / "teks_revisi_bo.md").write_text(table + "\n\n" + text + "\n", encoding="utf-8")
print("\n" + "=" * 78 + "\n6. TABEL & TEKS UNTUK NASKAH (cek dulu angka sebelum dipakai)\n" + "=" * 78)
print(table); print(text)
print(f"\nSemua file di: {OUT_DIR}")
print("  bo_trials.csv, bo_importance_bootstrap.csv, deltaE_best_params_valid.csv,")
print("  fig_bo_history.png, fig_bo_importance_bootstrap.png, teks_revisi_bo.md")
