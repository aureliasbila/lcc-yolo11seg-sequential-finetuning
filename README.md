# Effects of Adaptive Image Preprocessing and Sequential Fine-Tuning on YOLO11-Based Leaf Color Chart Classification of Rice Leaves

Code accompanying the article

> A. Salsabila, M. Z. S. Hadi, and T. Karlita, "Effects of Adaptive Image Preprocessing and Sequential
> Fine-Tuning on YOLO11-Based Leaf Color Chart Classification of Rice Leaves", *International Journal of
> Intelligent Engineering and Systems*, under review (manuscript ID 20266287).

This repository contains the colour-preserving preprocessing pipeline, the Bayesian optimization program,
the training, fine-tuning and ablation programs, the statistical and error analyses, the Grad-CAM/LayerCAM
analysis, the evaluation scripts, the final model weights, and the image-level dataset partition files used
in the article.

## Overview

The experiments follow these stages. Model selection in every stage uses the validation subset only.

- **Baseline:** YOLOv8n-seg, YOLOv9c-seg and YOLO11n-seg, each with three learning rates.
- **Preprocessing search:** Bayesian optimization (Optuna TPE) of adaptive gamma, edge enhancement and HSV
  saturation gain, with a colour-fidelity penalty (ΔE, CIEDE2000).
- **Fine-tuning grid:** number of frozen layers {0, 5, 10} × learning rate {0.0001, 0.0003, 0.0005}.
- **Ablation:** six training strategies (single-stage raw, single-stage processed, raw→raw,
  processed→processed, raw→processed, processed→raw), each with seeds 0, 1 and 2.
- **Final evaluation:** the selected configuration (raw→processed) only, on the internal test subset
  (reference) and the external test subset (final estimate).

Classes: **N23** (LCC 2.0–2.5), **N34** (LCC 3.0–3.5) and **N45** (LCC 4.0–5.0), used as visual proxies of
nitrogen status.

## Repository structure

```
configs/
  preprocessing_params.yaml    selected preprocessing parameters and BO search space
  train_args_reference.yaml    Ultralytics training settings and stage-specific values
  data_example.yaml            example dataset file (class order N23, N34, N45)
scripts/
  data/
    build_processed_dataset.py raw -> processed dataset (pipeline with the selected parameters)
    make_partition_csv.py      image-level partition files with SHA-256 checksums
  bayesian_optimization/
    bo_optimization.py         Optuna TPE search (imports the pipeline from build_processed_dataset.py)
  training/
    01_baseline_training.py    baseline models x learning rates
    finetune_grid.py           Stage-2 grid: freeze layers x learning rate
    ablation_study.py          six strategies x seeds, validation-based summary
    Dataset_Preparation+Preprocessing.ipynb   dataset preparation notebook
  analysis/
    bo_analysis.py             trial table, fANOVA importance with bootstrap, recomputed delta-E, figures
    ablation_statistics.py     ANOVA, blocked ANOVA and Friedman test on validation runs
    val_error_analysis.py      validation error types per configuration
    prediction_comparison.py   qualitative prediction comparison
    layercam_analysis.py       Grad-CAM/LayerCAM maps for internal and external examples
  evaluation/
    External_Data.ipynb        internal/external test evaluation (overall, per-class AP, confusion matrix, bootstrap CI)
    latency_benchmark.py       parameters, GFLOPs, latency, FPS
    reeval_weights.py          re-evaluation of the released checkpoints on the validation subset
results/                       CSV files with the reported results
training_logs/                training logs of the final models (three seeds)
weights/                      final model weights, checksums and usage notes
data/                         dataset description, availability and partition files
```

## Installation

```bash
git clone https://github.com/aureliasbila/lcc-yolo11seg-sequential-finetuning.git
cd lcc-yolo11seg-sequential-finetuning
pip install -r requirements.txt
```

Experiments were run on Google Colab (Tesla T4 GPU). Training was performed in separate Colab sessions, and
the Ultralytics versions recorded in the final checkpoints are 8.4.158, 8.4.161 and 8.4.163. Latency was
measured, and the released checkpoints were re-evaluated, with Ultralytics 8.4.175.
`bo_optimization.py`, `finetune_grid.py` and `ablation_study.py` take their paths as command-line arguments
(`--help` lists all options). The other scripts and notebooks contain a path block at the top that must be
edited to point to your data.

## Reproducing the main steps

All training uses Ultralytics with the settings in `configs/train_args_reference.yaml`. The settings actually
used for the final models are recorded in `training_logs/` (`args.yaml` of each run). Datasets use the YOLO
segmentation format (`train/`, `valid/`, `test/`, each with `images/` and `labels/`, plus `data.yaml`).

1. **Baseline models:**
   `python scripts/training/01_baseline_training.py`
2. **Processed dataset** with the selected parameters (`configs/preprocessing_params.yaml`): run
   `scripts/data/build_processed_dataset.py` after setting its path block, or use
   `scripts/training/Dataset_Preparation+Preprocessing.ipynb`.
3. **Bayesian optimization** (60-epoch trials, objective J = mAP@0.5:0.95 − 0.01·ΔE, TPE seed 42):
   ```bash
   PYTHONPATH=scripts/data python scripts/bayesian_optimization/bo_optimization.py \
       --data-root /path/to/Dataset_Raw --output-dir runs/bo_preprocessing
   ```
   The study is stored in SQLite and resumes when the same command is run again. The test subset is not
   read. The original runs used `optimizer="auto"` (AdamW, lr ≈ 0.001429 for three classes), which is the
   default of `--optimizer`; check `args.yaml` of a run for the value actually applied.
4. **BO analysis** (trial history, fANOVA importance, delta-E on the whole validation subset):
   `python scripts/analysis/bo_analysis.py`
5. **Fine-tuning grid** (validation mAP@0.5:0.95 selects the number of frozen layers and the learning rate):
   ```bash
   python scripts/training/finetune_grid.py \
       --stage1-weights runs/stage1/weights/best.pt \
       --data-yaml /path/to/Dataset_Processed/data.yaml \
       --output-dir runs/finetune_grid
   ```
6. **Ablation of training strategies** (seeds 0, 1, 2; validation subset by default):
   ```bash
   python scripts/training/ablation_study.py --seeds 0 1 2 \
       --data-root /path/to/datasets --output-dir runs/ablation
   python scripts/training/ablation_study.py --mode aggregate --output-dir runs/ablation
   ```
7. **Statistical comparison** of the configurations on the validation subset:
   ```bash
   python scripts/analysis/ablation_statistics.py --csv results/ablation_validation_runs.csv
   ```
8. **Error analysis and qualitative comparison:**
   `python scripts/analysis/val_error_analysis.py` and `python scripts/analysis/prediction_comparison.py`
9. **Final evaluation** of the selected configuration on the internal and external test subsets:
   `scripts/evaluation/External_Data.ipynb`
10. **Grad-CAM/LayerCAM figure:**
    `python scripts/analysis/layercam_analysis.py`
11. **Computational efficiency:**
    `python scripts/evaluation/latency_benchmark.py`
12. **Re-evaluation of the released weights** on the validation subset, compared with the training logs and
    with `results/ablation_validation_runs.csv`:
    ```bash
    python scripts/evaluation/reeval_weights.py --weights-dir weights \
        --data-yaml /path/to/Dataset_Processed/data.yaml \
        --results-dir training_logs --repo-csv results/ablation_validation_runs.csv
    ```

## From article results to scripts

- Baseline comparison: `01_baseline_training.py`, `results/baseline_validation_results.csv`.
- Bayesian optimization (settings, trials, parameter importance): `bo_optimization.py`, `bo_analysis.py`,
  and the files in `results/bayesian optimization/`.
- Fine-tuning grid: `finetune_grid.py`, `results/finetune_grid_results.csv`.
- Ablation and statistics: `ablation_study.py`, `ablation_statistics.py`,
  `results/ablation_validation_runs.csv`.
- Error analysis and predictions: `val_error_analysis.py`, `prediction_comparison.py`.
- Internal and external test results: `External_Data.ipynb`.
- Grad-CAM/LayerCAM figure: `layercam_analysis.py`.
- Efficiency: `latency_benchmark.py`.
- Final models and their training settings: `weights/` and `training_logs/`.

## Model weights

`weights/` contains the final selected configuration (raw→processed) for seeds 0, 1 and 2
(`rawproc_seed0.pt`, `rawproc_seed1.pt`, `rawproc_seed2.pt`), together with `SHA256SUMS.txt` and a
`README.md` with the class order, training settings, usage example and the results of a re-evaluation. The
external-test results in the article are the mean over these three seed-specific models. Weights of the other
ablation configurations are not included.

## Training logs

`training_logs/` holds, for each of the three final models, `results.csv`, the precision,
recall, F1 and precision–recall curves, and the confusion matrices.

## Data

The images and annotations are not publicly available. [`data/README.md`](data/README.md) describes the main
and external datasets, and `data/splits/` contains the image-level partition files (file names, instance
counts per class, image dimensions and SHA-256 checksums) for both datasets, generated with
`scripts/data/make_partition_csv.py`. The access procedure for verification is described in
`data/README.md`.

## License and citation

Code: MIT License (see `LICENSE`). Please cite the article if you use this code (`CITATION.cff`).