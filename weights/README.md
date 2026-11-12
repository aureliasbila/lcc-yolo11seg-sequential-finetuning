# Model Weights

This release contains the final selected configuration of the study, Raw to Processed (Stage 1 fine-tuned on raw images, Stage 2 fine-tuned on adaptively preprocessed images), for three random seeds. All models are YOLO11n-seg networks (about 6 MB each) with three classes in the order 0 = N23, 1 = N34, 2 = N45.

## Files

- `rawproc_seed0.pt`: final model for seed 0 (best checkpoint of Stage 2, selected on the validation subset).
- `rawproc_seed1.pt`: final model for seed 1.
- `rawproc_seed2.pt`: final model for seed 2.
- `SHA256SUMS.txt`: SHA-256 checksums of the three files. Verify with `sha256sum -c SHA256SUMS.txt`.

The external-test results in the manuscript are the mean over these three seed-specific models. The weights of the other ablation configurations and the Stage 1 checkpoints are not included.

## Training Settings of Stage 2

Each model was initialized from the Stage 1 raw-image checkpoint of the same seed and trained for 50 epochs with the AdamW optimizer, an initial learning rate of 0.0003, no frozen layers, batch size 16, input size 640, and deterministic mode. The seeds were 0, 1, and 2. The Ultralytics versions recorded inside the checkpoints are 8.4.163 (seed 0), 8.4.158 (seed 1), and 8.4.161 (seed 2).

## Re-evaluation of the Released Checkpoints

The three checkpoints were re-evaluated on the 134-image validation subset with Ultralytics 8.4.175 in a single environment. The resulting mask mAP@0.5:0.95 values are 0.9459 (seed 0), 0.9463 (seed 1), and 0.9429 (seed 2), with a mean of 0.9450. These values are 0.04–0.66% (mean 0.35%) below those reported in the manuscript and in `results/ablation_validation_runs.csv` (0.9492, 0.9467, and 0.9491; mean 0.9484), and 0.25–0.56% (mean 0.42%) below the final-epoch values in the training logs (0.9483, 0.9517, and 0.9472). The checkpoints were trained with Ultralytics 8.4.158–8.4.163, and the cause of the small differences has not been established. [TODO: state the validation data used for the re-evaluation (the same processed validation images as in training, or regenerated ones).]

## Usage

```python
from ultralytics import YOLO

model = YOLO("rawproc_seed0.pt")
results = model.predict("leaf.jpg", imgsz=640, conf=0.25)
print(model.names)  # {0: 'N23', 1: 'N34', 2: 'N45'}
```

Use the input pipeline, confidence threshold, and image size reported in the manuscript when reproducing the evaluation. [TODO: state whether the evaluation input is passed through the adaptive preprocessing step (`scripts/data/build_processed_dataset.py`) before inference, and the confidence threshold used. Cannot be verified from the checkpoints.]

## Evaluation Data

The validation subset and the internal test subset are described in `data/README.md`. The images are not distributed with this release; the image-level partition files with SHA-256 checksums are in `data/splits/`.
