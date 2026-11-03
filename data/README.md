# Dataset Documentation

## 1. Main Rice Leaf Dataset

The main dataset consists of 671 original field images of individual rice leaves acquired using a Sony α6000 digital camera. The dataset contains 677 annotated leaf instances belonging to three Leaf Color Chart (LCC) reference classes:

- **Class 0 – N23:** LCC scores of 2.0–2.5.
- **Class 1 – N34:** LCC scores of 3.0–3.5.
- **Class 2 – N45:** LCC scores of 4.0–5.0.

Each leaf instance is annotated using polygon-based instance segmentation annotations in YOLO segmentation format.

### 1.1 Original Dataset Split and Augmentation

The original dataset was divided at the image level into three subsets before augmentation:

- **Training:** 470 original images.
- **Validation:** 134 original images.
- **Internal test:** 67 original images.

The total number of original images was 671, comprising 677 annotated leaf instances.

Offline augmentation was subsequently applied exclusively to the training subset to increase the number of training samples and improve model generalization. Depending on the augmentation procedure, up to three additional augmented outputs could be generated per original training image.

The validation and internal test subsets were kept unaugmented to preserve their original image distributions and provide an unbiased evaluation of model performance.

### 1.2 Data Collection Locations

The original rice leaf images were collected from three agricultural areas in Indonesia:

- Gresik, East Java.
- Kediri, East Java.
- Yogyakarta, Special Region of Yogyakarta.

Site and session identifiers were not retained in the exported dataset. This limitation is considered when interpreting the model evaluation results and is discussed in the manuscript.

## 2. External Test Dataset

An additional external test dataset was collected independently from new acquisition sessions in Gresik and Kediri, as well as from Sidoarjo, a location not represented in the main dataset.

All external images were reserved exclusively for testing and were not used for model training, augmentation, or model configuration selection.

The external dataset is organized into the following directory structure:

```text
New_Gsk/
    images/
    labels/

New_Kdr/
    images/
    labels/

New_Sda/
    images/
    labels/
```

These external test sets were used to assess model generalization across different acquisition sessions and geographical locations.

## 3. Annotation Format

The dataset uses the YOLO instance segmentation annotation format.

Each annotation contains a class identifier followed by normalized polygon coordinates:

```text
class_id x1 y1 x2 y2 ... xn yn
```

The class identifiers are defined as follows:

- `0`: N23
- `1`: N34
- `2`: N45

Polygon coordinates are normalized relative to the image width and height, with values ranging from 0 to 1.

## 4. Dataset Availability

The rice leaf image dataset and corresponding annotations are not publicly available but may be obtained from the corresponding author upon reasonable request, subject to applicable data-sharing permissions.

## 5. Usage Notes

The dataset is intended for research on rice leaf color classification, instance segmentation, and the assessment of leaf nitrogen status using Leaf Color Chart (LCC) reference categories.

The LCC categories serve as reference labels for leaf color and nitrogen-status assessment in this study. They should not be interpreted as direct measurements of leaf nitrogen concentration.