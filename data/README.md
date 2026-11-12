# Dataset Documentation

This folder documents the datasets used in the study and provides image-level partition files. The images and annotations themselves are not redistributed in this repository (see Section 6).

## 1. Main Rice Leaf Dataset

The main dataset consists of 671 original field images of individual rice leaves acquired between 2024 and 2026 with a Sony a6000 digital camera at rice fields in Gresik, Kediri, and Yogyakarta, Indonesia. It contains 677 annotated leaf instances belonging to three Leaf Color Chart (LCC) reference classes:

- **Class 0 – N23:** LCC scores of 2.0–2.5 (309 instances).
- **Class 1 – N34:** LCC scores of 3.0–3.5 (165 instances).
- **Class 2 – N45:** LCC scores of 4.0–5.0 (203 instances).

Each leaf is annotated with a polygon in YOLO instance-segmentation format. The LCC reading was determined in the field before the leaf was photographed.

### 1.1 Partition and Augmentation

The 671 original images were partitioned at the original-image level (70:20:10) into three subsets before any augmentation:

- **Training:** 470 original images (471 instances).
- **Validation:** 134 images (138 instances).
- **Internal test:** 67 images (68 instances).

Offline augmentation (horizontal and vertical flipping and rotation between −12° and +12°, three outputs per image) was applied only to the training subset after partitioning. The exported training subset contains 1,405 images with 1,408 instances and is listed file by file in `splits/main_dataset_split.csv`. [TODO: state the reason why the exported count differs from 3 × 470 = 1,410, or remove this sentence. Cannot be verified from the manuscript.]

The validation and internal test subsets are not augmented.

### 1.2 Limitations of the Partition

Plant, acquisition-session, and site identifiers were not retained during dataset consolidation. The split could therefore not be grouped by these factors, and images from the same plant, session, or site may appear in different subsets. No byte-identical files exist across the exported subsets. Original file names that occur in both the training and validation subsets were inspected and correspond to different photographs.

## 2. External Test Dataset

The external test set comprises 130 images with 130 annotated instances, collected in September–October 2026 with the same camera and the same labelling and annotation protocols as the main dataset. It includes new acquisition sessions in Gresik (45 images, 45 instances) and Kediri (48 images, 48 instances), which are represented in the main dataset, and Sidoarjo (37 images, 37 instances), which is not. Overall, the set contains 118 N23 instances and 12 N34 instances (two in Gresik, two in Kediri, and eight in Sidoarjo). N45 is absent because no leaf with an LCC reading of 4.0 or higher was found.

All external images were used only for testing; they were not used for training, augmentation, or configuration selection, and they were exported without augmentation. In the external annotation files, class 0 is N23 and class 1 is N34.

The external set is organized per site as a Roboflow YOLO export:

```text
New_Gsk/   (Gresik)
    data.yaml
    test/images/
    test/labels/
New_Kdr/   (Kediri)
    data.yaml
    test/images/
    test/labels/
New_Sda/   (Sidoarjo)
    data.yaml
    test/images/
    test/labels/
```

## 3. Annotation Format

Each line of a label file describes one leaf: a class identifier followed by normalized polygon coordinates (0–1 relative to image width and height):

```text
class_id x1 y1 x2 y2 ... xn yn
```

In the main dataset, class 0 is N23, class 1 is N34, and class 2 is N45. In the external dataset, class 0 is N23 and class 1 is N34.

## 4. Image-Level Partition Files

The `splits/` folder lists every image of both datasets together with its SHA-256 checksum, so that a recipient of the data can verify that it matches the data used in the study.

`main_dataset_split.csv` contains one row for each exported image of the main dataset (1,606 images). Each row gives the subset (training, validation, or internal test), the exported file name, the camera-generated original name, whether the image is an augmented output, the number of annotated instances in total and per class, image width and height, and the SHA-256 checksum. `main_dataset_summary.csv` gives the numbers of images and instances per subset and class.

`external_test_files.csv` contains one row for each external image (130 images), with the site, file name, original name, instance counts per class, image width and height, and the SHA-256 checksum. `external_test_summary.csv` gives the numbers of images and instances per site and class.

The original name is the camera-generated file name without the export suffix. Camera file names were reused by different photographs, also across subsets, so files must be identified by their SHA-256 checksum (and, where helpful, by image dimensions), not by name.

The files were generated with `scripts/data/make_partition_csv.py`.

## 5. Intended Use

The data support research on rice leaf color classification and instance segmentation using LCC reference categories. The LCC categories are visual reference labels; they are not direct measurements of leaf nitrogen concentration.

## 6. Dataset Availability

The images and annotations are not publicly available. The partition files in the `splits/` folder are public. Access to the images and annotations for verification is available through the following controlled-access procedure: [TODO: contact (corresponding author e-mail), information a requester must provide, data-use conditions, and expected response time. Cannot be verified from the manuscript.]