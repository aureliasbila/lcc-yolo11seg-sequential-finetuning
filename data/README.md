# Data

## Main dataset
- 671 field images of individual rice leaves (Sony a6000), 677 annotated instances, three LCC classes:
  N23 (LCC 2.0–2.5), N34 (3.0–3.5), N45 (4.0–5.0), polygon (instance-segmentation) annotations.
- Split at the image level: train 470 / validation 134 / test 67 images; offline augmentation (up to three
  outputs per image) applied to the training subset only.
- Collected in Gresik, Kediri, and Yogyakarta (Indonesia). Site and session identifiers were not retained
  in the exported dataset (see the Limitations section of the paper).

## External test set
- Newly acquired images from Gresik and Kediri (new sessions) and Sidoarjo (site not represented in the
  main dataset); every image is used for testing only. Folder layout:

```
New_Gsk/{images,labels}   New_Kdr/{images,labels}   New_Sda/{images,labels}
```

## Availability
[Choose one before release and edit:  (a) public download link (Roboflow Universe / Zenodo DOI);
 (b) "available from the corresponding author upon reasonable request".]

Label format: YOLO segmentation (`class x1 y1 x2 y2 ...`, normalized), class IDs 0 = N23, 1 = N34, 2 = N45.
