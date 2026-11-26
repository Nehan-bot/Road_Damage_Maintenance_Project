
# Dataset Analysis Notes
## Road Damage Detection and Intelligent Maintenance Prioritization System Using Computer Vision and NLP

---

## 1. Dataset Overview

**Dataset:** RDD2022 – India subset  
**Final dataset:** India-specific, four-class road damage dataset  
**Annotation format:** YOLO bounding-box format

The dataset was segregated by country and converted to retain four road-damage categories. The original images were retained, while annotations belonging to the excluded category, "other corruption", were removed.

### Dataset locations

- Original multi-country dataset: `C:\College\M.tech\Datasets\RDD2022\RDD_SPLIT`
- India-only dataset: `C:\College\M.tech\Datasets\RDD2022_India`
- Final four-class dataset: `C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class`

### Dataset split

| Split | Images | Label files |
|---|---:|---:|
| Train | 5,368 | 5,368 |
| Validation | 1,172 | 1,172 |
| Test | 1,166 | 1,166 |
| **Total** | **7,706** | **7,706** |

All images have corresponding YOLO label files. Empty label files are retained for images without any of the four selected damage categories.

---

## 2. Selected Road Damage Categories

The final dataset contains four classes.

| Class ID | Road damage category |
|---:|---|
| 0 | Longitudinal crack |
| 1 | Transverse crack |
| 2 | Alligator crack |
| 3 | Pothole |

The original class "other corruption" was excluded at the annotation level. Images were retained even when their labels became empty after filtering.

---

## 3. Annotation Format

The annotations use the YOLO bounding-box format:

`class_id x_center y_center width height`

The bounding-box coordinates are normalized relative to image dimensions and are expected to be in the range 0 to 1.

The final dataset verification reported zero invalid annotation lines in all three splits.

---

## 4. Dataset Verification Results

### Training split

| Measure | Result |
|---|---:|
| Label files | 5,368 |
| Empty label files | 3,091 |
| Longitudinal crack annotations (class 0) | 1,103 |
| Transverse crack annotations (class 1) | 50 |
| Alligator crack annotations (class 2) | 1,428 |
| Pothole annotations (class 3) | 2,236 |
| Invalid annotation lines | 0 |

### Validation split

| Measure | Result |
|---|---:|
| Label files | 1,172 |
| Empty label files | 681 |
| Longitudinal crack annotations (class 0) | 223 |
| Transverse crack annotations (class 1) | 11 |
| Alligator crack annotations (class 2) | 307 |
| Pothole annotations (class 3) | 506 |
| Invalid annotation lines | 0 |

### Test split

| Measure | Result |
|---|---:|
| Label files | 1,166 |
| Empty label files | 711 |
| Longitudinal crack annotations (class 0) | 229 |
| Transverse crack annotations (class 1) | 7 |
| Alligator crack annotations (class 2) | 286 |
| Pothole annotations (class 3) | 445 |
| Invalid annotation lines | 0 |

---

## 5. Bounding-Box Area Analysis

Bounding-box area is calculated as:

`normalized_width × normalized_height`

The resulting value represents the proportion of the image covered by the bounding box.

### Results

| Road damage category | Bounding boxes | Mean area | Minimum area | Maximum area |
|---|---:|---:|---:|---:|
| Longitudinal crack | 1,555 | 0.02823 | 0.00083 | 0.25230 |
| Transverse crack | 68 | 0.02594 | 0.00179 | 0.11683 |
| Alligator crack | 2,021 | 0.11431 | 0.00305 | 0.66815 |
| Pothole | 3,187 | 0.02290 | 0.00059 | 0.38713 |

### Observations

1. Potholes have the highest number of bounding-box annotations (3,187).
2. Alligator cracks have the largest mean bounding-box area (0.11431).
3. Potholes have the smallest mean bounding-box area (0.02290).
4. Longitudinal and transverse cracks have similar mean bounding-box areas.
5. The bounding-box sizes vary considerably within each class.

These results indicate that the detector must learn to recognize damage regions with different spatial sizes.

---

## 6. Class Imbalance Analysis

The training annotations are distributed unevenly across the four categories.

| Class | Training annotations |
|---|---:|
| Longitudinal crack | 1,103 |
| Transverse crack | 50 |
| Alligator crack | 1,428 |
| Pothole | 2,236 |
| **Total** | **4,817** |

The transverse crack category is severely underrepresented compared with the other categories.

This imbalance may affect the model's ability to learn and detect transverse cracks reliably.

### Planned handling

- Preserve the original validation and test datasets.
- Consider targeted data augmentation for the underrepresented class in the training split.
- Avoid indiscriminate duplication of images.
- Report detection performance separately for each class.
- Inspect false negatives and false positives for the underrepresented class.

---

## 7. Dataset Limitations and Considerations

1. The four classes have an unequal number of annotations.
2. The transverse crack category has particularly few examples.
3. Bounding-box sizes vary substantially between categories and individual examples.
4. A considerable number of images have empty labels after filtering. These are retained as images without any of the selected four damage categories.
5. Annotation alignment and image quality should be visually inspected before training.

Empty label files are not automatically annotation errors. They can represent valid negative images for the selected four-class detection task.

---

## 8. Planned Visual Inspection

Before model training, sample images from the training, validation and test splits will be inspected with their YOLO bounding boxes drawn over the original images.

The inspection will check:

- Whether bounding boxes align with visible road damage.
- Whether class IDs correspond to the correct damage categories.
- Whether boxes are clipped, misplaced or unusually large.
- Whether small damage regions are adequately represented.
- Whether empty-label images genuinely contain none of the selected categories.

The original dataset files will remain unchanged during this inspection.

---

## 9. Preliminary Dataset Suitability Assessment

The dataset has a useful structure for a four-class road-damage object-detection experiment:

- It contains separate training, validation and test splits.
- It has 7,706 images with corresponding label files.
- The final annotations use the YOLO format.
- The verification reported zero invalid annotation lines.
- The four selected damage categories are represented.

However, the severe class imbalance, particularly for transverse cracks, is an important limitation. Visual annotation inspection and a baseline YOLO training experiment are required before concluding that the dataset is fully suitable for the final model.

**Current status:** Structurally prepared and programmatically verified; visual quality inspection and baseline model evaluation are pending.
