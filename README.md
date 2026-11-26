# Road Damage Detection and Intelligent Maintenance
# Prioritization System Using Computer Vision and NLP

## Project Overview

A road maintenance decision-support system that combines
road damage detection, complaint-text analysis and
supervised maintenance-priority classification.

## Main Components

1. Computer Vision - YOLO
2. NLP - Text preprocessing and TF-IDF
3. Phase 1 - Decision Tree
4. Phase 2 - Random Forest
5. Application - Streamlit
6. Database - SQLite
7. Deployment - Docker

## Development Environment

- Windows 11
- Python
- Visual Studio Code
- Scikit-learn
- OpenCV
- Ultralytics
- Streamlit

## Final YOLO26s Test Evaluation

The selected YOLO26s model (`best.pt`) was evaluated on the held-out test split using an image size of 640.

- Test images: 2,278
- Annotated objects: 4,673
- Precision: 58.5%
- Recall: 46.3%
- mAP@50: 47.4%
- mAP@50–95: 21.4%

The highest class-wise mAP@50 was achieved for transverse cracks (55.6%). Longitudinal cracks had the lowest recall (40.7%), indicating that this class remains a key area for improvement.

The best checkpoint was selected during training using validation performance. The test split was used for final evaluation.

Model checkpoint: `experiments/yolo_final/yolo26s_640_final/weights/best.pt`
