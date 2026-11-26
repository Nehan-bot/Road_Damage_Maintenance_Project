
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# Project paths
MODEL_PATH = Path(
    "runs/detect/experiments/yolo_final/"
    "yolo26n_rtx4070_corrected_balanced600/"
    "weights/best.pt"
)

TRAIN_IMAGES = Path(
    "data/raw/road_images/"
    "RDD2022_India_4Class_Corrected_Balanced600/"
    "train/images"
)

OUTPUT_DIR = Path("data/processed/vision")

# Prediction settings
IMAGE_SIZE = 416
CONFIDENCE = 0.25
BATCH_SIZE = 8
DEVICE = 0

# Validate paths
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TRAIN_IMAGES.exists():
    raise FileNotFoundError(f"Training images not found: {TRAIN_IMAGES}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
print("Loading final YOLO model...")
model = YOLO(str(MODEL_PATH))
class_names = model.names

print("Classes:", class_names)

# Run inference on training images
print("\nGenerating training image features...")

results = model.predict(
    source=str(TRAIN_IMAGES),
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    conf=CONFIDENCE,
    stream=True,
    save=False,
    verbose=False
)

image_records = []
detection_records = []

for result in results:
    image_name = Path(result.path).name

    class_counts = {
        name: 0 for name in class_names.values()
    }

    class_confidences = {
        name: [] for name in class_names.values()
    }

    boxes = result.boxes

    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(
            xyxy, confidences, class_ids
        ):
            class_name = class_names[class_id]

            class_counts[class_name] += 1
            class_confidences[class_name].append(
                float(confidence)
            )

            detection_records.append({
                "image_name": image_name,
                "class_id": class_id,
                "damage_class": class_name,
                "confidence": float(confidence),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            })

    image_record = {
        "image_name": image_name,
        "total_detections": sum(class_counts.values())
    }

    for class_name, count in class_counts.items():
        image_record[f"{class_name}_count"] = count

        scores = class_confidences[class_name]

        image_record[f"{class_name}_mean_confidence"] = (
            sum(scores) / len(scores) if scores else 0.0
        )

    image_records.append(image_record)

# Save outputs
features_df = pd.DataFrame(image_records)
detections_df = pd.DataFrame(detection_records)

features_file = (
    OUTPUT_DIR / "yolo_train_image_features.csv"
)

detections_file = (
    OUTPUT_DIR / "yolo_train_detections.csv"
)

features_df.to_csv(features_file, index=False)
detections_df.to_csv(detections_file, index=False)

# Summary
print("\nTRAIN PREDICTION EXPORT COMPLETED")
print("---------------------------------")
print("Images processed:", len(features_df))
print("Total detections:", len(detections_df))
print("Images with no detections:",
      (features_df["total_detections"] == 0).sum())

print("\nClass totals:")
for class_name in class_names.values():
    print(
        class_name,
        features_df[f"{class_name}_count"].sum()
    )

print("\nSaved files:")
print(features_file)
print(detections_file)
