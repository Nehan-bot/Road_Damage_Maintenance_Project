
from ultralytics import YOLO
from pathlib import Path

model = YOLO(
    "runs/detect/experiments/yolo_baseline/"
    "yolo26n_quick_baseline/weights/best.pt"
)

val_images = Path(
    r"C:\College\M.tech\Projects\ML S1\Code\Road_Damage_Maintenance_Project\data\raw\road_images\RDD2022_India\val\images"
)

model.predict(
    source=str(val_images),
    imgsz=416,
    conf=0.25,
    iou=0.5,
    save=True,
    device="cpu",
    project="experiments/prediction_review",
    name="validation_conf025"
)

print("Validation prediction review completed.")
