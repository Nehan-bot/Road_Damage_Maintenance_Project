
from ultralytics import YOLO
from pathlib import Path

model_path = Path(
    "runs/detect/experiments/yolo_baseline/"
    "yolo26n_quick_baseline/weights/best.pt"
)

test_images = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class\test\images"
)

model = YOLO(str(model_path))

results = model.predict(
    source=str(test_images),
    imgsz=416,
    conf=0.15,
    save=True,
    device="cpu"
)

print("Prediction completed.")
print("Number of images processed:", len(results))
print("Check the runs/detect/predict folder.")
