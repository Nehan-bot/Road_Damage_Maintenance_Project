
from pathlib import Path
from ultralytics import YOLO

MODEL = Path(
    r"runs\detect\experiments\yolo_training\yolo26n_india_25epochs_balanced600\weights\best.pt"
)

SOURCE = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class\val\images"
)

OUTPUT = Path(
    "experiments/transverse_inspection/balanced600"
)

def main():

    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    model = YOLO(str(MODEL))

    results = model.predict(
        source=str(SOURCE),
        imgsz=416,
        conf=0.10,
        device="cpu",
        batch=4,
        workers=0,
        save=True,
        project=str(OUTPUT.parent),
        name=OUTPUT.name,
        exist_ok=True,
        verbose=False
    )

    print("Prediction inspection completed.")
    print("Images processed:", len(results))
    print("Output:", OUTPUT.resolve())

if __name__ == "__main__":
    main()
