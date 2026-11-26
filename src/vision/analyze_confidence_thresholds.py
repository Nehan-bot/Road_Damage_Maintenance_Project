
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(
    r"runs\detect\experiments\yolo_training\yolo26n_india_25epochs_balanced600\weights\best.pt"
)

DATA_YAML = Path("data/road_damage_balanced.yaml")

OUTPUT_DIR = Path(
    "experiments/confidence_analysis/balanced600"
)

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

IMAGE_SIZE = 416
DEVICE = "cpu"
BATCH_SIZE = 4
WORKERS = 0

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole"
}


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    records = []

    for threshold in THRESHOLDS:

        print(f"\nTesting confidence threshold: {threshold}")

        metrics = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            conf=threshold,
            plots=False,
            verbose=False
        )

        for class_id, class_name in CLASS_NAMES.items():

            records.append({
                "confidence": threshold,
                "class": class_name,
                "precision": float(metrics.box.p[class_id]),
                "recall": float(metrics.box.r[class_id]),
                "mAP50": float(metrics.box.ap50[class_id]),
                "mAP50_95": float(metrics.box.ap[class_id])
            })

        print(
            "Overall precision:",
            round(float(metrics.box.mp), 4)
        )

        print(
            "Overall recall:",
            round(float(metrics.box.mr), 4)
        )

    results = pd.DataFrame(records)

    output_file = OUTPUT_DIR / "confidence_threshold_results.csv"

    results.to_csv(output_file, index=False)

    print("\n========== TRANSVERSE CRACK RESULTS ==========")

    print(
        results[
            results["class"] == "Transverse crack"
        ].to_string(index=False)
    )

    print("\nResults saved:", output_file)


if __name__ == "__main__":
    main()
