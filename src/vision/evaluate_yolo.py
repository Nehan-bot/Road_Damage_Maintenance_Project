
from pathlib import Path
from ultralytics import YOLO


# -----------------------------------
# 1. Dataset configuration
# -----------------------------------

DATA_YAML = Path(
    r"C:\College\M.tech\Datasets\RDD2022\RDD2022_India_4Class\data.yaml"
)


# -----------------------------------
# 2. Trained model location
# -----------------------------------

MODEL_PATH = Path(
    r"runs\detect\experiments\yolo_training\yolo26n_india_25epochs\weights\best.pt"
)


# -----------------------------------
# 3. Check required files
# -----------------------------------

def main():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("Dataset configuration found.")
    print("Trained model found.")


    # -----------------------------------
    # 4. Load trained YOLO model
    # -----------------------------------

    print("\nLoading trained YOLO26n model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")


    # -----------------------------------
    # 5. Evaluate on validation dataset
    # -----------------------------------

    print("\nStarting validation...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=416,
        batch=4,
        device="cpu",
        plots=True,
        project="experiments/yolo_evaluation",
        name="india_25epochs_validation"
    )


    # -----------------------------------
    # 6. Display overall metrics
    # -----------------------------------

    print("\n========== VALIDATION RESULTS ==========")

    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")
    print(f"mAP50         : {metrics.box.map50:.4f}")
    print(f"mAP50-95      : {metrics.box.map:.4f}")


    # -----------------------------------
    # 7. Display per-class metrics
    # -----------------------------------

    class_names = {
        0: "Longitudinal crack",
        1: "Transverse crack",
        2: "Alligator crack",
        3: "Pothole"
    }

    print("\n========== CLASS-WISE RESULTS ==========")

    for class_id, class_name in class_names.items():

        if class_id < len(metrics.box.p):

            precision = metrics.box.p[class_id]
            recall = metrics.box.r[class_id]
            map50 = metrics.box.ap50[class_id]
            map50_95 = metrics.box.ap[class_id].mean()

            print(f"\n{class_name}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"mAP50     : {map50:.4f}")
            print(f"mAP50-95  : {map50_95:.4f}")


    # -----------------------------------
    # 8. Completion message
    # -----------------------------------

    print("\n========================================")
    print("YOLO MODEL EVALUATION COMPLETED")
    print("========================================")

    print(
        "\nEvaluation outputs saved in:"
        "\nexperiments/yolo_evaluation/"
        "india_25epochs_validation"
    )


if __name__ == "__main__":
    main()
