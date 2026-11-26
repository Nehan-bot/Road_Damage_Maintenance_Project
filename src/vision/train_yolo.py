
from pathlib import Path
from time import time

import torch
from ultralytics import YOLO


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_YAML = Path("data/road_damage_balanced.yaml")

MODEL_NAME = "yolo26n.pt"

EPOCHS = 25
IMAGE_SIZE = 416
BATCH_SIZE = 8

DEVICE = "cpu"
WORKERS = 0

SEED = 42

PROJECT_DIR = "experiments/yolo_training"
EXPERIMENT_NAME = "yolo26n_india_25epochs_balanced600"


# --------------------------------------------------
# 2. CPU OPTIMIZATION
# --------------------------------------------------

# Ryzen 9 7940HS has 8 CPU cores / 16 threads.
# Start with 8 PyTorch threads and monitor utilization.

torch.set_num_threads(8)

try:
    torch.set_num_interop_threads(4)
except RuntimeError:
    pass


# --------------------------------------------------
# 3. MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    print("\n========== TRAINING CONFIGURATION ==========")

    print("Dataset:", DATA_YAML.resolve())
    print("Model:", MODEL_NAME)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Device:", DEVICE)
    print("CPU threads:", torch.get_num_threads())

    # Load pretrained model
    model = YOLO(MODEL_NAME)

    print("\nStarting balanced dataset training...")

    start_time = time()

    # --------------------------------------------------
    # 4. TRAINING
    # --------------------------------------------------

    results = model.train(

        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=WORKERS,

        fraction=1.0,
        patience=8,

        seed=SEED,
        pretrained=True,

        cache=False,
        plots=True,
        save=True,

        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,

        exist_ok=False
    )

    training_time = (time() - start_time) / 3600

    # Get actual checkpoint location
    best_model_path = (
        Path(model.trainer.save_dir)
        / "weights"
        / "best.pt"
    )

    print("\n========== TRAINING COMPLETED ==========")

    print(f"Training time: {training_time:.2f} hours")
    print("Best model:", best_model_path.resolve())

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_model_path}"
        )

    # --------------------------------------------------
    # 5. FINAL VALIDATION
    # --------------------------------------------------

    print("\nLoading best checkpoint for final validation...")

    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(

        data=str(DATA_YAML),
        split="val",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=WORKERS
    )

    print("\n========== FINAL VALIDATION METRICS ==========")

    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    print("\nBalanced experiment completed.")


if __name__ == "__main__":
    main()
