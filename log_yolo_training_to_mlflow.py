from pathlib import Path
import re
import pandas as pd
import mlflow


# Project paths
ROOT = Path(__file__).resolve().parent

RUN_DIR = (
    ROOT
    / "experiments"
    / "yolo_final"
    / "yolo26s_640_final"
)

RESULTS_CSV = RUN_DIR / "results.csv"
BEST_MODEL = RUN_DIR / "weights" / "best.pt"


# Check files
if not RESULTS_CSV.exists():
    raise FileNotFoundError(
        f"Training results not found: {RESULTS_CSV}"
    )

if not BEST_MODEL.exists():
    raise FileNotFoundError(
        f"Best model not found: {BEST_MODEL}"
    )


# Connect to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(
    "RoadSight AI - YOLO26s Road Damage Detection"
)


# Read training history
df = pd.read_csv(RESULTS_CSV)

# Remove spaces from CSV column names
df.columns = df.columns.str.strip()


with mlflow.start_run(
    run_name="YOLO26s Existing Training"
):

    # Log known training parameters
    mlflow.log_params({
        "model": "YOLO26s",
        "epochs_configured": 100,
        "image_size": 640,
        "batch_size": 4,
        "device": "0",
        "workers": 4,
        "pretrained": True,
        "patience": 20,
        "cos_lr": True,
        "close_mosaic": 10,
        "seed": 42,
        "deterministic": True,
    })

    mlflow.set_tag(
        "logging_type",
        "retrospective_existing_training"
    )

    # Log each numeric training metric by epoch
    for index, row in df.iterrows():

        if "epoch" in df.columns:
            epoch = int(row["epoch"])
        else:
            epoch = index

        for column in df.columns:

            if column.lower() == "epoch":
                continue

            value = pd.to_numeric(
                row[column],
                errors="coerce"
            )

            if pd.notna(value):

                metric_name = re.sub(
                    r"[^a-zA-Z0-9_./-]",
                    "_",
                    column
                )

                mlflow.log_metric(
                    metric_name,
                    float(value),
                    step=epoch
                )

    # Log the final test metrics
    mlflow.log_metrics({
        "test/precision": 0.585,
        "test/recall": 0.463,
        "test/mAP50": 0.474,
        "test/mAP50_95": 0.214,
    })

    # Upload training history and best weights
    mlflow.log_artifact(
        str(RESULTS_CSV),
        artifact_path="training_results"
    )

    mlflow.log_artifact(
        str(BEST_MODEL),
        artifact_path="model"
    )

    print("MLflow logging completed.")
    print("Run ID:", mlflow.active_run().info.run_id)