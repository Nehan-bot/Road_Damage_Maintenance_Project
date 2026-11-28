
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "backend" / "nlp" / "dataset"
MODEL_DIR = PROJECT_ROOT / "backend" / "nlp" / "models"

TRAIN_FILE = DATA_DIR / "complaint_dataset_train.csv"
VALIDATION_FILE = DATA_DIR / "complaint_dataset_validation.csv"
TEST_FILE = DATA_DIR / "complaint_dataset_test.csv"


# ---------------------------------------------------------
# Model settings
# ---------------------------------------------------------

BASE_MODEL = "distilbert-base-uncased"

MAX_LENGTH = 192
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5


# ---------------------------------------------------------
# Label definitions
# Keep these labels exactly consistent with the CSV files.
# ---------------------------------------------------------

TASKS = {
    "damage_classifier": [
        "alligator crack",
        "longitudinal crack",
        "other",
        "pothole",
        "transverse crack",
    ],
    "urgency_classifier": [
        "High",
        "Low",
        "Medium",
    ],
}


# ---------------------------------------------------------
# Load and validate CSV files
# ---------------------------------------------------------

def load_dataset_file(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {
        "text",
        "damage_type",
        "urgency",
    }

    if set(dataframe.columns) != required_columns:
        raise ValueError(
            f"{file_path.name} must contain exactly these columns: "
            "text, damage_type, urgency"
        )

    if dataframe.isnull().any().any():
        raise ValueError(
            f"Missing values found in {file_path.name}"
        )

    dataframe["text"] = (
        dataframe["text"]
        .astype(str)
        .str.strip()
    )

    if (dataframe["text"].str.len() == 0).any():
        raise ValueError(
            f"Blank complaint text found in {file_path.name}"
        )

    return dataframe


def load_all_data():
    train_df = load_dataset_file(TRAIN_FILE)
    validation_df = load_dataset_file(VALIDATION_FILE)
    test_df = load_dataset_file(TEST_FILE)

    # Prevent exact duplicate complaint text across splits.
    train_texts = set(train_df["text"].str.lower())
    validation_texts = set(
        validation_df["text"].str.lower()
    )
    test_texts = set(test_df["text"].str.lower())

    if train_texts.intersection(validation_texts):
        raise ValueError(
            "Duplicate complaint text found between "
            "training and validation datasets."
        )

    if train_texts.intersection(test_texts):
        raise ValueError(
            "Duplicate complaint text found between "
            "training and test datasets."
        )

    if validation_texts.intersection(test_texts):
        raise ValueError(
            "Duplicate complaint text found between "
            "validation and test datasets."
        )

    print("\nDataset sizes:")
    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:      {len(test_df)}")

    return train_df, validation_df, test_df


# ---------------------------------------------------------
# Convert pandas data to Hugging Face Dataset
# ---------------------------------------------------------

def create_huggingface_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    label_to_id: dict,
):
    dataset_df = pd.DataFrame({
        "text": dataframe["text"].tolist(),
        "labels": [
            label_to_id[label]
            for label in dataframe[target_column]
        ],
    })

    return Dataset.from_pandas(
        dataset_df,
        preserve_index=False,
    )


# ---------------------------------------------------------
# Training metrics
# ---------------------------------------------------------

def calculate_metrics(prediction):
    logits, labels = prediction

    predicted_ids = np.argmax(
        logits,
        axis=1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predicted_ids,
        ),
        "macro_f1": f1_score(
            labels,
            predicted_ids,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# Train one classification model
# ---------------------------------------------------------

def train_one_model(
    task_name: str,
    target_column: str,
    labels: list,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    print("\n" + "=" * 60)
    print(f"Training task: {task_name}")
    print("=" * 60)

    task_output_dir = MODEL_DIR / task_name
    task_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    # Check that all dataset labels are supported.
    for split_name, dataframe in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        unknown_labels = (
            set(dataframe[target_column])
            - set(labels)
        )

        if unknown_labels:
            raise ValueError(
                f"Unknown labels in {split_name}: "
                f"{unknown_labels}"
            )

    print("Labels:", labels)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    train_dataset = create_huggingface_dataset(
        train_df,
        target_column,
        label_to_id,
    )

    validation_dataset = create_huggingface_dataset(
        validation_df,
        target_column,
        label_to_id,
    )

    test_dataset = create_huggingface_dataset(
        test_df,
        target_column,
        label_to_id,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    validation_dataset = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    test_dataset = test_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    training_arguments = TrainingArguments(
        output_dir=str(
            task_output_dir / "training_checkpoints"
        ),

        num_train_epochs=EPOCHS,

        learning_rate=LEARNING_RATE,
        weight_decay=0.01,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,

        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=25,

        fp16=torch.cuda.is_available(),

        report_to="none",

        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=calculate_metrics,
    )

    print("\nStarting training...")
    print("GPU available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU name:",
            torch.cuda.get_device_name(0),
        )

    trainer.train()

    # Save the best model selected using validation macro F1.
    trainer.save_model(
        str(task_output_dir)
    )

    tokenizer.save_pretrained(
        str(task_output_dir)
    )

    with open(
        task_output_dir / "label_map.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": {
                    str(key): value
                    for key, value in id_to_label.items()
                },
            },
            file,
            indent=4,
        )

    # Evaluate the final selected model on the test set.
    print("\nEvaluating on test dataset...")

    test_output = trainer.predict(
        test_dataset,
        metric_key_prefix="test",
    )

    test_logits = test_output.predictions
    true_ids = test_output.label_ids

    predicted_ids = np.argmax(
        test_logits,
        axis=1,
    )

    metrics = {
        "task": task_name,
        "target_column": target_column,
        "test_accuracy": accuracy_score(
            true_ids,
            predicted_ids,
        ),
        "test_macro_f1": f1_score(
            true_ids,
            predicted_ids,
            average="macro",
            zero_division=0,
        ),
        "test_macro_precision": precision_score(
            true_ids,
            predicted_ids,
            average="macro",
            zero_division=0,
        ),
        "test_macro_recall": recall_score(
            true_ids,
            predicted_ids,
            average="macro",
            zero_division=0,
        ),
    }

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_ids,
        predicted_ids,
        labels=list(range(len(labels))),
    )

    # Save test metrics.
    with open(
        task_output_dir / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save detailed per-class precision, recall and F1.
    with open(
        task_output_dir / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    # Save confusion matrix as CSV.
    confusion_dataframe = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    confusion_dataframe.to_csv(
        task_output_dir / "confusion_matrix.csv"
    )

    # Save test predictions for manual inspection.
    predictions_dataframe = pd.DataFrame({
        "text": test_df["text"].tolist(),
        "actual_label": [
            id_to_label[int(value)]
            for value in true_ids
        ],
        "predicted_label": [
            id_to_label[int(value)]
            for value in predicted_ids
        ],
    })

    predictions_dataframe.to_csv(
        task_output_dir / "test_predictions.csv",
        index=False,
    )

    print("\nTest results:")
    print(json.dumps(metrics, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            true_ids,
            predicted_ids,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")
    print(confusion_dataframe)

    return metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df, validation_df, test_df = load_all_data()

    all_metrics = {}

    all_metrics["damage_classifier"] = train_one_model(
        task_name="damage_classifier",
        target_column="damage_type",
        labels=TASKS["damage_classifier"],
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    all_metrics["urgency_classifier"] = train_one_model(
        task_name="urgency_classifier",
        target_column="urgency",
        labels=TASKS["urgency_classifier"],
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
    )

    summary_file = MODEL_DIR / "training_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_metrics,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("Models saved to:")
    print(MODEL_DIR)

    print("\nSummary saved to:")
    print(summary_file)


if __name__ == "__main__":
    main()
