
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BACKEND_ROOT = Path(__file__).resolve().parent
COMPLAINT_DB = BACKEND_ROOT / "complaints.db"

MODEL_ROOT = BACKEND_ROOT / "nlp" / "models"
DAMAGE_MODEL_PATH = MODEL_ROOT / "damage_classifier"
URGENCY_MODEL_PATH = MODEL_ROOT / "urgency_classifier"

router = APIRouter(
    prefix="/api/complaints",
    tags=["NLP Complaints"],
)

LOCATION_PATTERNS = [
    r"\bnear\s+([^,.!?;]+)",
    r"\bat\s+([^,.!?;]+)",
    r"\bin\s+([^,.!?;]+)",
    r"\bon\s+([^,.!?;]+)",
    r"\bopposite\s+([^,.!?;]+)",
    r"\bbehind\s+([^,.!?;]+)",
    r"\bbeside\s+([^,.!?;]+)",
]

DAMAGE_LABELS = [
    "alligator crack",
    "longitudinal crack",
    "other",
    "pothole",
    "transverse crack",
]

URGENCY_LABELS = [
    "High",
    "Low",
    "Medium",
]


class ComplaintRequest(BaseModel):
    text: str = Field(min_length=10, max_length=3000)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

damage_tokenizer = None
damage_model = None
urgency_tokenizer = None
urgency_model = None


def initialize_database():
    COMPLAINT_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(COMPLAINT_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_text TEXT NOT NULL,
                damage_type TEXT NOT NULL,
                urgency TEXT NOT NULL,
                location_text TEXT,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        # Add new columns only when they do not already exist.
        # Existing complaint rows and IDs are preserved.
        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(complaints)"
            ).fetchall()
        }

        new_columns = {
            "damage_confidence": "REAL",
            "urgency_confidence": "REAL",
            "nlp_model": "TEXT",
        }

        for column_name, column_type in new_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE complaints "
                    f"ADD COLUMN {column_name} {column_type}"
                )

        connection.commit()


def load_complaint_models():
    global damage_tokenizer
    global damage_model
    global urgency_tokenizer
    global urgency_model

    if not DAMAGE_MODEL_PATH.exists():
        raise RuntimeError(
            f"Damage model folder not found: {DAMAGE_MODEL_PATH}"
        )

    if not URGENCY_MODEL_PATH.exists():
        raise RuntimeError(
            f"Urgency model folder not found: {URGENCY_MODEL_PATH}"
        )

    damage_tokenizer = AutoTokenizer.from_pretrained(
        str(DAMAGE_MODEL_PATH)
    )
    damage_model = AutoModelForSequenceClassification.from_pretrained(
        str(DAMAGE_MODEL_PATH)
    )

    urgency_tokenizer = AutoTokenizer.from_pretrained(
        str(URGENCY_MODEL_PATH)
    )
    urgency_model = AutoModelForSequenceClassification.from_pretrained(
        str(URGENCY_MODEL_PATH)
    )

    damage_model.to(device)
    urgency_model.to(device)

    damage_model.eval()
    urgency_model.eval()


def get_model_label(model, predicted_id, fallback_labels):
    id2label = model.config.id2label

    label = id2label.get(predicted_id)

    if label is None:
        label = id2label.get(str(predicted_id))

    if label is None or str(label).upper() == f"LABEL_{predicted_id}":
        if predicted_id < len(fallback_labels):
            return fallback_labels[predicted_id]

        return "other"

    return str(label).strip()


def predict_with_model(text, tokenizer, model, fallback_labels):
    if tokenizer is None or model is None:
        raise RuntimeError("Complaint NLP models have not been loaded.")

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=192,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item())

    predicted_label = get_model_label(
        model,
        predicted_id,
        fallback_labels,
    )

    return predicted_label, confidence


def predict_damage_type(text):
    label, confidence = predict_with_model(
        text,
        damage_tokenizer,
        damage_model,
        DAMAGE_LABELS,
    )

    label = label.lower().strip()

    if label not in DAMAGE_LABELS:
        label = "other"

    return label, confidence


def predict_urgency(text):
    label, confidence = predict_with_model(
        text,
        urgency_tokenizer,
        urgency_model,
        URGENCY_LABELS,
    )

    label = label.strip().capitalize()

    if label not in URGENCY_LABELS:
        label = "Medium"

    return label, confidence


def extract_location(text):
    for pattern in LOCATION_PATTERNS:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            location = match.group(1).strip()
            location = re.sub(r"\s+", " ", location)
            return location[:150]

    return None


def create_summary(damage_type, location, urgency):
    if location:
        location_part = f" near {location}"
    else:
        location_part = ""

    return (
        f"{urgency} urgency complaint about {damage_type}"
        f"{location_part}."
    )


def analyze_complaint(text):
    damage_type, damage_confidence = predict_damage_type(text)
    urgency, urgency_confidence = predict_urgency(text)

    location = extract_location(text)

    summary = create_summary(
        damage_type,
        location,
        urgency,
    )

    return {
        "damage_types": [damage_type],
        "urgency": urgency,
        "location_text": location,
        "summary": summary,
        "damage_confidence": round(damage_confidence, 4),
        "urgency_confidence": round(urgency_confidence, 4),
        "location_is_verified": False,
        "urgency_is_estimated": True,
        "nlp_model": "DistilBERT fine-tuned",
    }


@router.on_event("startup")
def startup_complaints():
    initialize_database()
    load_complaint_models()


@router.post("/analyze")
def analyze_and_save(request: ComplaintRequest):
    text = request.text.strip()

    if len(text) < 10:
        raise HTTPException(
            status_code=400,
            detail="Please enter a more detailed complaint.",
        )

    try:
        result = analyze_complaint(text)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Complaint analysis failed: {str(error)}",
        )

    primary_damage = result["damage_types"][0]
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(COMPLAINT_DB) as connection:
        cursor = connection.execute(
            """
            INSERT INTO complaints (
                complaint_text,
                damage_type,
                urgency,
                location_text,
                summary,
                created_at,
                damage_confidence,
                urgency_confidence,
                nlp_model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                text,
                primary_damage,
                result["urgency"],
                result["location_text"],
                result["summary"],
                created_at,
                result["damage_confidence"],
                result["urgency_confidence"],
                result["nlp_model"],
            ),
        )

        complaint_id = cursor.lastrowid
        connection.commit()

    return {
        "id": complaint_id,
        "complaint_text": text,
        **result,
        "created_at": created_at,
    }


@router.get("")
def complaint_history():
    with sqlite3.connect(COMPLAINT_DB) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT *
            FROM complaints
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

    return [dict(row) for row in rows]
