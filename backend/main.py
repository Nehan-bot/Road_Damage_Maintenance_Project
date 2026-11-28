
from pathlib import Path
import io
import uuid

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from backend.database import (
    init_db,
    save_inspection,
    list_inspections,
    get_inspection,
)
from backend.complaints import router as complaints_router


PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "yolo_final"
    / "yolo26s_640_final"
    / "weights"
    / "best.pt"
)

PRIORITY_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "random_forest"
    / "random_forest_priority.joblib"
)

CLASS_NAMES = {
    0: "longitudinal crack",
    1: "transverse crack",
    2: "alligator crack",
    3: "pothole",
}

PRIORITY_FEATURE_NAMES = [
    "total_detections",
    "longitudinal crack_count",
    "longitudinal crack_mean_confidence",
    "transverse crack_count",
    "transverse crack_mean_confidence",
    "alligator crack_count",
    "alligator crack_mean_confidence",
    "pothole_count",
    "pothole_mean_confidence",
]


app = FastAPI(
    title="RoadSight AI API",
    description=(
        "Road damage detection and exploratory "
        "maintenance-priority estimation"
    ),
    version="1.1.0",
)

# Register the complaint-analysis API.
app.include_router(complaints_router)


@app.on_event("startup")
def startup_event():
    init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model weights not found: {YOLO_MODEL_PATH}"
    )

if not PRIORITY_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Random Forest model not found: {PRIORITY_MODEL_PATH}"
    )

yolo_model = YOLO(str(YOLO_MODEL_PATH))
priority_model = joblib.load(PRIORITY_MODEL_PATH)


def create_priority_features(detections):
    """
    Build the exact 9 features expected by the trained
    Random Forest.

    Mean confidence is 0.0 when a class has no detections.
    """
    class_names = [
        "longitudinal crack",
        "transverse crack",
        "alligator crack",
        "pothole",
    ]

    features = {
        "total_detections": len(detections),
    }

    for class_name in class_names:
        class_detections = [
            detection
            for detection in detections
            if detection["class_name"] == class_name
        ]

        confidences = [
            detection["confidence"]
            for detection in class_detections
        ]

        features[f"{class_name}_count"] = len(class_detections)

        features[f"{class_name}_mean_confidence"] = (
            float(np.mean(confidences))
            if confidences
            else 0.0
        )

    return features


def predict_priority(detections):
    """Return the RF priority, confidence, and class probabilities."""
    feature_dict = create_priority_features(detections)

    feature_values = [
        feature_dict[name]
        for name in PRIORITY_FEATURE_NAMES
    ]

    feature_array = np.asarray(
        [feature_values],
        dtype=float,
    )

    predicted_priority = str(
        priority_model.predict(feature_array)[0]
    )

    probabilities_array = priority_model.predict_proba(
        feature_array
    )[0]

    model_classes = [
        str(class_name)
        for class_name in priority_model.classes_
    ]

    probabilities = {
        class_name: round(float(probability), 4)
        for class_name, probability in zip(
            model_classes,
            probabilities_array,
        )
    }

    priority_confidence = probabilities.get(
        predicted_priority,
        0.0,
    )

    return {
        "priority": predicted_priority,
        "priority_confidence": priority_confidence,
        "priority_probabilities": probabilities,
        "priority_features": feature_dict,
    }


@app.get("/")
def root():
    return {
        "application": "RoadSight AI",
        "message": "Road damage detection API is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "yolo_model_loaded": True,
        "yolo_model": YOLO_MODEL_PATH.name,
        "priority_model_loaded": True,
        "priority_model": PRIORITY_MODEL_PATH.name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    results = yolo_model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(
                        class_id,
                        result.names.get(class_id, "unknown"),
                    ),
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    priority_result = predict_priority(detections)

    stored_filename = f"{uuid.uuid4().hex}.jpg"
    stored_image_path = UPLOAD_DIR / stored_filename

    image.save(
        stored_image_path,
        format="JPEG",
        quality=95,
    )

    relative_image_path = stored_image_path.relative_to(
        PROJECT_ROOT
    ).as_posix()

    saved_inspection = save_inspection(
        filename=file.filename or "uploaded_image",
        image_width=image.width,
        image_height=image.height,
        detections=detections,
        image_path=relative_image_path,
        priority=priority_result["priority"],
        priority_confidence=priority_result[
            "priority_confidence"
        ],
        priority_probabilities=priority_result[
            "priority_probabilities"
        ],
    )

    return {
        "id": saved_inspection["id"],
        "created_at": saved_inspection["created_at"],
        "filename": file.filename or "uploaded_image",
        "image_width": image.width,
        "image_height": image.height,
        "detection_count": len(detections),
        "detections": detections,
        "priority": priority_result["priority"],
        "priority_confidence": priority_result[
            "priority_confidence"
        ],
        "priority_probabilities": priority_result[
            "priority_probabilities"
        ],
        "priority_features": priority_result[
            "priority_features"
        ],
    }


@app.get("/inspections")
def inspection_history():
    return list_inspections()


@app.get("/inspections/{inspection_id}")
def inspection_detail(inspection_id: int):
    inspection = get_inspection(inspection_id)

    if inspection is None:
        raise HTTPException(
            status_code=404,
            detail="Inspection not found.",
        )

    return inspection


@app.get("/inspections/{inspection_id}/image")
def inspection_image(inspection_id: int):
    inspection = get_inspection(inspection_id)

    if inspection is None:
        raise HTTPException(
            status_code=404,
            detail="Inspection not found.",
        )

    image_path = inspection.get("image_path")

    if not image_path:
        raise HTTPException(
            status_code=404,
            detail="No saved image is available for this inspection.",
        )

    full_path = (PROJECT_ROOT / image_path).resolve()

    if not full_path.is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(
            status_code=400,
            detail="Invalid image path.",
        )

    if not full_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Saved image file was not found.",
        )

    return FileResponse(
        full_path,
        media_type="image/jpeg",
        filename=inspection["filename"],
    )
