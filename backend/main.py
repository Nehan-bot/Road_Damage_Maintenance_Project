
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
import io


# Find the project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to your trained YOLO26s model.
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "yolo_final"
    / "yolo26s_640_final"
    / "weights"
    / "best.pt"
)

CLASS_NAMES = {
    0: "longitudinal crack",
    1: "transverse crack",
    2: "alligator crack",
    3: "pothole",
}

app = FastAPI(
    title="RoadSight AI API",
    description="Road damage detection using the trained YOLO26s model",
    version="1.0.0",
)

# Allow the Next.js frontend to call this API during local development.
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

# Load the model once when the API starts.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model weights were not found at: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


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
        "model_loaded": True,
        "model": MODEL_PATH.name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check that the uploaded file is an image.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    contents = await file.read()

    # Limit uploads to 10 MB.
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # Run the real trained YOLO model.
    results = model.predict(
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

    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "detection_count": len(detections),
        "detections": detections,
    }
