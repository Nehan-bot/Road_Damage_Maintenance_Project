
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System libraries needed by OpenCV and scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch to avoid requiring an NVIDIA GPU in the container
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and supporting modules
COPY app/ ./app/
COPY database/ ./database/
COPY src/ ./src/

# Trained model files from your local project
COPY experiments/yolo_final/yolo26s_640_final/weights/best.pt \
     ./experiments/yolo_final/yolo26s_640_final/weights/best.pt

COPY experiments/random_forest/random_forest_priority.joblib \
     ./experiments/random_forest/random_forest_priority.joblib

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
