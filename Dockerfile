FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Hugging Face Space Port
EXPOSE 7860

# Launch FastAPI Microservice
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
