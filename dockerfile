FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#container listens on port 8000
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0" ,"--port", "8000"]
