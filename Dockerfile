FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04
RUN apt-get update && apt-get install -y \
    python3.10 \ 
    python3-pip \
    python3-venv \
    git

WORKDIR /app

RUN python3 -m venv venv

COPY requirements.txt .

RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt


COPY . .

#container listens on port 8000
EXPOSE 8000

CMD ["sh", "-c", ". venv/bin/activate && uvicorn api:app --host 0.0.0.0 --port 8000"]

