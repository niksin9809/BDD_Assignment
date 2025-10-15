FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
COPY test.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless

