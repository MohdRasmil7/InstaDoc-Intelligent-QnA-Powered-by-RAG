FROM python:3.8-slim

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --only-binary=:all: numpy && \
    pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py"]