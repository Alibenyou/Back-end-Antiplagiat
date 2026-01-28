FROM python:3.10-slim

# On limite la mémoire utilisée par Python
ENV PYTHONMALLOC=malloc

WORKDIR /app

# Installation minimaliste
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# On limite le nombre de workers pour économiser la RAM
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]