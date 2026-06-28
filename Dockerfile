FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    CASE_FINDER_DB=/store/cases.db \
    CASE_FINDER_DATA_DIR=/store/data

# easyocr/opencv が必要とする共有ライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
# モデルは初回起動/取り込み時にダウンロードされ、/root 配下のボリュームに永続化されます
CMD ["python", "app.py"]
