# Unit V DevOps: Standalone AI Prediction Engine
FROM python:3.9-slim

# ── Install Java (Required by PySpark) ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

# ── Install Python Dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Project Logic (now self-contained in U5) ──
COPY src/ ./src/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# ── Copy NCRB Data (now embedded in U5/data) ──
COPY data/ ./data/

# ── Environment ──
ENV FLASK_APP=src/app/app.py
ENV PYTHONUNBUFFERED=1
ENV CRIME_DATA_PATH=data/Crimes_in_india_2001-2013.csv

EXPOSE 5000

# ── Metadata ──
LABEL maintainer="AYUSH JHA"
LABEL description="Unit V DevOps: Self-Contained AI Crime Analytics Pipeline"

# ── Execute Startup Chain ──
CMD ["./entrypoint.sh"]
