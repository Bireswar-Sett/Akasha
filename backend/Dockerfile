FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8080

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY . .

# Expose port (Cloud Run sets PORT automatically, defaults to 8080)
EXPOSE 8080

# Command to launch FastAPI with Uvicorn binding to 0.0.0.0:PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
