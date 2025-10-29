# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching layers)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of your project files
COPY src ./src
COPY data ./data

# Create necessary directories (if not already in repo)
RUN mkdir -p data/raw data/processed data/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV PORT=8000

# Expose FastAPI port
EXPOSE ${PORT}

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
