# Use Python slim image as base
FROM python:3.10-slim

# Install system dependencies and wget (for downloading Ollama)
RUN apt-get update && \
    apt-get install -y \
    curl \
    procps \
    git \
    wget \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY src/ ./src/
COPY *.py ./
COPY execute.sh ./execute.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Create directory for Ollama models
RUN mkdir -p /root/.ollama

# Expose ports for both Streamlit and Ollama
EXPOSE 8501 11434

# Make sure execute.sh is executable
RUN chmod +x ./execute.sh

# Set the entrypoint
ENTRYPOINT ["./execute.sh"]