# Build stage for Ollama
FROM ollama/ollama as ollama

# Final stage
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy Ollama binary from build stage
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY src/ ./src/
COPY *.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for CPU-only operation
ENV CUDA_VISIBLE_DEVICES=""
ENV OLLAMA_HOST="0.0.0.0"

# Download Ollama models with CPU-only flag
RUN ollama serve & \
    sleep 5 && \
    OLLAMA_CPU=1 ollama pull nomic-embed-text && \
    OLLAMA_CPU=1 ollama pull llama3.2:3b && \
    pkill ollama

# Set Python path
ENV PYTHONPATH=/app

# Expose ports for Streamlit and Ollama
EXPOSE 8501 11434

# Create startup script with CPU-only configuration
RUN echo '#!/bin/bash\nOLLAMA_CPU=1 ollama serve & \nsleep 5 && streamlit run --server.port 8501 app.py' > /app/start.sh && \
    chmod +x /app/start.sh

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]