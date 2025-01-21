#!/bin/bash

# Function to log messages
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Install and run Ollama
log "Starting Ollama installation..."
if ! curl -fsSL https://ollama.com/install.sh | sh; then
    log "Failed to install Ollama."
    exit 1
fi
log "Ollama installation completed."

# Sleep for a short duration to ensure installation completes
sleep 5

# Check if ollama command is available
if ! command -v ollama &> /dev/null; then
    log "Ollama command not found. Installation may have failed."
    exit 1
fi

# Start the Ollama server in the background
log "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama server to start (you may need to adjust sleep duration)
log "Waiting for Ollama server to start..."
sleep 10

# Check if Ollama server is running
if ! pgrep -x "ollama" > /dev/null; then
    log "Ollama server did not start successfully."
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

# Pull the required Ollama model(s) during runtime
log "Pulling Ollama models..."
if ! ollama pull nomic-embed-text; then
    log "Failed to pull nomic-embed-text model."
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi

if ! ollama pull mistral:7b; then
    log "Failed to pull mistral:7b model."
    kill $OLLAMA_PID 2>/dev/null || true
    exit 1
fi
log "Models pulled successfully."

# Sleep for a short duration to ensure models are downloaded and ready
sleep 5

# Start Streamlit app
log "Starting Streamlit app..."
exec streamlit run --server.address 0.0.0.0 --server.port 8501 app.py