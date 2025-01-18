
# Set the Python path
export PYTHONPATH=.

# Install and run Ollama
curl -fsSL https://ollama.com/install.sh | sh
sleep 1
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
