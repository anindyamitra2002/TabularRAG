
# Set the Python path
export PYTHONPATH=.
pip install -r requirements.txt
# Install and run Ollama
curl -fsSL https://ollama.com/install.sh | sh
sleep 1
ollama pull nomic-embed-text
ollama pull mistral:7b
