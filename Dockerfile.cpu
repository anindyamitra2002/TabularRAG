# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY src/ ./src/
COPY *.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501 11434

# Create startup script in a different location
RUN echo '#!/bin/bash\n\
# Start Streamlit\n\
exec streamlit run --server.address 0.0.0.0 --server.port 8501 app.py\n\
' > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/start.sh"]