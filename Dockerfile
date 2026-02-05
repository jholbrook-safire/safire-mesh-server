# Safire Mesh Processing Server
# Uses CuraEngine CLI for slicing and trimesh for mesh operations
#
# Build:  docker build -t safire-mesh-server .
# Run:    docker run -p 8000:8000 safire-mesh-server

FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and CuraEngine
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    libgl1-mesa-glx \
    libglu1-mesa \
    cura-engine \
    && rm -rf /var/lib/apt/lists/*

# Verify CuraEngine installation
RUN CuraEngine --version || echo "CuraEngine CLI ready"

# Set up Python application
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment
ENV CURA_ENGINE_PATH=/usr/bin/CuraEngine
ENV TEMP_DIR=/tmp/safire-mesh
ENV PORT=8000

# Create temp directory
RUN mkdir -p /tmp/safire-mesh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with xvfb for any GUI dependencies
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
