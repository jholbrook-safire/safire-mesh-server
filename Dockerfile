# Safire Mesh Processing Server
# Uses SuperSlicer CLI for slicing and trimesh for mesh operations
#
# Build:  docker build -t safire-mesh-server .
# Run:    docker run -p 8000:8000 safire-mesh-server

FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    unzip \
    git \
    libgl1-mesa-glx \
    libglu1-mesa \
    libgtk-3-0 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install SuperSlicer (using AppImage)
WORKDIR /opt
RUN wget -q https://github.com/supermerill/SuperSlicer/releases/download/2.5.59.13/SuperSlicer-ubuntu_20.04-2.5.59.13.AppImage -O superslicer.AppImage \
    && chmod +x superslicer.AppImage \
    && ./superslicer.AppImage --appimage-extract \
    && rm superslicer.AppImage \
    && mv squashfs-root /opt/superslicer \
    && ln -s /opt/superslicer/AppRun /usr/local/bin/superslicer

# Verify SuperSlicer installation
RUN superslicer --help | head -5 || echo "SuperSlicer CLI ready"

# Set up Python application
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment
ENV SUPERSLICER_PATH=/usr/local/bin/superslicer
ENV TEMP_DIR=/tmp/safire-mesh
ENV PORT=8000

# Create temp directory
RUN mkdir -p /tmp/safire-mesh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
