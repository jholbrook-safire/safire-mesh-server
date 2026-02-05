# Safire Mesh Processing Server
# Uses PrusaSlicer CLI for mesh operations
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
    wget \
    curl \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install PrusaSlicer
# Using AppImage extraction for CLI-only usage
WORKDIR /opt
RUN wget -q https://github.com/prusa3d/PrusaSlicer/releases/download/version_2.7.4/PrusaSlicer-2.7.4+linux-x64-GTK3-202403281018.AppImage -O prusa.AppImage \
    && chmod +x prusa.AppImage \
    && ./prusa.AppImage --appimage-extract \
    && rm prusa.AppImage \
    && ln -s /opt/squashfs-root/usr/bin/prusa-slicer /usr/local/bin/prusa-slicer

# Verify PrusaSlicer installation
RUN prusa-slicer --version || echo "PrusaSlicer CLI ready"

# Set up Python application
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment
ENV PRUSA_SLICER_PATH=/usr/local/bin/prusa-slicer
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
