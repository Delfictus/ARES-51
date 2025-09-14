#!/bin/bash
# Build PRCT Docker image with proper permissions handling

set -euo pipefail

echo "🐳 PRCT Docker Image Builder"
echo "============================"

cd "$(dirname "$0")"

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

# Try to build without sudo first
echo "🔨 Attempting to build image..."
if docker build -t prct-h100-validation:latest . 2>/dev/null; then
    echo "✅ Build successful!"
else
    echo "⚠️  Build failed with current permissions"
    echo "🔐 Trying with sudo..."
    
    if command -v sudo &> /dev/null; then
        echo "Please enter your password for sudo access:"
        sudo docker build -t prct-h100-validation:latest .
    else
        echo "❌ Neither regular Docker nor sudo is available"
        echo ""
        echo "Solutions:"
        echo "1. Add user to docker group: sudo usermod -aG docker \$USER && newgrp docker"
        echo "2. Use rootless Docker"
        echo "3. Run as root user"
        exit 1
    fi
fi

echo ""
echo "🎉 PRCT Docker Image Build Complete!"
echo "===================================="

# Verify the image
echo "📋 Image verification:"
if docker images | grep prct-h100-validation; then
    echo "✅ Image created successfully"
    
    # Get image size
    IMAGE_SIZE=$(docker images prct-h100-validation:latest --format "table {{.Size}}" | tail -1)
    echo "📦 Image size: $IMAGE_SIZE"
    
    echo ""
    echo "🚀 Next steps:"
    echo "1. Test locally: docker run --rm prct-h100-validation:latest"
    echo "2. Push to hub: ./push_to_hub.sh"
    echo "3. Use in RunPod with custom image"
    
else
    echo "❌ Image verification failed"
    exit 1
fi