#!/bin/bash
# PRCT Docker Image Build Script

set -euo pipefail

echo "🐳 Building PRCT H100 Validation Docker Image"
echo "=============================================="

# Check if we're running with proper permissions
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker permission denied. Run one of these commands:"
    echo ""
    echo "Option 1: Add user to docker group (requires logout/login):"
    echo "  sudo usermod -aG docker \$USER"
    echo "  newgrp docker"
    echo ""
    echo "Option 2: Run with sudo:"
    echo "  sudo docker build -t prct-h100-validation:latest ."
    echo ""
    echo "Option 3: Use rootless Docker"
    echo ""
    exit 1
fi

# Build the image
echo "🔨 Building Docker image..."
docker build -t prct-h100-validation:latest . --progress=plain

# Verify the build
echo "✅ Verifying image..."
docker images | grep prct-h100-validation

# Test the image
echo "🧪 Quick test of the image..."
docker run --rm prct-h100-validation:latest echo "Image test successful"

echo ""
echo "🎉 PRCT Docker Image Build Complete!"
echo "===================================="
echo ""
echo "📋 Next Steps:"
echo "1. Test the image locally:"
echo "   docker run --gpus all -it prct-h100-validation:latest"
echo ""
echo "2. Tag for Docker Hub:"
echo "   docker tag prct-h100-validation:latest yourusername/prct-h100-validation:latest"
echo ""
echo "3. Push to Docker Hub:"
echo "   docker login"
echo "   docker push yourusername/prct-h100-validation:latest"
echo ""
echo "4. Use in RunPod:"
echo "   Custom Image: yourusername/prct-h100-validation:latest"
echo "   GPU: 8x NVIDIA H100 PCIe"
echo "   Storage: 100GB+"
echo ""
echo "🚀 Ready for instant PRCT validation deployment!"