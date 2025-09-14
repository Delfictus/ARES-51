#!/bin/bash
# Push PRCT Docker Image to Docker Hub

set -euo pipefail

echo "📤 Pushing PRCT Image to Docker Hub"
echo "==================================="

# Check if image exists
if ! docker images | grep -q prct-h100-validation; then
    echo "❌ Image 'prct-h100-validation:latest' not found"
    echo "Build it first with: ./build_image.sh"
    exit 1
fi

# Get Docker Hub username
read -p "Enter your Docker Hub username: " DOCKER_USERNAME

if [[ -z "$DOCKER_USERNAME" ]]; then
    echo "❌ Docker Hub username required"
    exit 1
fi

echo "🏷️ Tagging image for Docker Hub..."
docker tag prct-h100-validation:latest $DOCKER_USERNAME/prct-h100-validation:latest

echo "🔐 Logging in to Docker Hub..."
docker login

echo "📤 Pushing image to Docker Hub..."
docker push $DOCKER_USERNAME/prct-h100-validation:latest

echo ""
echo "✅ Push Complete!"
echo "=================="
echo ""
echo "🎯 Your custom image is now available:"
echo "   $DOCKER_USERNAME/prct-h100-validation:latest"
echo ""
echo "🚀 Use in RunPod:"
echo "   1. Launch new instance"
echo "   2. Custom Image: $DOCKER_USERNAME/prct-h100-validation:latest"
echo "   3. GPU: 8x NVIDIA H100 PCIe"
echo "   4. Launch and run: ./run_casp16_validation.sh"
echo ""
echo "💰 Estimated validation cost: $500-1000 for breakthrough results"
echo "🏆 Ready for revolutionary protein folding validation!"