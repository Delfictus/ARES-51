#!/bin/bash
# PRCT Engine Release Build Script for Cloud Deployment
# Builds optimized binaries for H100 GPU validation

set -euo pipefail

echo "🔧 PRCT Engine Release Build for Cloud Deployment"
echo "================================================="

# Configuration
RUST_TARGET_DIR="/tmp/prct-release-build"
RELEASE_DIR="$(pwd)/validation/release"
BUILD_FEATURES="cuda-h100,tensor-cores,hbm3-optimization"

# Create release directory
mkdir -p "$RELEASE_DIR"
cd prct-engine

echo "📋 Build Configuration:"
echo "   Target Directory: $RUST_TARGET_DIR"
echo "   Release Directory: $RELEASE_DIR"
echo "   Features: $BUILD_FEATURES"
echo "   Rust Version: $(rustc --version)"
echo ""

# Set environment variables for optimal H100 compilation
export CARGO_TARGET_DIR="$RUST_TARGET_DIR"
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C codegen-units=1 -C panic=abort"
export CUDA_NVCC_FLAGS="--gpu-architecture=sm_90 --optimize=3 --use_fast_math"

echo "🛠️ Compiling PRCT Engine (Release Mode)..."
echo "   This may take 5-10 minutes for full optimization..."

# Build release version with all optimizations
cargo build --release --bins --features="$BUILD_FEATURES" --verbose

echo "✅ Compilation completed successfully!"
echo ""

# Copy binaries to release directory
echo "📦 Packaging binaries..."

# Main PRCT engine binary
if [ -f "$RUST_TARGET_DIR/release/prct-engine" ]; then
    cp "$RUST_TARGET_DIR/release/prct-engine" "$RELEASE_DIR/"
    echo "   ✅ prct-engine"
fi

# CASP16 data downloader
if [ -f "$RUST_TARGET_DIR/release/casp16-downloader" ]; then
    cp "$RUST_TARGET_DIR/release/casp16-downloader" "$RELEASE_DIR/"
    echo "   ✅ casp16-downloader"
fi

# Validation runner
if [ -f "$RUST_TARGET_DIR/release/prct-validator" ]; then
    cp "$RUST_TARGET_DIR/release/prct-validator" "$RELEASE_DIR/"
    echo "   ✅ prct-validator"
fi

# Report generator
if [ -f "$RUST_TARGET_DIR/release/report-generator" ]; then
    cp "$RUST_TARGET_DIR/release/report-generator" "$RELEASE_DIR/"
    echo "   ✅ report-generator"
fi

# Benchmark suite
if [ -f "$RUST_TARGET_DIR/release/benchmark-suite" ]; then
    cp "$RUST_TARGET_DIR/release/benchmark-suite" "$RELEASE_DIR/"
    echo "   ✅ benchmark-suite"
fi

echo ""

# Create binary information file
echo "📝 Creating binary information..."
cat > "$RELEASE_DIR/build_info.json" << EOF
{
  "build_timestamp": "$(date -Iseconds)",
  "rust_version": "$(rustc --version)",
  "cargo_version": "$(cargo --version)",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "$(git branch --show-current)",
  "build_features": "$BUILD_FEATURES",
  "target_architecture": "x86_64-unknown-linux-gnu",
  "optimization_level": "3",
  "cuda_version": "12.8.1",
  "target_gpu": "NVIDIA H100 PCIe",
  "memory_target": "80GB HBM3"
}
EOF

echo "   ✅ build_info.json"

# Create checksums for verification
echo "🔐 Generating checksums..."
cd "$RELEASE_DIR"
sha256sum * > checksums.sha256
echo "   ✅ checksums.sha256"

# Display binary sizes and information
echo ""
echo "📊 Binary Information:"
ls -lah "$RELEASE_DIR"

echo ""
echo "🎯 Binary Verification:"
for binary in prct-engine casp16-downloader prct-validator; do
    if [ -f "$binary" ]; then
        echo "   $binary:"
        echo "     Size: $(du -h "$binary" | cut -f1)"
        echo "     Type: $(file "$binary" | cut -d: -f2 | xargs)"
        if ldd "$binary" >/dev/null 2>&1; then
            echo "     Libraries: $(ldd "$binary" | wc -l) dependencies"
        fi
    fi
done

echo ""
echo "✅ Release build completed successfully!"
echo "📁 Binaries available in: $RELEASE_DIR"
echo "🚀 Ready for cloud deployment!"

# Create deployment package
echo ""
echo "📦 Creating deployment package..."
PACKAGE_NAME="prct-engine-$(date +%Y%m%d-%H%M%S).tar.gz"
cd "$(dirname "$RELEASE_DIR")"
tar -czf "$PACKAGE_NAME" -C release .

echo "   ✅ Deployment package: validation/$PACKAGE_NAME"
echo "   📏 Package size: $(du -h "$PACKAGE_NAME" | cut -f1)"

echo ""
echo "🎉 PRCT Engine ready for H100 cloud validation!"
echo "💡 Upload $PACKAGE_NAME to cloud instance and extract for execution"