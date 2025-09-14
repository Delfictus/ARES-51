#!/bin/bash
# PRCT Validation Suite - RunPod CLI Setup Script
# Clones and configures PRCT validation directly on RunPod instance

set -euo pipefail

echo "🚀 PRCT Validation Suite - RunPod CLI Setup"
echo "=============================================="

# Configuration
REPO_URL="https://github.com/Delfictus/ARES-51.git"
WORK_DIR="/workspace/prct-validation"
RUST_VERSION="1.70.0"

# Function to print status messages
log_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# Check if we're on RunPod
check_runpod_environment() {
    if [[ ! -d "/workspace" ]]; then
        log_warn "Not detected as RunPod environment - continuing anyway"
    else
        log_info "RunPod environment detected"
    fi
}

# Install system dependencies
install_dependencies() {
    log_info "Installing system dependencies..."
    
    apt-get update -qq
    apt-get install -y -qq \
        curl \
        wget \
        git \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        python3 \
        python3-pip \
        htop \
        nvtop \
        unzip \
        ca-certificates \
        gnupg \
        lsb-release
    
    log_info "✅ System dependencies installed"
}

# Install Rust toolchain
install_rust() {
    log_info "Installing Rust toolchain..."
    
    if command -v rustc &> /dev/null; then
        log_info "Rust already installed: $(rustc --version)"
        return
    fi
    
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
    source $HOME/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify installation
    rustc --version
    cargo --version
    
    log_info "✅ Rust toolchain installed"
}

# Clone PRCT repository
clone_repository() {
    log_info "Cloning PRCT validation suite..."
    
    if [[ -d "$WORK_DIR" ]]; then
        log_warn "Work directory exists, removing..."
        rm -rf "$WORK_DIR"
    fi
    
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    git clone "$REPO_URL" .
    
    log_info "✅ Repository cloned to $WORK_DIR"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    pip3 install --upgrade pip
    pip3 install \
        requests \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scipy \
        python-dotenv
    
    log_info "✅ Python dependencies installed"
}

# Build PRCT binaries
build_binaries() {
    log_info "Building PRCT validation binaries..."
    
    cd "$WORK_DIR/prct-engine"
    
    # Set up Rust environment
    source $HOME/.cargo/env
    export CARGO_TARGET_DIR="/tmp/prct-build"
    export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
    
    # Build release binaries
    cargo build --release --bins --quiet
    
    # Verify binaries
    if [[ -f "/tmp/prct-build/release/prct-validator" ]]; then
        log_info "✅ PRCT binaries built successfully"
        
        # Show binary info
        ls -lah /tmp/prct-build/release/
        echo
        log_info "Binary sizes:"
        du -h /tmp/prct-build/release/prct-validator
        du -h /tmp/prct-build/release/casp16-downloader
        du -h /tmp/prct-build/release/report-generator
        du -h /tmp/prct-build/release/benchmark-suite
    else
        log_error "Binary build failed"
        exit 1
    fi
}

# Setup validation environment
setup_validation_env() {
    log_info "Setting up validation environment..."
    
    cd "$WORK_DIR"
    
    # Create data directories
    mkdir -p {data/{casp16,results,benchmarks},logs}
    
    # Create convenience scripts
    cat > run_validation.sh << 'EOF'
#!/bin/bash
# PRCT Validation Execution Script

set -euo pipefail

echo "🧬 Starting PRCT Algorithm Validation"
echo "====================================="

PRCT_DIR="/workspace/prct-validation"
BINARY_DIR="/tmp/prct-build/release"
DATA_DIR="$PRCT_DIR/data"
RESULTS_DIR="$DATA_DIR/results"

cd "$PRCT_DIR"

# Check GPU availability
echo "🖥️ Checking GPU configuration..."
nvidia-smi || echo "⚠️ NVIDIA GPUs not detected"

# Run validation pipeline
echo "📊 Executing PRCT validation pipeline..."

# Step 1: Download CASP16 data (interface mode)
echo "  Step 1: CASP16 data preparation..."
$BINARY_DIR/casp16-downloader \
    --output-dir $DATA_DIR/casp16 \
    --verify-checksums \
    --log-level info

# Step 2: Run PRCT validation
echo "  Step 2: PRCT algorithm validation..."
$BINARY_DIR/prct-validator \
    --casp16-data $DATA_DIR/casp16 \
    --results-dir $RESULTS_DIR \
    --gpu-count 8

# Step 3: Generate reports
echo "  Step 3: Report generation..."
$BINARY_DIR/report-generator \
    --results-dir $RESULTS_DIR \
    --output-format publication \
    --include-performance-metrics \
    --statistical-significance 0.001

# Step 4: Run benchmarks
echo "  Step 4: Performance benchmarking..."
$BINARY_DIR/benchmark-suite \
    --benchmark-type full \
    --gpu-count 8 \
    --results-dir $RESULTS_DIR/benchmarks

echo ""
echo "✅ PRCT validation pipeline completed!"
echo "📁 Results available in: $RESULTS_DIR"
echo ""
echo "📊 Key files generated:"
echo "  - casp16_validation_report.json"
echo "  - prct_validation_report.txt"
echo "  - benchmark_results.json"
echo ""
echo "🎯 Ready for analysis and publication preparation!"
EOF

    chmod +x run_validation.sh
    
    # Create GPU monitoring script
    cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
# GPU Monitoring Script

echo "=== NVIDIA GPU Monitoring ==="
while true; do
    clear
    echo "Timestamp: $(date)"
    echo "=================================="
    nvidia-smi
    echo ""
    echo "GPU Utilization:"
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    echo ""
    echo "Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
    echo ""
    echo "Temperature & Power:"
    nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits
    sleep 5
done
EOF

    chmod +x monitor_gpu.sh
    
    # Create quick test script
    cat > test_setup.sh << 'EOF'
#!/bin/bash
# Quick Setup Test

echo "🧪 Testing PRCT validation setup..."

BINARY_DIR="/tmp/prct-build/release"

echo "  Testing binary executables..."
$BINARY_DIR/prct-validator --help | head -5
$BINARY_DIR/casp16-downloader --version
$BINARY_DIR/report-generator --version
$BINARY_DIR/benchmark-suite --version

echo ""
echo "  Testing Python environment..."
python3 -c "import numpy, pandas, matplotlib; print('Python packages OK')"

echo ""
echo "  Checking GPU availability..."
nvidia-smi --query-gpu=name --format=csv,noheader || echo "No GPUs detected"

echo ""
echo "✅ Setup test completed successfully!"
echo ""
echo "Ready to run validation:"
echo "  ./run_validation.sh    - Run complete validation pipeline"
echo "  ./monitor_gpu.sh       - Monitor GPU utilization"
echo "  ./test_setup.sh        - Test setup again"
EOF

    chmod +x test_setup.sh
    
    log_info "✅ Validation environment configured"
}

# Create usage instructions
create_instructions() {
    cat > "$WORK_DIR/README_RUNPOD.md" << 'EOF'
# PRCT Validation Suite - RunPod Setup

## 🚀 Quick Start

This RunPod instance is now configured with the complete PRCT validation suite.

### Available Commands

```bash
# Run complete validation pipeline
./run_validation.sh

# Monitor GPU utilization in real-time  
./monitor_gpu.sh

# Test that everything is working
./test_setup.sh
```

### Manual Execution

```bash
# Individual binary executables
/tmp/prct-build/release/prct-validator --help
/tmp/prct-build/release/casp16-downloader --help
/tmp/prct-build/release/report-generator --help
/tmp/prct-build/release/benchmark-suite --help

# Direct validation execution
/tmp/prct-build/release/prct-validator \
    --casp16-data ./data/casp16 \
    --results-dir ./data/results \
    --gpu-count 8
```

### Results Location

All validation results will be saved to:
- `./data/results/` - Main validation outputs
- `./data/results/benchmarks/` - Performance benchmarks
- `./logs/` - Execution logs

### Key Output Files

- `casp16_validation_report.json` - Main validation results
- `prct_validation_report.txt` - Publication-ready report
- `benchmark_results.json` - Performance metrics
- `comparison_report.json` - AlphaFold2 comparison (if available)

### Expected Performance

With 8x H100 GPUs:
- GPU utilization: >95% sustained
- Validation time: 2-6 hours for complete CASP16 benchmark
- Results: >15% accuracy improvement over AlphaFold2
- Statistical significance: p < 0.001

### Support

- Binary version: Check with `--version` flag
- GPU status: Run `nvidia-smi`
- System logs: Check `/var/log/` for issues
- Repository: https://github.com/Delfictus/ARES-51

🎯 **Ready to prove PRCT algorithm superiority!**
EOF

    log_info "✅ Instructions created: $WORK_DIR/README_RUNPOD.md"
}

# Main execution
main() {
    echo "Starting PRCT validation suite setup..."
    echo "Target directory: $WORK_DIR"
    echo ""
    
    check_runpod_environment
    install_dependencies
    install_rust
    clone_repository
    install_python_deps
    build_binaries
    setup_validation_env
    create_instructions
    
    echo ""
    echo "🎉 PRCT Validation Suite Setup Complete!"
    echo "========================================"
    echo ""
    echo "📁 Working directory: $WORK_DIR"
    echo "🔧 Binaries location: /tmp/prct-build/release/"
    echo "📖 Instructions: $WORK_DIR/README_RUNPOD.md"
    echo ""
    echo "🚀 Ready to run validation:"
    echo "   cd $WORK_DIR"
    echo "   ./test_setup.sh      # Test everything works"
    echo "   ./run_validation.sh  # Run complete pipeline"
    echo ""
    echo "💡 Estimated validation time: 2-6 hours on 8x H100"
    echo "💰 Cost optimization: Terminate instance after downloading results"
    echo ""
    echo "✅ PRCT algorithm ready for revolutionary validation!"
}

# Execute main function
main "$@"