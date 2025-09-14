#!/bin/bash
# Direct RunPod Setup for PRCT Validation
# Run this script in your RunPod instance

set -euo pipefail

echo "🚀 PRCT Validation Suite - Direct RunPod Setup"
echo "=============================================="

# Configuration
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

# Create work directory and basic structure
setup_validation_env() {
    log_info "Setting up validation environment..."
    
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    
    # Create data directories
    mkdir -p {data/{casp16,results,benchmarks},logs,binaries}
    
    # Create test script
    cat > test_environment.sh << 'EOF'
#!/bin/bash
# Test RunPod Environment

echo "🧪 Testing PRCT validation environment..."

echo "  System Information:"
echo "    OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "    CPUs: $(nproc)"
echo "    Memory: $(free -h | grep Mem | awk '{print $2}')"

echo ""
echo "  GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPUs detected"

echo ""
echo "  Software Versions:"
echo "    Python: $(python3 --version)"
python3 -c "import numpy, pandas, matplotlib; print('    NumPy/Pandas/Matplotlib: OK')" || echo "    Python packages: Missing"

if command -v rustc &> /dev/null; then
    echo "    Rust: $(rustc --version)"
else
    echo "    Rust: Not installed"
fi

echo ""
echo "✅ Environment test completed!"
EOF

    chmod +x test_environment.sh
    
    # Create placeholder validation script
    cat > run_prct_validation.sh << 'EOF'
#!/bin/bash
# PRCT Validation Placeholder

echo "🧬 PRCT Algorithm Validation"
echo "============================"

echo "📍 Current Status: Environment setup complete"
echo "⚠️  Next Step: Upload/compile PRCT binaries"
echo ""
echo "Directory Structure:"
find /workspace/prct-validation -type d | head -10

echo ""
echo "🎯 Ready for PRCT algorithm deployment!"
EOF

    chmod +x run_prct_validation.sh
    
    log_info "✅ Validation environment configured"
}

# Main execution
main() {
    echo "Starting PRCT validation environment setup..."
    echo "Target directory: $WORK_DIR"
    echo ""
    
    check_runpod_environment
    install_dependencies
    install_rust
    install_python_deps
    setup_validation_env
    
    echo ""
    echo "🎉 PRCT Validation Environment Setup Complete!"
    echo "============================================="
    echo ""
    echo "📁 Working directory: $WORK_DIR"
    echo "🔧 Environment test: ./test_environment.sh"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Run ./test_environment.sh to verify setup"
    echo "   2. Upload PRCT binaries to binaries/ directory"
    echo "   3. Execute ./run_prct_validation.sh"
    echo ""
    echo "💡 RunPod instance ready for PRCT algorithm validation!"
}

# Execute main function
main "$@"