#!/bin/bash
# PRCT Build Testing Commands for RunPod Container

echo "🧬 PRCT Build Validation Tests"
echo "================================"

# 1. Check GPU availability
echo -e "\n📊 GPU Status:"
nvidia-smi || echo "⚠️  No GPU detected"

# 2. Check Rust installation
echo -e "\n🦀 Rust Environment:"
source /opt/rust/env
rustc --version
cargo --version

# 3. Check Python packages
echo -e "\n🐍 Python Packages:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# 4. Check PRCT repository
echo -e "\n📁 Repository Status:"
cd /workspace/prct-validation
ls -la
cd prct-engine
pwd

# 5. Try to build PRCT
echo -e "\n🔨 Attempting PRCT Build:"
export CARGO_TARGET_DIR=/tmp/prct-build
cargo check --lib 2>&1 | head -20

# 6. Check for built binaries
echo -e "\n📦 Checking for binaries:"
ls -la /tmp/prct-build/release/ 2>/dev/null || echo "No release binaries found yet"

# 7. Memory and system info
echo -e "\n💾 System Resources:"
free -h
df -h /workspace
nproc

echo -e "\n✅ Test complete!"
