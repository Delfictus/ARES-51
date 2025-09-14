#!/bin/bash
# Fix script for RunPod container

echo "🔧 PRCT RunPod Container Fix Script"
echo "===================================="

# 1. Fix CUDA environment
echo -e "\n1️⃣ Fixing CUDA environment..."
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ldconfig

# 2. Test CUDA again
echo -e "\n2️⃣ Testing CUDA after fix..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" || {
    echo "Trying alternative CUDA fix..."
    export CUDA_VISIBLE_DEVICES=0
    export NVIDIA_VISIBLE_DEVICES=all
    python3 -c "import torch; torch.cuda.init(); print(f'CUDA: {torch.cuda.is_available()}')"
}

# 3. Check what's in workspace
echo -e "\n3️⃣ Checking workspace contents..."
ls -la /workspace/
ls -la /workspace/prct-validation/

# 4. Clone repository if missing
echo -e "\n4️⃣ Cloning ARES-51 repository..."
cd /workspace
if [ ! -d "prct-validation/prct-engine" ]; then
    rm -rf prct-validation
    git clone https://github.com/Delfictus/ARES-51.git prct-validation
    echo "Repository cloned successfully"
else
    echo "Repository already exists"
fi

# 5. Navigate to prct-engine
echo -e "\n5️⃣ Setting up PRCT engine..."
cd /workspace/prct-validation
if [ ! -d "prct-engine" ]; then
    echo "⚠️ Warning: prct-engine directory not found in repository"
    echo "Repository structure:"
    find . -maxdepth 2 -type d | head -20
else
    cd prct-engine
    pwd
    ls -la
fi

# 6. Setup Rust environment
echo -e "\n6️⃣ Setting up Rust build environment..."
source /opt/rust/env
export CARGO_TARGET_DIR=/tmp/prct-build
rustc --version
cargo --version

echo -e "\n✅ Setup complete! You can now work with the PRCT engine."
