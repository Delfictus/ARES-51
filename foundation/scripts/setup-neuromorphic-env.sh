#!/bin/bash
# ARES Neuromorphic Environment Setup Script
# Enterprise-grade Python environment configuration for Brian2/Lava integration
# Author: Ididia Serfaty

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv-neuromorphic"

echo "🧠 ARES Neuromorphic Environment Setup"
echo "======================================"

# Check for Python 3.8+
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating neuromorphic virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core scientific computing stack
echo "🔬 Installing scientific computing dependencies..."
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.5.0

# Install Brian2 for spiking neural networks
echo "🧠 Installing Brian2 for spiking neural network simulation..."
pip install brian2>=2.5.0

# Install Lava SDK for neuromorphic computing
echo "⚛️  Installing Lava SDK for neuromorphic computing..."
pip install lava-nc>=0.8.0 lava-optimization>=0.3.0

# Install additional neuromorphic libraries
echo "🔬 Installing additional neuromorphic libraries..."
pip install nengo>=3.2.0 snntorch>=0.7.0

# Install CUDA support if available
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 NVIDIA GPU detected - installing CUDA support..."
    pip install cupy-cuda11x  # Adjust version based on CUDA installation
    pip install brian2cuda
else
    echo "ℹ️  No NVIDIA GPU detected - skipping CUDA support"
fi

# Install PyO3 development dependencies
echo "🔗 Installing PyO3 development dependencies..."
pip install pyo3-stub-gen

# Create environment validation script
cat > "$PROJECT_ROOT/scripts/validate-neuromorphic-env.py" << 'EOF'
#!/usr/bin/env python3
"""
ARES Neuromorphic Environment Validation
Enterprise-grade validation for Brian2/Lava integration
"""

import sys
import traceback

def validate_environment():
    """Validate neuromorphic computing environment"""
    print("🧠 ARES Neuromorphic Environment Validation")
    print("=" * 45)
    
    success = True
    
    # Test Brian2
    try:
        import brian2
        print(f"✓ Brian2 {brian2.__version__} - OK")
        
        # Test basic Brian2 functionality
        from brian2 import *
        clear(True)
        N = 100
        neurons = NeuronGroup(N, 'dv/dt = -v/(10*ms) : volt', threshold='v>-50*mV', reset='v=-70*mV')
        print("✓ Brian2 neuron simulation - OK")
        
    except Exception as e:
        print(f"❌ Brian2 validation failed: {e}")
        success = False
    
    # Test Lava
    try:
        import lava
        print(f"✓ Lava {lava.__version__} - OK")
        
        # Test basic Lava functionality
        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense
        lif_proc = LIF(shape=(10,), du=4095, dv=4095, bias_mant=0, bias_exp=0)
        print("✓ Lava LIF process creation - OK")
        
    except Exception as e:
        print(f"❌ Lava validation failed: {e}")
        success = False
    
    # Test NumPy
    try:
        import numpy as np
        test_array = np.random.rand(1000, 1000)
        result = np.dot(test_array, test_array.T)
        print(f"✓ NumPy {np.__version__} matrix operations - OK")
    except Exception as e:
        print(f"❌ NumPy validation failed: {e}")
        success = False
    
    # Test CUDA if available
    try:
        import cupy as cp
        test_gpu = cp.random.rand(1000, 1000)
        result_gpu = cp.dot(test_gpu, test_gpu.T)
        print(f"✓ CuPy GPU acceleration - OK")
    except ImportError:
        print("ℹ️  CuPy not available - CPU-only mode")
    except Exception as e:
        print(f"⚠️  GPU acceleration issue: {e}")
    
    # Memory test
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024**3:  # 2GB
            print("⚠️  Low memory warning: Less than 2GB available")
        else:
            print(f"✓ Memory check - {memory.available // (1024**3)}GB available")
    except ImportError:
        print("ℹ️  psutil not available - skipping memory check")
    
    print("=" * 45)
    if success:
        print("🎉 Environment validation PASSED")
        print("✓ Ready for enterprise neuromorphic computing")
        return 0
    else:
        print("❌ Environment validation FAILED")
        print("⚠️  Fix issues before running ARES neuromorphic CLI")
        return 1

if __name__ == "__main__":
    sys.exit(validate_environment())
EOF

chmod +x "$PROJECT_ROOT/scripts/validate-neuromorphic-env.py"

# Run validation
echo ""
echo "🔍 Running environment validation..."
python3 "$PROJECT_ROOT/scripts/validate-neuromorphic-env.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Neuromorphic environment setup completed successfully!"
    echo ""
    echo "To activate the environment:"
    echo "  source $VENV_PATH/bin/activate"
    echo ""
    echo "To run ARES neuromorphic CLI:"
    echo "  cargo run -p ares-neuromorphic-cli -- enhanced"
    echo ""
else
    echo ""
    echo "⚠️  Environment setup completed with warnings"
    echo "Check the validation output above for issues"
fi