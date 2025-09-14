# PRCT Engine Cloud Deployment - H100 Optimized Container
# Target: 8x NVIDIA H100 PCIe (80GB HBM3 each)
# CUDA 12.8.1, PyTorch 2.8.0, Tensor Core optimization

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL maintainer="CapoAI Team"
LABEL description="PRCT Algorithm Cloud Validation Container"
LABEL version="1.0.0"
LABEL gpu.architecture="H100"
LABEL gpu.memory="80GB_HBM3"

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VERSION=12.8.1
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /opt/prct-engine

# Update system and install essential packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libhdf5-dev \
    python3 \
    python3-pip \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for PRCT engine compilation
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install PyTorch 2.8.0 for AlphaFold2 comparison
RUN pip3 install torch==2.8.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Install scientific computing dependencies
RUN pip3 install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    biopython==1.81 \
    biotite==0.39.0 \
    h5py==3.9.0

# Install AlphaFold2 for comparison benchmarks
RUN pip3 install \
    alphafold==2.3.2 \
    dm-haiku==0.0.10 \
    dm-tree==0.1.8 \
    jax==0.4.13 \
    jaxlib==0.4.13+cuda12.cudnn89 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install LGA software for GDT-TS scoring
RUN wget -O /tmp/lga.tar.gz "https://proteinmodel.org/AS2TS/LGA/lga.tar.gz" \
    && tar -xzf /tmp/lga.tar.gz -C /opt \
    && cd /opt/lga && make \
    && ln -s /opt/lga/lga /usr/local/bin/lga \
    && rm /tmp/lga.tar.gz

# Install additional bioinformatics tools
RUN apt-get update && apt-get install -y \
    clustalw \
    muscle \
    mafft \
    && rm -rf /var/lib/apt/lists/*

# Create directories for data and results
RUN mkdir -p /opt/prct-engine/{data,results,benchmarks,logs} \
    && mkdir -p /data/{casp16,alphafold2,pdb,validation}

# Copy PRCT engine source code (will be mounted or copied during deployment)
COPY . /opt/prct-engine/

# Set Rust compilation target directory to avoid conflicts
ENV CARGO_TARGET_DIR=/tmp/prct-build

# Build PRCT engine in release mode with H100 optimizations
RUN cd /opt/prct-engine/prct-engine && \
    cargo build --release --features="cuda-h100,tensor-cores,hbm3-optimization"

# Create H100 performance monitoring script
RUN cat > /opt/prct-engine/monitor_h100.sh << 'EOF'
#!/bin/bash
# H100 Performance Monitoring Script

echo "=== NVIDIA H100 Performance Monitoring ==="
echo "Timestamp: $(date)"
echo ""

# GPU Information
echo "=== GPU Configuration ==="
nvidia-smi -q | grep -A 20 "GPU 0"
echo ""

# Memory Usage
echo "=== Memory Usage ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
echo ""

# Temperature and Power
echo "=== Thermal and Power ==="
nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
echo ""

# Utilization
echo "=== Utilization ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits
echo ""

# NVLink Status (if applicable)
echo "=== NVLink Status ==="
nvidia-smi nvlink -s
echo ""

# CUDA Processes
echo "=== CUDA Processes ==="
nvidia-smi pmon -c 1
echo ""
EOF

RUN chmod +x /opt/prct-engine/monitor_h100.sh

# Create validation execution script
RUN cat > /opt/prct-engine/run_validation.sh << 'EOF'
#!/bin/bash
# PRCT Engine Cloud Validation Script

set -euo pipefail

echo "🧬 PRCT Algorithm Cloud Validation - H100 Deployment"
echo "===================================================="
echo ""

# Check H100 availability
echo "=== Verifying H100 GPUs ==="
nvidia-smi | grep "H100"
if [ $? -ne 0 ]; then
    echo "ERROR: H100 GPUs not detected"
    exit 1
fi
echo "✅ H100 GPUs verified"
echo ""

# Start performance monitoring
echo "=== Starting Performance Monitoring ==="
./monitor_h100.sh > logs/h100_initial_status.log
echo "✅ Initial H100 status logged"
echo ""

# Download CASP16 data
echo "=== Downloading CASP16 Dataset ==="
cd /opt/prct-engine/prct-engine
cargo run --release --bin casp16_downloader -- \
    --output-dir /data/casp16 \
    --verify-checksums \
    --log-level info
echo "✅ CASP16 data downloaded"
echo ""

# Run PRCT validation
echo "=== Executing PRCT Algorithm Validation ==="
RUST_LOG=info cargo run --release --bin prct_validator -- \
    --casp16-data /data/casp16 \
    --results-dir /opt/prct-engine/results \
    --gpu-count 8 \
    --precision mixed \
    --batch-size auto \
    --log-performance \
    2>&1 | tee logs/prct_validation.log
echo "✅ PRCT validation completed"
echo ""

# Compare with AlphaFold2
echo "=== Running AlphaFold2 Comparison ==="
python3 /opt/prct-engine/validation/compare_alphafold2.py \
    --prct-results /opt/prct-engine/results \
    --casp16-targets /data/casp16 \
    --output-dir /opt/prct-engine/results/comparison \
    --statistical-analysis \
    2>&1 | tee logs/alphafold2_comparison.log
echo "✅ AlphaFold2 comparison completed"
echo ""

# Generate final report
echo "=== Generating Validation Report ==="
cargo run --release --bin report_generator -- \
    --results-dir /opt/prct-engine/results \
    --output-format publication \
    --include-performance-metrics \
    --statistical-significance 0.001
echo "✅ Validation report generated"
echo ""

# Final performance snapshot
./monitor_h100.sh > logs/h100_final_status.log
echo "✅ Final H100 status logged"

echo ""
echo "🎯 PRCT Algorithm Cloud Validation Completed Successfully"
echo "Results available in: /opt/prct-engine/results/"
EOF

RUN chmod +x /opt/prct-engine/run_validation.sh

# Create cleanup script
RUN cat > /opt/prct-engine/cleanup.sh << 'EOF'
#!/bin/bash
# Cloud Resource Cleanup Script

echo "🧹 Cleaning up cloud resources..."

# Clear sensitive data
rm -rf /data/casp16/experimental_structures/ 2>/dev/null || true
rm -rf /tmp/prct-build/ 2>/dev/null || true

# Compress results for download
cd /opt/prct-engine
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz results/ logs/

echo "✅ Cleanup completed"
echo "📦 Results archived: results_$(date +%Y%m%d_%H%M%S).tar.gz"
EOF

RUN chmod +x /opt/prct-engine/cleanup.sh

# Set default command
CMD ["/bin/bash", "/opt/prct-engine/run_validation.sh"]

# Expose ports for monitoring (optional)
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD nvidia-smi || exit 1

# Set resource limits and optimization flags
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV CUDA_CACHE_PATH=/tmp/cuda-cache
ENV OMP_NUM_THREADS=32
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Final optimization for H100
RUN echo 'export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' >> /root/.bashrc
RUN echo 'export NCCL_DEBUG=INFO' >> /root/.bashrc
RUN echo 'export NCCL_TREE_THRESHOLD=0' >> /root/.bashrc