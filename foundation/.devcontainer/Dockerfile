# Base image with Rust and development tools
ARG VARIANT="1.75-bullseye"
FROM mcr.microsoft.com/devcontainers/rust:1-${VARIANT}

# Install additional OS packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        # Build essentials
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        libssl-dev \
        # LLVM/MLIR dependencies
        llvm-14-dev \
        libclang-14-dev \
        clang-14 \
        lld-14 \
        # CUDA development (optional)
        nvidia-cuda-toolkit \
        nvidia-cuda-toolkit-gcc \
        # Performance tools
        linux-tools-generic \
        valgrind \
        heaptrack \
        # Network tools
        iproute2 \
        iputils-ping \
        netcat \
        tcpdump \
        # Database tools
        sqlite3 \
        redis-tools \
        # Python for tooling
        python3-pip \
        python3-dev \
        # Documentation tools
        graphviz \
        plantuml \
        # Debugging tools
        gdb \
        strace \
        ltrace \
    && apt-get autoremove -y && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain components
RUN rustup component add \
    rustfmt \
    clippy \
    rust-src \
    rust-analyzer \
    llvm-tools-preview

# Install additional Rust tools
RUN cargo install \
    cargo-audit \
    cargo-outdated \
    cargo-edit \
    cargo-watch \
    cargo-expand \
    cargo-tarpaulin \
    cargo-criterion \
    cargo-flamegraph \
    cargo-deny \
    cargo-license \
    cargo-machete \
    cargo-nextest \
    tokio-console \
    just

# Install MLIR tools
RUN pip3 install --no-cache-dir \
    mlir \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    notebook

# Install Node.js for web tooling
ARG NODE_VERSION="18"
RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"

# Set up CUDA environment (if available)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create directories for mounted volumes
RUN mkdir -p /workspace/.cargo /workspace/.rustup /workspace/target \
    && chown -R vscode:vscode /workspace

# Configure Git
RUN git config --global core.editor "code --wait" \
    && git config --global init.defaultBranch main

# Install global npm packages
RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g @commitlint/cli @commitlint/config-conventional" 2>&1

# Set up shell
RUN echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /home/vscode/.zshrc \
    && echo 'export RUST_SRC_PATH="$(rustc --print sysroot)/lib/rustlib/src/rust/library"' >> /home/vscode/.zshrc

# Pre-download commonly used crates to speed up builds
RUN cargo install cargo-quickinstall \
    && cargo quickinstall tokio serde

WORKDIR /workspace

# Switch to non-root user
USER vscode