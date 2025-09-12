#!/usr/bin/env bash
# Development environment setup script for ARES CSF

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   log_error "This script should not be run as root!"
   exit 1
fi

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    DISTRO=$(lsb_release -si 2>/dev/null || echo "unknown")
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

log_info "Detected OS: $OS"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        if [[ "$DISTRO" == "Ubuntu" || "$DISTRO" == "Debian" ]]; then
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                pkg-config \
                libssl-dev \
                libclang-dev \
                llvm-dev \
                protobuf-compiler \
                python3-pip \
                curl \
                git \
                jq \
                htop \
                valgrind \
                perf-tools-unstable
        elif [[ "$DISTRO" == "Fedora" || "$DISTRO" == "CentOS" || "$DISTRO" == "RedHat" ]]; then
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                cmake \
                openssl-devel \
                clang-devel \
                llvm-devel \
                protobuf-compiler \
                python3-pip \
                curl \
                git \
                jq \
                htop \
                valgrind \
                perf
        else
            log_warning "Unknown Linux distribution. Please install dependencies manually."
        fi
    elif [[ "$OS" == "macos" ]]; then
        if ! command_exists brew; then
            log_error "Homebrew is required. Please install from https://brew.sh"
            exit 1
        fi
        
        brew update
        brew install \
            cmake \
            protobuf \
            llvm \
            python3 \
            jq \
            htop
    fi
}

# Function to install Rust
install_rust() {
    if command_exists rustc; then
        log_info "Rust is already installed ($(rustc --version))"
        
        # Update Rust
        log_info "Updating Rust..."
        rustup update
    else
        log_info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    
    # Install required Rust components
    log_info "Installing Rust components..."
    rustup component add rustfmt clippy rust-src rust-analyzer
    
    # Install nightly for specific features
    rustup toolchain install nightly
    rustup component add rustfmt clippy --toolchain nightly
}

# Function to install Rust tools
install_rust_tools() {
    log_info "Installing Rust development tools..."
    
    cargo install --locked cargo-audit || true
    cargo install --locked cargo-outdated || true
    cargo install --locked cargo-edit || true
    cargo install --locked cargo-watch || true
    cargo install --locked cargo-tarpaulin || true
    cargo install --locked cargo-criterion || true
    cargo install --locked cargo-deny || true
    cargo install --locked cargo-license || true
    cargo install --locked cargo-machete || true
    cargo install --locked cargo-nextest || true
    cargo install --locked just || true
    
    # Optional performance tools
    if [[ "$OS" == "linux" ]]; then
        cargo install --locked cargo-flamegraph || true
        cargo install --locked tokio-console || true
    fi
}

# Function to setup Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    if ! command_exists python3; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Create virtual environment
    python3 -m venv .venv
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    # Install MLIR Python bindings if available
    pip install mlir || true
}

# Function to check CUDA installation
check_cuda() {
    if command_exists nvcc; then
        log_success "CUDA detected: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
        echo "export CUDA_HOME=/usr/local/cuda" >> "$HOME/.bashrc"
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> "$HOME/.bashrc"
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> "$HOME/.bashrc"
    else
        log_warning "CUDA not detected. GPU features will be disabled."
    fi
}

# Function to setup git hooks
setup_git_hooks() {
    log_info "Setting up git hooks..."
    
    # Create hooks directory
    mkdir -p .git/hooks
    
    # Pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "Running pre-commit checks..."

# Format check
cargo fmt --all -- --check
if [ $? -ne 0 ]; then
    echo "Code is not formatted. Run 'cargo fmt --all'"
    exit 1
fi

# Clippy check
cargo clippy --all-targets --all-features -- -D warnings
if [ $? -ne 0 ]; then
    echo "Clippy warnings found"
    exit 1
fi

# Test check
cargo test --lib --bins
if [ $? -ne 0 ]; then
    echo "Tests failed"
    exit 1
fi

echo "Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    
    # Commit message hook
    cat > .git/hooks/commit-msg << 'EOF'
#!/usr/bin/env bash
# Conventional commit message validation

commit_regex='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Format: <type>(<scope>): <subject>"
    echo "Example: feat(drpp): add transfer entropy calculation"
    exit 1
fi
EOF
    
    chmod +x .git/hooks/commit-msg
}

# Function to create local configuration
create_local_config() {
    log_info "Creating local configuration..."
    
    if [[ ! -f "config/local.toml" ]]; then
        mkdir -p config
        cat > config/local.toml << 'EOF'
# Local development configuration
# This file is gitignored and can be customized for your environment

[node]
node_id = "dev-node-01"
deployment_mode = "development"

[scheduler]
scheduler_cores = [0, 1]
max_tasks = 1000

[telemetry]
log_level = "debug"
enable_tracing = true
EOF
        log_success "Created config/local.toml"
    fi
}

# Function to download test data
download_test_data() {
    log_info "Downloading test data..."
    
    mkdir -p data/test
    
    # Download sample datasets if they exist
    # This is a placeholder - replace with actual test data URLs
    # wget -O data/test/sample_sensors.csv https://example.com/sample_sensors.csv
    
    log_info "Test data setup complete"
}

# Function to verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Rust
    if ! command_exists rustc; then
        log_error "Rust is not installed correctly"
        return 1
    fi
    
    # Check cargo
    if ! command_exists cargo; then
        log_error "Cargo is not installed correctly"
        return 1
    fi
    
    # Try to build
    log_info "Running test build..."
    if cargo build --all; then
        log_success "Build successful!"
    else
        log_error "Build failed"
        return 1
    fi
    
    # Run basic tests
    log_info "Running basic tests..."
    if cargo test --lib -- --test-threads=1; then
        log_success "Basic tests passed!"
    else
        log_error "Tests failed"
        return 1
    fi
    
    return 0
}

# Main setup flow
main() {
    log_info "Starting ARES CSF development environment setup..."
    
    # Check prerequisites
    if ! command_exists git; then
        log_error "Git is required but not installed"
        exit 1
    fi
    
    # Install system dependencies
    install_system_deps
    
    # Install Rust
    install_rust
    
    # Install Rust tools
    install_rust_tools
    
    # Setup Python environment
    setup_python
    
    # Check for CUDA
    check_cuda
    
    # Setup git hooks
    setup_git_hooks
    
    # Create local configuration
    create_local_config
    
    # Download test data
    download_test_data
    
    # Verify installation
    if verify_installation; then
        log_success "Development environment setup complete!"
        echo
        echo "Next steps:"
        echo "1. Source your shell configuration: source ~/.bashrc"
        echo "2. Activate Python venv: source .venv/bin/activate"
        echo "3. Run the development server: cargo run -- --config config/local.toml"
        echo "4. Open VS Code: code ."
    else
        log_error "Setup completed with errors. Please check the output above."
        exit 1
    fi
}

# Run main function
main "$@"