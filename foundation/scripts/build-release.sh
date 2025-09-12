#!/usr/bin/env bash
# Build release binaries for ARES CSF

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/target/release"
DIST_DIR="$PROJECT_ROOT/dist"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
TARGET=""
FEATURES="default"
PROFILE="release"
STRIP=true
COMPRESS=true

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --target <TARGET>     Target triple (e.g., x86_64-unknown-linux-gnu)"
    echo "  --features <FEATURES> Comma-separated list of features"
    echo "  --profile <PROFILE>   Build profile (release, perf, minsize)"
    echo "  --no-strip           Don't strip debug symbols"
    echo "  --no-compress        Don't compress output"
    echo "  --help               Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --features)
            FEATURES="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --no-strip)
            STRIP=false
            shift
            ;;
        --no-compress)
            COMPRESS=false
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect target if not specified
if [[ -z "$TARGET" ]]; then
    TARGET=$(rustc -vV | sed -n 's|host: ||p')
    log_info "Building for native target: $TARGET"
fi

# Create dist directory
mkdir -p "$DIST_DIR"

# Get version from Cargo.toml
VERSION=$(grep '^version' "$PROJECT_ROOT/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
log_info "Building ARES CSF v$VERSION"

# Set build flags
export RUSTFLAGS="-C target-cpu=native -C link-arg=-s"
if [[ "$PROFILE" == "minsize" ]]; then
    export RUSTFLAGS="$RUSTFLAGS -C opt-level=z"
fi

# Clean previous builds
log_info "Cleaning previous builds..."
cargo clean --release

# Build the project
log_info "Building with features: $FEATURES"
BUILD_CMD="cargo build --release --features $FEATURES"
if [[ -n "$TARGET" ]]; then
    BUILD_CMD="$BUILD_CMD --target $TARGET"
fi

if $BUILD_CMD; then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Determine binary location
if [[ -n "$TARGET" ]]; then
    BINARY_PATH="$PROJECT_ROOT/target/$TARGET/release/chronofabric"
else
    BINARY_PATH="$PROJECT_ROOT/target/release/chronofabric"
fi

# Strip binary if requested
if [[ "$STRIP" == true ]] && [[ -f "$BINARY_PATH" ]]; then
    log_info "Stripping debug symbols..."
    strip "$BINARY_PATH"
fi

# Package binary
PACKAGE_NAME="ares-csf-v$VERSION-$TARGET"
PACKAGE_DIR="$DIST_DIR/$PACKAGE_NAME"

log_info "Creating package: $PACKAGE_NAME"
mkdir -p "$PACKAGE_DIR"

# Copy binary
cp "$BINARY_PATH" "$PACKAGE_DIR/"

# Copy configuration files
mkdir -p "$PACKAGE_DIR/config"
cp "$PROJECT_ROOT/config/default.toml" "$PACKAGE_DIR/config/"
cp "$PROJECT_ROOT/config/production.toml" "$PACKAGE_DIR/config/" 2>/dev/null || true

# Copy documentation
cp "$PROJECT_ROOT/README.md" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/LICENSE" "$PACKAGE_DIR/"

# Create wrapper script
cat > "$PACKAGE_DIR/chronofabric.sh" << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/chronofabric" "$@"
EOF
chmod +x "$PACKAGE_DIR/chronofabric.sh"

# Generate checksums
cd "$PACKAGE_DIR"
sha256sum chronofabric > checksums.txt
cd - > /dev/null

# Compress if requested
if [[ "$COMPRESS" == true ]]; then
    log_info "Compressing package..."
    cd "$DIST_DIR"
    
    if [[ "$TARGET" == *"windows"* ]]; then
        # Create ZIP for Windows
        zip -r "$PACKAGE_NAME.zip" "$PACKAGE_NAME"
        ARCHIVE="$PACKAGE_NAME.zip"
    else
        # Create tarball for Unix
        tar czf "$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"
        ARCHIVE="$PACKAGE_NAME.tar.gz"
    fi
    
    # Generate checksum for archive
    sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
    
    # Clean up uncompressed directory
    rm -rf "$PACKAGE_NAME"
    
    cd - > /dev/null
    
    log_success "Package created: $DIST_DIR/$ARCHIVE"
else
    log_success "Package created: $PACKAGE_DIR"
fi

# Build summary
echo
echo "Build Summary:"
echo "=============="
echo "Version:  $VERSION"
echo "Target:   $TARGET"
echo "Features: $FEATURES"
echo "Profile:  $PROFILE"
echo "Output:   $DIST_DIR"

# Optional: Run basic smoke test
if [[ -f "$BINARY_PATH" ]]; then
    log_info "Running smoke test..."
    if "$BINARY_PATH" --version > /dev/null 2>&1; then
        log_success "Binary smoke test passed"
    else
        log_error "Binary smoke test failed"
        exit 1
    fi
fi