#!/bin/bash
# Simple script to publish ARES crates to your personal registry

set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REGISTRY="ares"

echo -e "${BLUE}🚀 Publishing ARES Crates${NC}"
echo "=========================="

# Check if registry is configured
if ! cargo config get registries.ares.index > /dev/null 2>&1; then
    echo -e "${RED}❌ ARES registry not configured!${NC}"
    echo "Please run: cargo config set registries.ares.index 'https://github.com/delfictus/ares-registry.git'"
    exit 1
fi

REGISTRY_URL=$(cargo config get registries.ares.index | tr -d '"')
echo -e "📍 Using registry: ${YELLOW}$REGISTRY_URL${NC}"

# Function to publish a single crate
publish_crate() {
    local crate_name=$1
    local crate_dir=$2
    
    if [[ ! -d "$crate_dir" ]]; then
        echo -e "${YELLOW}⚠️  Skipping $crate_name (directory not found)${NC}"
        return
    fi
    
    echo -e "\n${BLUE}📦 Publishing $crate_name...${NC}"
    cd "$crate_dir"
    
    # Get current version
    VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*= *"\([^"]*\)".*/\1/')
    echo -e "📌 Version: ${YELLOW}$VERSION${NC}"
    
    # Clean and test
    echo "🧹 Cleaning..."
    cargo clean
    
    echo "🧪 Running tests..."
    if ! cargo test --lib --quiet; then
        echo -e "${RED}❌ Tests failed for $crate_name${NC}"
        cd ..
        return 1
    fi
    
    echo "🔍 Checking crate..."
    if ! cargo check --all-features --quiet; then
        echo -e "${RED}❌ Check failed for $crate_name${NC}"
        cd ..
        return 1
    fi
    
    echo "📋 Packaging..."
    if ! cargo package --registry "$REGISTRY" --quiet; then
        echo -e "${RED}❌ Package failed for $crate_name${NC}"
        cd ..
        return 1
    fi
    
    echo "🚀 Publishing..."
    if cargo publish --registry "$REGISTRY" --allow-dirty; then
        echo -e "${GREEN}✅ $crate_name v$VERSION published successfully!${NC}"
        
        # Show where the .crate file is
        CRATE_FILE="target/package/${crate_name}-${VERSION}.crate"
        if [[ -f "$CRATE_FILE" ]]; then
            echo -e "📦 Crate file: ${YELLOW}$(pwd)/$CRATE_FILE${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to publish $crate_name${NC}"
    fi
    
    cd ..
}

# Publish in dependency order
echo -e "\n${YELLOW}📋 Publishing in dependency order...${NC}"

# 1. Spike encoding (no ARES dependencies)
publish_crate "ares-spike-encoding" "ares-spike-encoding"

# 2. CSF core (no ARES dependencies) 
publish_crate "ares-csf-core" "ares-csf-core"

# 3. Neuromorphic core (may depend on spike encoding)
publish_crate "ares-neuromorphic-core" "ares-neuromorphic-core"

echo -e "\n${GREEN}🎉 Publishing complete!${NC}"
echo -e "\n${BLUE}💡 Next steps:${NC}"
echo "1. Upload .crate files to GitHub releases in your ares-crates repo"
echo "2. Update the registry index manually or with automation"
echo "3. Test installing in another project"

echo -e "\n${YELLOW}📦 Crate files location:${NC}"
find . -name "*.crate" -path "*/target/package/*" | head -10