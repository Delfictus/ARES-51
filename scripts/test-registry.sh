#!/bin/bash
# Test script to verify ARES registry setup

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 Testing ARES Registry Setup${NC}"
echo "==============================="

# Test 1: Check if registry is configured
echo -e "\n${YELLOW}1. Checking registry configuration...${NC}"
if cargo config get registries.ares.index > /dev/null 2>&1; then
    REGISTRY_URL=$(cargo config get registries.ares.index | tr -d '"')
    echo -e "${GREEN}✅ ARES registry configured: $REGISTRY_URL${NC}"
else
    echo -e "${RED}❌ ARES registry not configured${NC}"
    echo "Add this to ~/.cargo/config.toml:"
    echo "[registries]"
    echo "ares = { index = \"https://github.com/YOUR_USERNAME/ares-registry.git\" }"
    exit 1
fi

# Test 2: Check if we can reach the registry
echo -e "\n${YELLOW}2. Testing registry connectivity...${NC}"
if git ls-remote "$REGISTRY_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Registry is accessible${NC}"
else
    echo -e "${RED}❌ Cannot access registry${NC}"
    echo "Make sure the repository exists and you have access"
    exit 1
fi

# Test 3: Check workspace structure
echo -e "\n${YELLOW}3. Checking workspace structure...${NC}"
if [[ -f "Cargo.toml" ]] && grep -q "\[workspace\]" Cargo.toml; then
    echo -e "${GREEN}✅ Workspace configured${NC}"
    
    # List workspace members
    echo "Workspace members:"
    grep -A 10 "members.*=" Cargo.toml | sed 's/.*"\(.*\)".*/  - \1/' | grep "  -"
else
    echo -e "${RED}❌ Workspace not configured${NC}"
fi

# Test 4: Check individual crates
echo -e "\n${YELLOW}4. Checking crate readiness...${NC}"

check_crate() {
    local crate_name=$1
    local crate_dir=$2
    
    if [[ ! -d "$crate_dir" ]]; then
        echo -e "${RED}❌ $crate_name: Directory not found${NC}"
        return 1
    fi
    
    cd "$crate_dir"
    
    if ! cargo check --quiet > /dev/null 2>&1; then
        echo -e "${RED}❌ $crate_name: Compilation errors${NC}"
        cd ..
        return 1
    fi
    
    if ! cargo test --lib --quiet > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  $crate_name: Test failures (may be expected)${NC}"
    else
        echo -e "${GREEN}✅ $crate_name: Ready to publish${NC}"
    fi
    
    cd ..
}

check_crate "ares-spike-encoding" "ares-spike-encoding"
check_crate "ares-csf-core" "ares-csf-core"
check_crate "ares-neuromorphic-core" "ares-neuromorphic-core"

# Test 5: Check if we can package (dry run)
echo -e "\n${YELLOW}5. Testing packaging (dry run)...${NC}"
if [[ -d "ares-spike-encoding" ]]; then
    cd ares-spike-encoding
    if cargo package --registry ares --dry-run > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Packaging test successful${NC}"
    else
        echo -e "${RED}❌ Packaging test failed${NC}"
    fi
    cd ..
fi

echo -e "\n${BLUE}📋 Summary${NC}"
echo "============"
echo "If all tests passed, you're ready to:"
echo "1. Run: ./scripts/publish-ares-crates.sh"
echo "2. Upload .crate files to GitHub releases" 
echo "3. Update registry index"
echo ""
echo -e "${GREEN}🚀 Registry setup verification complete!${NC}"