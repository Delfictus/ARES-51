#!/bin/bash
# Verify ARES registry setup for delfictus

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Verifying ARES Registry Setup for delfictus${NC}"
echo "================================================"

# Expected configuration
EXPECTED_REGISTRY="https://github.com/delfictus/ares-registry.git"

echo -e "\n${YELLOW}1. Checking Cargo configuration...${NC}"

# Check global config
if [[ -f ~/.cargo/config.toml ]] && grep -q "delfictus/ares-registry" ~/.cargo/config.toml; then
    echo -e "${GREEN}✅ Global Cargo config configured${NC}"
else
    echo -e "${YELLOW}⚠️  Global Cargo config not found or not configured${NC}"
fi

# Check project config  
if [[ -f .cargo/config.toml ]] && grep -q "delfictus/ares-registry" .cargo/config.toml; then
    echo -e "${GREEN}✅ Project Cargo config configured${NC}"
else
    echo -e "${YELLOW}⚠️  Project Cargo config not found or not configured${NC}"
fi

# Test cargo config command
echo -e "\n${YELLOW}2. Testing registry detection...${NC}"
if cargo config get registries.ares.index > /dev/null 2>&1; then
    CURRENT_REGISTRY=$(cargo config get registries.ares.index | tr -d '"')
    if [[ "$CURRENT_REGISTRY" == "$EXPECTED_REGISTRY" ]]; then
        echo -e "${GREEN}✅ Registry correctly configured: $CURRENT_REGISTRY${NC}"
    else
        echo -e "${RED}❌ Wrong registry URL: $CURRENT_REGISTRY${NC}"
        echo -e "Expected: $EXPECTED_REGISTRY"
    fi
else
    echo -e "${RED}❌ Registry not detected by Cargo${NC}"
fi

echo -e "\n${YELLOW}3. Checking repository accessibility...${NC}"
if git ls-remote "$EXPECTED_REGISTRY" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Registry repository is accessible${NC}"
else
    echo -e "${RED}❌ Cannot access registry repository${NC}"
    echo "Make sure you have:"
    echo "1. Created https://github.com/delfictus/ares-registry"
    echo "2. Set up SSH keys or personal access token"
    echo "3. Made the repository accessible to your account"
fi

echo -e "\n${YELLOW}4. Checking workspace structure...${NC}"
if [[ -f Cargo.toml ]] && grep -q "\[workspace\]" Cargo.toml; then
    echo -e "${GREEN}✅ Workspace configured${NC}"
    
    # Check for our crates
    for crate in "ares-spike-encoding" "ares-csf-core" "ares-neuromorphic-core"; do
        if [[ -d "$crate" ]]; then
            echo -e "${GREEN}  ✅ $crate found${NC}"
        else
            echo -e "${RED}  ❌ $crate missing${NC}"
        fi
    done
else
    echo -e "${RED}❌ Workspace not configured properly${NC}"
fi

echo -e "\n${YELLOW}5. Testing crate packaging...${NC}"
if [[ -d "ares-spike-encoding" ]]; then
    cd ares-spike-encoding
    if cargo package --registry ares --dry-run > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Packaging test successful${NC}"
    else
        echo -e "${RED}❌ Packaging test failed${NC}"
        echo "Try running: cargo clean && cargo check"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  Cannot test packaging - ares-spike-encoding not found${NC}"
fi

echo -e "\n${BLUE}📋 Summary${NC}"
echo "=========="
echo -e "Registry URL: ${YELLOW}$EXPECTED_REGISTRY${NC}"
echo ""
echo "If all checks passed, you can proceed with:"
echo "1. ./scripts/init-delfictus-registry.sh  # Initialize registry structure"
echo "2. Create GitHub repositories manually"
echo "3. ./scripts/publish-ares-crates.sh     # Publish crates"
echo ""
echo -e "${GREEN}🚀 Setup verification complete!${NC}"