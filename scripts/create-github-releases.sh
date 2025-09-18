#!/bin/bash
# Create GitHub releases for ARES crates

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Creating GitHub Releases for ARES Crates${NC}"
echo "=============================================="

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) not found${NC}"
    echo "Please install GitHub CLI: https://cli.github.com/"
    echo "Or create releases manually in the GitHub web interface."
    exit 1
fi

# Configuration
CRATES_REPO="delfictus/ares-crates"
CRATES_DIR="./ares-csf-core/published-crates"

# Check if crates directory exists
if [[ ! -d "$CRATES_DIR" ]]; then
    echo -e "${RED}❌ Crates directory not found: $CRATES_DIR${NC}"
    exit 1
fi

# Function to create a release
create_release() {
    local crate_name=$1
    local version=$2
    local description=$3
    
    local tag="${crate_name}-${version}"
    local title="${crate_name} v${version}"
    local crate_file="${CRATES_DIR}/${crate_name}-${version}.crate"
    
    echo -e "\n${BLUE}📦 Creating release: $title${NC}"
    
    if [[ ! -f "$crate_file" ]]; then
        echo -e "${RED}❌ Crate file not found: $crate_file${NC}"
        return 1
    fi
    
    # Create the release
    gh release create "$tag" "$crate_file" \
        --repo "$CRATES_REPO" \
        --title "$title" \
        --notes "$description" \
        --target main
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ Created release: $title${NC}"
    else
        echo -e "${RED}❌ Failed to create release: $title${NC}"
        return 1
    fi
}

# Create releases for each crate
echo -e "\n${YELLOW}📋 Creating releases...${NC}"

create_release "ares-spike-encoding" "0.1.0" \
"Neural spike encoding algorithms for neuromorphic computing.

**Features:**
- 6 encoding methods (Rate, Temporal, Population, Phase, Latency, Burst)
- Financial market data optimization
- High-performance implementations
- Comprehensive pattern analysis

**Usage:**
\`\`\`toml
[dependencies]
ares-spike-encoding = { version = \"0.1.0\", registry = \"ares\" }
\`\`\`"

create_release "ares-csf-core" "0.1.0" \
"ChronoSynclastic Fabric computational infrastructure.

**Features:**
- High-performance tensor operations
- Distributed computing framework
- Variational optimization algorithms
- Quantum-temporal processing
- Memory optimization and GPU acceleration

**Usage:**
\`\`\`toml
[dependencies]
ares-csf-core = { version = \"0.1.0\", registry = \"ares\" }
\`\`\`"

create_release "ares-neuromorphic-core" "0.1.0" \
"Neuromorphic computing engine with reservoir computing.

**Features:**
- Reservoir computing implementation
- Pattern detection and signal processing
- Async neuromorphic engine
- Market data prediction capabilities
- Liquid state machine support

**Usage:**
\`\`\`toml
[dependencies]
ares-neuromorphic-core = { version = \"0.1.0\", registry = \"ares\" }
\`\`\`"

echo -e "\n${GREEN}🎉 All releases created successfully!${NC}"
echo -e "\n${BLUE}📋 Next steps:${NC}"
echo "1. Run: ./scripts/update-registry-index.sh"
echo "2. Commit and push registry index changes"
echo "3. Test your registry with: ./scripts/verify-delfictus-setup.sh"

echo -e "\n${GREEN}🚀 ARES crates are now published!${NC}"