#!/bin/bash
# Update ARES registry index manually after uploading crates to GitHub releases

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔄 Updating ARES Registry Index${NC}"
echo "=================================="

# Configuration
REGISTRY_DIR="../ares-registry"
CRATES_BASE_URL="https://github.com/delfictus/ares-crates/releases/download"

# Check if registry directory exists
if [[ ! -d "$REGISTRY_DIR" ]]; then
    echo -e "${YELLOW}⚠️  Registry directory not found at $REGISTRY_DIR${NC}"
    echo "Please ensure you've cloned the ares-registry repository to the parent directory."
    exit 1
fi

cd "$REGISTRY_DIR"

# Function to add crate to index
add_crate_to_index() {
    local crate_name=$1
    local version=$2
    local checksum=$3
    
    echo -e "${BLUE}📝 Adding $crate_name v$version to index${NC}"
    
    # Create index directory structure (first 2 characters of crate name)
    local index_dir="index/${crate_name:0:2}"
    mkdir -p "$index_dir"
    
    # Create index entry
    local index_file="$index_dir/$crate_name"
    local download_url="$CRATES_BASE_URL/$crate_name-$version/$crate_name-$version.crate"
    
    # Add entry to index file
    cat >> "$index_file" << EOF
{"name":"$crate_name","vers":"$version","deps":[],"cksum":"$checksum","features":{},"yanked":false,"links":null}
EOF
    
    echo -e "${GREEN}✅ Added $crate_name v$version${NC}"
}

# Calculate checksums and add crates
echo -e "\n${YELLOW}📊 Processing crate checksums...${NC}"

# You'll need to update these checksums after uploading to GitHub
# Run: sha256sum path/to/crate.crate to get the actual checksums

echo -e "${YELLOW}⚠️  MANUAL STEP REQUIRED:${NC}"
echo "1. Upload the .crate files to GitHub releases"
echo "2. Calculate SHA256 checksums of the uploaded files"
echo "3. Update this script with the actual checksums"
echo "4. Run this script again to update the index"

echo ""
echo "Example checksum calculation:"
echo "sha256sum ../ARES-51/published-crates/ares-spike-encoding-0.1.0.crate"

# Add crates to index with calculated checksums
add_crate_to_index "ares-spike-encoding" "0.1.0" "50513c44d466f0dc1c4c675380710815a2f1a247c6fe86ee374f338cc1fded6a"
add_crate_to_index "ares-csf-core" "0.1.0" "eb9fc817cf9283ba34371bdce250478c79034f32789d8dc6145477506faea69b"  
add_crate_to_index "ares-neuromorphic-core" "0.1.0" "89adc625dbcf4e0eb612942d0da31eb5660256fedd45c24209f8693c484155e3"

echo -e "\n${BLUE}📋 Next steps:${NC}"
echo "1. Upload .crate files to GitHub releases"
echo "2. Get checksums and update this script"
echo "3. Run this script to update the index"
echo "4. Commit and push index changes"

echo -e "\n${GREEN}🚀 Registry index update script ready!${NC}"