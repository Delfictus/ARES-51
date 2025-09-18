#!/bin/bash
# GitHub-based Personal Registry Setup Script for ARES Libraries
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 ARES Personal Registry Setup${NC}"
echo "=================================="

# Get user input
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter registry repository name (default: ares-registry): " REGISTRY_REPO
REGISTRY_REPO=${REGISTRY_REPO:-ares-registry}

read -p "Enter crates storage repository name (default: ares-crates): " CRATES_REPO
CRATES_REPO=${CRATES_REPO:-ares-crates}

REGISTRY_URL="https://github.com/${GITHUB_USERNAME}/${REGISTRY_REPO}.git"
CRATES_URL="https://github.com/${GITHUB_USERNAME}/${CRATES_REPO}"

echo -e "\n${YELLOW}Configuration:${NC}"
echo "Registry URL: $REGISTRY_URL"
echo "Crates URL: $CRATES_URL"
echo ""

read -p "Continue with setup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# Step 1: Create temporary directory for registry setup
TEMP_DIR=$(mktemp -d)
echo -e "\n${BLUE}📁 Setting up registry structure...${NC}"

cd "$TEMP_DIR"

# Initialize registry repository
git init "$REGISTRY_REPO"
cd "$REGISTRY_REPO"

# Create registry index structure
mkdir -p index

# Create config.json for the registry
cat > config.json << EOF
{
  "dl": "${CRATES_URL}/releases/download/{crate}-{version}/{crate}-{version}.crate",
  "api": "${CRATES_URL}"
}
EOF

# Create README for registry
cat > README.md << EOF
# ARES Personal Cargo Registry

This is a private Cargo registry for ARES library crates.

## Available Crates

- \`ares-spike-encoding\` - Neural spike encoding algorithms
- \`ares-csf-core\` - ChronoSynclastic Fabric core infrastructure  
- \`ares-neuromorphic-core\` - Neuromorphic computation engine

## Usage

Add to your \`~/.cargo/config.toml\`:

\`\`\`toml
[registries]
ares = { index = "${REGISTRY_URL}" }
\`\`\`

Then in your \`Cargo.toml\`:

\`\`\`toml
[dependencies]
ares-spike-encoding = { version = "0.1.0", registry = "ares" }
ares-csf-core = { version = "0.1.0", registry = "ares" }
ares-neuromorphic-core = { version = "0.1.0", registry = "ares" }
\`\`\`
EOF

# Create .gitignore
cat > .gitignore << EOF
.DS_Store
Thumbs.db
*.swp
*.swo
*~
EOF

# Initial commit
git add .
git commit -m "🎉 Initialize ARES personal registry

- Registry index structure
- Configuration for GitHub releases
- Documentation for usage"

echo -e "${GREEN}✅ Registry structure created${NC}"

# Step 2: Create Cargo configuration
echo -e "\n${BLUE}⚙️  Creating Cargo configuration...${NC}"

CARGO_CONFIG_DIR="$HOME/.cargo"
CARGO_CONFIG_FILE="$CARGO_CONFIG_DIR/config.toml"

# Create .cargo directory if it doesn't exist
mkdir -p "$CARGO_CONFIG_DIR"

# Backup existing config if it exists
if [[ -f "$CARGO_CONFIG_FILE" ]]; then
    cp "$CARGO_CONFIG_FILE" "$CARGO_CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}📋 Backed up existing Cargo config${NC}"
fi

# Add registry configuration
cat >> "$CARGO_CONFIG_FILE" << EOF

# ARES Personal Registry Configuration
[registries]
ares = { index = "${REGISTRY_URL}" }

[source.ares]
registry = "${REGISTRY_URL}"
EOF

echo -e "${GREEN}✅ Cargo configuration updated${NC}"

# Step 3: Create publishing script
cd /mnt/m/Projects/ARES-51
cat > scripts/publish-to-registry.sh << 'EOF'
#!/bin/bash
# Publish ARES crates to personal GitHub registry
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REGISTRY="ares"

echo -e "${BLUE}📦 Publishing ARES crates to personal registry...${NC}"

# Check if registry is configured
if ! cargo config get registries.ares > /dev/null 2>&1; then
    echo -e "${RED}❌ Registry 'ares' not configured. Run setup-github-registry.sh first.${NC}"
    exit 1
fi

# Function to publish a crate
publish_crate() {
    local crate_dir=$1
    local crate_name=$2
    
    echo -e "\n${BLUE}📦 Publishing $crate_name...${NC}"
    cd "$crate_dir"
    
    # Ensure clean state
    cargo clean
    
    # Run tests
    echo "🧪 Running tests..."
    cargo test --lib
    
    # Check the crate
    echo "🔍 Checking crate..."
    cargo check --all-features
    
    # Package the crate
    echo "📋 Packaging crate..."
    cargo package --registry "$REGISTRY"
    
    # Publish to registry
    echo "🚀 Publishing to registry..."
    cargo publish --registry "$REGISTRY" --allow-dirty
    
    echo -e "${GREEN}✅ $crate_name published successfully${NC}"
    cd - > /dev/null
}

# Function to create GitHub release with crate file
create_github_release() {
    local crate_name=$1
    local version=$2
    local crate_file=$3
    
    echo -e "\n${BLUE}🏷️  Creating GitHub release for $crate_name v$version...${NC}"
    
    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        echo -e "${YELLOW}⚠️  GitHub CLI not found. Please manually upload $crate_file to GitHub releases.${NC}"
        return
    fi
    
    # Create release and upload crate file
    gh release create "${crate_name}-${version}" \
        --title "$crate_name v$version" \
        --notes "Release of $crate_name version $version" \
        "$crate_file" \
        --repo "$GITHUB_USERNAME/$CRATES_REPO" || true
}

# Get version from Cargo.toml
get_version() {
    local crate_dir=$1
    grep "^version" "$crate_dir/Cargo.toml" | sed 's/version = "\(.*\)"/\1/' | tr -d '"'
}

# Publish in dependency order
echo -e "\n${YELLOW}📋 Publishing crates in dependency order...${NC}"

# 1. Spike encoding (no dependencies)
if [[ -d "ares-spike-encoding" ]]; then
    publish_crate "ares-spike-encoding" "ares-spike-encoding"
    VERSION=$(get_version "ares-spike-encoding")
    create_github_release "ares-spike-encoding" "$VERSION" "ares-spike-encoding/target/package/ares-spike-encoding-${VERSION}.crate"
fi

# 2. CSF core (no ARES dependencies)
if [[ -d "ares-csf-core" ]]; then
    publish_crate "ares-csf-core" "ares-csf-core"
    VERSION=$(get_version "ares-csf-core")
    create_github_release "ares-csf-core" "$VERSION" "ares-csf-core/target/package/ares-csf-core-${VERSION}.crate"
fi

# 3. Neuromorphic core (depends on spike encoding)
if [[ -d "ares-neuromorphic-core" ]]; then
    publish_crate "ares-neuromorphic-core" "ares-neuromorphic-core"
    VERSION=$(get_version "ares-neuromorphic-core")
    create_github_release "ares-neuromorphic-core" "$VERSION" "ares-neuromorphic-core/target/package/ares-neuromorphic-core-${VERSION}.crate"
fi

echo -e "\n${GREEN}🎉 All crates published successfully!${NC}"
echo -e "${BLUE}💡 Next steps:${NC}"
echo "1. Push your registry repository to GitHub"
echo "2. Create the crates storage repository on GitHub"
echo "3. Set up GitHub releases for automatic crate distribution"
EOF

chmod +x scripts/publish-to-registry.sh

echo -e "${GREEN}✅ Publishing script created${NC}"

# Step 4: Create GitHub Actions workflow
echo -e "\n${BLUE}🔄 Creating GitHub Actions workflow...${NC}"

mkdir -p .github/workflows

cat > .github/workflows/publish-registry.yml << EOF
name: Publish to ARES Registry

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      crate:
        description: 'Crate to publish (all, spike-encoding, csf-core, neuromorphic-core)'
        required: true
        default: 'all'

env:
  CARGO_TERM_COLOR: always

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Install Rust toolchain
      uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt, clippy
        
    - name: Cache cargo dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/bin/
          ~/.cargo/registry/index/
          ~/.cargo/registry/cache/
          ~/.cargo/git/db/
          target/
        key: \${{ runner.os }}-cargo-\${{ hashFiles('**/Cargo.lock') }}
        
    - name: Configure ARES registry
      run: |
        mkdir -p ~/.cargo
        cat > ~/.cargo/config.toml << 'CONFIG_EOF'
        [registries]
        ares = { index = "${REGISTRY_URL}" }
        
        [source.ares]
        registry = "${REGISTRY_URL}"
        CONFIG_EOF
        
    - name: Publish crates
      env:
        CARGO_REGISTRIES_ARES_TOKEN: \${{ secrets.CARGO_REGISTRY_TOKEN }}
        GITHUB_USERNAME: ${GITHUB_USERNAME}
        CRATES_REPO: ${CRATES_REPO}
      run: |
        chmod +x scripts/publish-to-registry.sh
        ./scripts/publish-to-registry.sh
        
    - name: Create GitHub releases
      env:
        GITHUB_TOKEN: \${{ secrets.GITHUB_TOKEN }}
      run: |
        # Install GitHub CLI
        gh --version || {
          curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
          echo "deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
          sudo apt update
          sudo apt install gh
        }
        
        # The publishing script will handle GitHub releases
        echo "GitHub releases will be created by the publishing script"
EOF

echo -e "${GREEN}✅ GitHub Actions workflow created${NC}"

# Step 5: Create registry update script
cat > scripts/update-registry-index.sh << 'EOF'
#!/bin/bash
# Update registry index with new crate versions
set -e

REGISTRY_DIR="/tmp/ares-registry-update"
REGISTRY_URL="${REGISTRY_URL}"

echo "🔄 Updating ARES registry index..."

# Clone registry
rm -rf "$REGISTRY_DIR"
git clone "$REGISTRY_URL" "$REGISTRY_DIR"
cd "$REGISTRY_DIR"

# Update index for each crate
update_crate_index() {
    local crate_name=$1
    local version=$2
    local checksum=$3
    
    # Create directory structure based on crate name
    local first_char=${crate_name:0:1}
    local second_char=${crate_name:1:1}
    
    if [[ ${#crate_name} -eq 1 ]]; then
        local index_dir="index/1"
    elif [[ ${#crate_name} -eq 2 ]]; then
        local index_dir="index/2"
    elif [[ ${#crate_name} -eq 3 ]]; then
        local index_dir="index/3/${first_char}"
    else
        local index_dir="index/${first_char}${second_char}/${crate_name}"
    fi
    
    mkdir -p "$index_dir"
    
    # Create crate entry
    cat >> "${index_dir}/${crate_name}" << ENTRY_EOF
{"name":"${crate_name}","vers":"${version}","deps":[],"features":{},"cksum":"${checksum}","yanked":false,"links":null,"v":2}
ENTRY_EOF
}

echo "Registry index updated. Commit and push changes manually."
EOF

chmod +x scripts/update-registry-index.sh

echo -e "${GREEN}✅ Registry update script created${NC}"

# Step 6: Create setup completion summary
echo -e "\n${GREEN}🎉 GitHub Registry Setup Complete!${NC}"
echo "=================================="
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo ""
echo "1. 📚 Create GitHub repositories:"
echo "   - Create: https://github.com/${GITHUB_USERNAME}/${REGISTRY_REPO}"
echo "   - Create: https://github.com/${GITHUB_USERNAME}/${CRATES_REPO}"
echo ""
echo "2. 🔑 Set up repository secrets (for ${CRATES_REPO}):"
echo "   - CARGO_REGISTRY_TOKEN: Your registry authentication token"
echo "   - GITHUB_TOKEN: Automatically provided by GitHub Actions"
echo ""
echo "3. 📤 Push registry to GitHub:"
echo "   cd $TEMP_DIR/$REGISTRY_REPO"
echo "   git remote add origin $REGISTRY_URL"
echo "   git push -u origin main"
echo ""
echo "4. 🚀 Publish your first crates:"
echo "   cd /mnt/m/Projects/ARES-51"
echo "   ./scripts/publish-to-registry.sh"
echo ""
echo "5. 🔧 Your Cargo is now configured with the ARES registry"
echo ""

# Step 7: Create helper script to use registry in other projects
cat > scripts/use-ares-registry.sh << 'EOF'
#!/bin/bash
# Helper script to configure ARES registry in other projects

PROJECT_DIR=${1:-.}
cd "$PROJECT_DIR"

echo "🔧 Configuring project to use ARES registry..."

# Create or update .cargo/config.toml in project
mkdir -p .cargo

cat > .cargo/config.toml << CONFIG_EOF
[registries]
ares = { index = "REGISTRY_URL_PLACEHOLDER" }

[source.ares]
registry = "REGISTRY_URL_PLACEHOLDER"
CONFIG_EOF

echo "✅ Project configured for ARES registry"
echo "Add this to your Cargo.toml dependencies:"
echo ""
echo "[dependencies]"
echo "ares-spike-encoding = { version = \"0.1.0\", registry = \"ares\" }"
echo "ares-csf-core = { version = \"0.1.0\", registry = \"ares\" }"
echo "ares-neuromorphic-core = { version = \"0.1.0\", registry = \"ares\" }"
EOF

# Replace placeholder with actual URL
sed -i "s|REGISTRY_URL_PLACEHOLDER|$REGISTRY_URL|g" scripts/use-ares-registry.sh
chmod +x scripts/use-ares-registry.sh

echo -e "${BLUE}📁 Registry files created in: $TEMP_DIR/$REGISTRY_REPO${NC}"
echo -e "${BLUE}🔧 Cargo configured at: $CARGO_CONFIG_FILE${NC}"
echo -e "${BLUE}📜 Scripts created in: /mnt/m/Projects/ARES-51/scripts/${NC}"

# Cleanup
cd /mnt/m/Projects/ARES-51
echo -e "\n${GREEN}✨ Setup complete! Follow the next steps above to finish the process.${NC}"
EOF