#!/bin/bash
# Initialize ARES registry for delfictus GitHub account

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Initializing ARES Registry for delfictus${NC}"
echo "============================================="

# Configuration
GITHUB_USERNAME="delfictus"
REGISTRY_REPO="ares-registry"
CRATES_REPO="ares-crates"
REGISTRY_URL="https://github.com/${GITHUB_USERNAME}/${REGISTRY_REPO}.git"
CRATES_URL="https://github.com/${GITHUB_USERNAME}/${CRATES_REPO}"

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "GitHub Username: $GITHUB_USERNAME"
echo "Registry URL: $REGISTRY_URL"
echo "Crates URL: $CRATES_URL"

# Create temporary directory for registry setup
TEMP_DIR=$(mktemp -d)
echo -e "\n${BLUE}📁 Setting up registry in: $TEMP_DIR${NC}"

cd "$TEMP_DIR"

# Clone or create registry repository
if git clone "$REGISTRY_URL" "$REGISTRY_REPO" 2>/dev/null; then
    echo -e "${GREEN}✅ Registry repository cloned${NC}"
    cd "$REGISTRY_REPO"
else
    echo -e "${YELLOW}📁 Creating new registry repository${NC}"
    mkdir "$REGISTRY_REPO"
    cd "$REGISTRY_REPO"
    git init
    git remote add origin "$REGISTRY_URL"
fi

# Create registry structure
echo -e "\n${BLUE}🏗️  Creating registry structure...${NC}"

# Create index directory
mkdir -p index

# Create config.json
cat > config.json << EOF
{
  "dl": "${CRATES_URL}/releases/download/{crate}-{version}/{crate}-{version}.crate",
  "api": "${CRATES_URL}"
}
EOF

# Create comprehensive README
cat > README.md << 'EOF'
# ARES Personal Cargo Registry

Private Cargo registry for ARES library crates by delfictus.

## Available Crates

| Crate | Description | Version |
|-------|-------------|---------|
| `ares-spike-encoding` | Neural spike encoding algorithms for neuromorphic computing | 0.1.0 |
| `ares-csf-core` | ChronoSynclastic Fabric core infrastructure | 0.1.0 |
| `ares-neuromorphic-core` | Neuromorphic computation engine with reservoir computing | 0.1.0 |

## Usage

### 1. Configure Registry

Add to your `~/.cargo/config.toml`:

```toml
[registries]
ares = { index = "https://github.com/delfictus/ares-registry.git" }

[source.ares]
registry = "https://github.com/delfictus/ares-registry.git"
```

### 2. Use in Projects

Add to your `Cargo.toml`:

```toml
[dependencies]
ares-spike-encoding = { version = "0.1.0", registry = "ares" }
ares-csf-core = { version = "0.1.0", registry = "ares" }
ares-neuromorphic-core = { version = "0.1.0", registry = "ares" }
```

### 3. Example Usage

```rust
use ares_spike_encoding::{SpikeEncoder, EncodingMethod};
use ares_csf_core::prelude::*;
use ares_neuromorphic_core::NeuromorphicEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create spike encoder
    let mut encoder = SpikeEncoder::new(1000, 1000.0)?
        .with_method(EncodingMethod::Rate)?;
    
    // Create CSF context
    let config = CSFConfig::default();
    let csf_context = CSFContext::new(config)?;
    
    // Create neuromorphic engine
    let neuro_config = ares_neuromorphic_core::EngineConfig::default();
    let neuro_engine = NeuromorphicEngine::new(neuro_config).await?;
    
    println!("ARES libraries loaded successfully!");
    Ok(())
}
```

## Publishing

For maintainers only:

```bash
# Publish all crates
cargo publish --registry ares

# Or use the automation script
./scripts/publish-ares-crates.sh
```

## Security

This is a private registry. Access is controlled through GitHub repository permissions.

## Support

For issues and questions, please contact the maintainer or create an issue in the ARES-51 repository.
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
.DS_Store
Thumbs.db
*.swp
*.swo
*~
.vscode/
.idea/
target/
Cargo.lock
EOF

# Create registry metadata
cat > registry.json << EOF
{
  "name": "ARES Registry",
  "description": "Private Cargo registry for ARES neuromorphic computing libraries",
  "owner": "${GITHUB_USERNAME}",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "crates": [
    {
      "name": "ares-spike-encoding",
      "description": "Neural spike encoding algorithms"
    },
    {
      "name": "ares-csf-core", 
      "description": "ChronoSynclastic Fabric core infrastructure"
    },
    {
      "name": "ares-neuromorphic-core",
      "description": "Neuromorphic computation engine"
    }
  ]
}
EOF

# Commit everything
git add .
git commit -m "🎉 Initialize ARES registry for delfictus

- Registry index structure for Cargo
- Configuration for GitHub releases distribution
- Comprehensive documentation and usage examples
- Support for ares-spike-encoding, ares-csf-core, ares-neuromorphic-core
- Private registry setup with proper .gitignore"

echo -e "\n${GREEN}✅ Registry initialized successfully!${NC}"
echo -e "\n${YELLOW}📋 Next Steps:${NC}"
echo "1. Create repositories on GitHub:"
echo "   - https://github.com/delfictus/ares-registry (private recommended)"
echo "   - https://github.com/delfictus/ares-crates (private recommended)"
echo ""
echo "2. Push this registry:"
echo "   cd $TEMP_DIR/$REGISTRY_REPO"
echo "   git push -u origin main"
echo ""
echo "3. Test the configuration:"
echo "   cd /mnt/m/Projects/ARES-51"
echo "   ./scripts/test-registry.sh"
echo ""
echo "4. Publish your first crates:"
echo "   ./scripts/publish-ares-crates.sh"
echo ""

echo -e "${BLUE}📁 Registry files created in: $TEMP_DIR/$REGISTRY_REPO${NC}"
EOF