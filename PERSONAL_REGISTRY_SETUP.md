# Personal Cargo Registry Setup Guide

This guide will help you set up a personal Cargo registry to distribute your ARES library crates privately.

## Option 1: Git-based Registry (Recommended for Private Use)

### 1. Create a Git Repository for Your Registry

```bash
# Create a new repository for your registry
mkdir ares-registry
cd ares-registry
git init
git remote add origin git@github.com:your-username/ares-registry.git
# or use your preferred Git hosting (GitLab, private Git server, etc.)
```

### 2. Set up Registry Structure

```bash
# Create registry index structure
mkdir -p index/{ar,es}/es-{c,s}
mkdir -p index/ar/es-{n,s}

# Create config.json for the registry
cat > config.json << 'EOF'
{
  "dl": "https://github.com/your-username/ares-crates/releases/download/{crate}-{version}/{crate}-{version}.crate",
  "api": "https://github.com/your-username/ares-registry"
}
EOF

git add .
git commit -m "Initialize ARES registry"
git push -u origin main
```

### 3. Configure Cargo to Use Your Registry

Add this to your `~/.cargo/config.toml`:

```toml
[registries]
ares = { index = "https://github.com/your-username/ares-registry.git" }

[source.ares]
registry = "https://github.com/your-username/ares-registry.git"

# Optional: Set as default for ARES crates
[source.crates-io]
replace-with = "ares"
local-registry = "/path/to/local/registry"  # Optional
```

### 4. Publishing Script

Create `scripts/publish.sh` in your ARES-51 directory:

```bash
#!/bin/bash
set -e

REGISTRY="ares"
REGISTRY_URL="https://github.com/your-username/ares-registry.git"

echo "Publishing ARES crates to personal registry..."

# Function to publish a crate
publish_crate() {
    local crate_dir=$1
    local crate_name=$2
    
    echo "Publishing $crate_name..."
    cd "$crate_dir"
    
    # Ensure clean build
    cargo clean
    cargo check
    cargo test
    
    # Package the crate
    cargo package --registry "$REGISTRY"
    
    # Publish to your registry
    cargo publish --registry "$REGISTRY" --allow-dirty
    
    cd -
}

# Publish in dependency order
publish_crate "ares-spike-encoding" "ares-spike-encoding"
publish_crate "ares-csf-core" "ares-csf-core"  
publish_crate "ares-neuromorphic-core" "ares-neuromorphic-core"

echo "All crates published successfully!"
```

## Option 2: Local File-based Registry (For Development)

### 1. Create Local Registry

```bash
# Create local registry directory
mkdir -p ~/.cargo/local-registry
cd ~/.cargo/local-registry

# Initialize the registry index
cargo index init --registry-url file://~/.cargo/local-registry
```

### 2. Configure Cargo

Add to `~/.cargo/config.toml`:

```toml
[registries]
local = { index = "file://~/.cargo/local-registry" }

[source.local]
local-registry = "/home/yourusername/.cargo/local-registry"
```

### 3. Local Publishing Script

Create `scripts/publish-local.sh`:

```bash
#!/bin/bash
set -e

REGISTRY_DIR="$HOME/.cargo/local-registry"

echo "Publishing ARES crates to local registry..."

publish_local() {
    local crate_dir=$1
    local crate_name=$2
    
    echo "Publishing $crate_name locally..."
    cd "$crate_dir"
    
    cargo clean
    cargo package
    
    # Copy to local registry
    cp "target/package/${crate_name}-*.crate" "$REGISTRY_DIR/crates/"
    
    cd -
}

# Ensure registry directories exist
mkdir -p "$REGISTRY_DIR/crates"

# Publish crates
publish_local "ares-spike-encoding" "ares-spike-encoding"
publish_local "ares-csf-core" "ares-csf-core"
publish_local "ares-neuromorphic-core" "ares-neuromorphic-core"
```

## Option 3: kellnr.io (Third-party Private Registry Service)

### 1. Set up kellnr.io Account

1. Sign up at https://kellnr.io
2. Create your private registry
3. Get your API token

### 2. Configure Cargo

```toml
[registries]
kellnr = { index = "https://your-registry.kellnr.io/git/index" }

[source.kellnr]
registry = "https://your-registry.kellnr.io/git/index"
```

### 3. Publish to kellnr.io

```bash
# Set your API token
export CARGO_REGISTRIES_KELLNR_TOKEN="your-api-token"

# Publish
cargo publish --registry kellnr
```

## Using Your Crates in Other Projects

Once you've set up your registry, use your crates in other projects:

### In Cargo.toml

```toml
[dependencies]
ares-spike-encoding = { version = "0.1.0", registry = "ares" }
ares-csf-core = { version = "0.1.0", registry = "ares" }
ares-neuromorphic-core = { version = "0.1.0", registry = "ares" }
```

### Example Usage

```rust
// In your project's main.rs or lib.rs
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

## Security Considerations

### For Git-based Registry:
- Use private repositories for proprietary code
- Set up SSH keys for secure access
- Consider using GitLab/GitHub access tokens
- Enable branch protection on your registry repo

### Access Control:
```bash
# Create .gitignore for sensitive files
cat > .gitignore << 'EOF'
target/
*.crate
Cargo.lock
.env
*.key
*.pem
EOF
```

### Environment Variables:
```bash
# Set up environment for secure publishing
export CARGO_REGISTRY_TOKEN="your-secure-token"
export REGISTRY_URL="https://your-secure-registry.com"
```

## Automation with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to Registry

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
      
    - name: Configure Registry
      run: |
        mkdir -p ~/.cargo
        echo '[registries]' > ~/.cargo/config.toml
        echo 'ares = { index = "${{ secrets.REGISTRY_URL }}" }' >> ~/.cargo/config.toml
        
    - name: Publish Crates
      env:
        CARGO_REGISTRIES_ARES_TOKEN: ${{ secrets.REGISTRY_TOKEN }}
      run: |
        chmod +x scripts/publish.sh
        ./scripts/publish.sh
```

## Troubleshooting

### Common Issues:

1. **Permission Denied**: Ensure your SSH keys are set up correctly
2. **Registry Not Found**: Check your `~/.cargo/config.toml` configuration
3. **Version Conflicts**: Ensure version numbers are incremented
4. **Dependency Resolution**: Make sure all dependencies are available in your registry

### Debug Commands:

```bash
# Check registry configuration
cargo config get registries

# Verify crate package
cargo package --list

# Test local build before publishing
cargo build --release

# Check dependency tree
cargo tree
```

Choose the option that best fits your security and collaboration needs!