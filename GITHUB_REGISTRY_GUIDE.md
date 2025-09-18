# GitHub Personal Registry Setup Guide

## Quick Setup Steps

### 1. Create GitHub Repositories

Go to GitHub and create these two repositories:

1. **Registry Index Repository**: `ares-registry` (private recommended)
   - This holds the registry index
   - Clone URL: `https://github.com/YOUR_USERNAME/ares-registry.git`

2. **Crates Storage Repository**: `ares-crates` (private recommended)  
   - This holds the actual `.crate` files as releases
   - Clone URL: `https://github.com/YOUR_USERNAME/ares-crates.git`

### 2. Configure Your Cargo

Add this to your `~/.cargo/config.toml` (create if it doesn't exist):

```toml
[registries]
ares = { index = "https://github.com/YOUR_USERNAME/ares-registry.git" }

[source.ares]
registry = "https://github.com/YOUR_USERNAME/ares-registry.git"
```

### 3. Initialize the Registry Index

```bash
# Clone your empty registry repo
git clone https://github.com/YOUR_USERNAME/ares-registry.git
cd ares-registry

# Create the registry structure
mkdir -p index

# Create config.json
cat > config.json << 'EOF'
{
  "dl": "https://github.com/YOUR_USERNAME/ares-crates/releases/download/{crate}-{version}/{crate}-{version}.crate",
  "api": "https://github.com/YOUR_USERNAME/ares-crates"
}
EOF

# Create README
cat > README.md << 'EOF'
# ARES Personal Cargo Registry

Private registry for ARES library crates.

## Usage

```toml
[dependencies] 
ares-spike-encoding = { version = "0.1.0", registry = "ares" }
ares-csf-core = { version = "0.1.0", registry = "ares" }
ares-neuromorphic-core = { version = "0.1.0", registry = "ares" }
```
EOF

# Commit and push
git add .
git commit -m "Initialize ARES registry"
git push origin main
```

### 4. Test Publishing

From your ARES-51 directory:

```bash
# Test with spike encoding first (no dependencies)
cd ares-spike-encoding
cargo package --registry ares
cargo publish --registry ares --dry-run

# If successful, do the real publish
cargo publish --registry ares
```

### 5. Manual Registry Index Update

After publishing, you need to manually update the registry index:

```bash
cd ../ares-registry
git pull

# Create index entry for the crate
mkdir -p index/ar/es
echo '{"name":"ares-spike-encoding","vers":"0.1.0","deps":[],"features":{},"cksum":"CHECKSUM_HERE","yanked":false,"links":null,"v":2}' >> index/ar/es/ares-spike-encoding

git add .
git commit -m "Add ares-spike-encoding 0.1.0"
git push
```

### 6. Create GitHub Release

1. Go to your `ares-crates` repository on GitHub
2. Click "Releases" → "Create a new release"
3. Tag: `ares-spike-encoding-0.1.0`
4. Title: `ares-spike-encoding v0.1.0`
5. Upload the `.crate` file from `ares-spike-encoding/target/package/`
6. Publish release

## Using in Other Projects

In any other project, add to `Cargo.toml`:

```toml
[dependencies]
ares-spike-encoding = { version = "0.1.0", registry = "ares" }
ares-csf-core = { version = "0.1.0", registry = "ares" }
ares-neuromorphic-core = { version = "0.1.0", registry = "ares" }
```

## Automation (Optional)

The included scripts can help automate this process:

- `scripts/setup-github-registry.sh` - Interactive setup
- `scripts/publish-to-registry.sh` - Automated publishing
- `.github/workflows/publish-registry.yml` - GitHub Actions automation

## Security

- Keep repositories private for proprietary code
- Use personal access tokens for authentication
- Set up proper branch protection rules