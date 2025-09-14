#!/bin/bash
# PRCT Docker Entrypoint Script
# Handles initialization and validation

set -e

# Color output for better UX
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🧬 PRCT Algorithm Validation System${NC}"
echo "============================================"

# Check GPU availability
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo -e "${YELLOW}⚠${NC} No GPU detected - running in CPU mode"
fi

# Verify PRCT installation
if [ -d "${PRCT_HOME}" ]; then
    echo -e "${GREEN}✓${NC} PRCT installation found at ${PRCT_HOME}"
    echo "  Version: $(prct-validator --version 2>/dev/null || echo 'unknown')"
else
    echo -e "${RED}✗${NC} PRCT installation not found"
    exit 1
fi

# Check data directory
if [ -d "${PRCT_DATA}" ]; then
    echo -e "${GREEN}✓${NC} Data directory found with $(ls -1 ${PRCT_DATA}/*.pdb 2>/dev/null | wc -l) PDB files"
else
    echo -e "${YELLOW}⚠${NC} Data directory not found - creating..."
    mkdir -p ${PRCT_DATA}
fi

# Memory check
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -ge 32 ]; then
    echo -e "${GREEN}✓${NC} Memory check passed: ${TOTAL_MEM}GB available"
else
    echo -e "${YELLOW}⚠${NC} Low memory warning: ${TOTAL_MEM}GB available (recommended: 32GB+)"
fi

echo "============================================"
echo -e "${GREEN}System ready for PRCT validation${NC}"
echo ""

# Execute command
exec "$@"