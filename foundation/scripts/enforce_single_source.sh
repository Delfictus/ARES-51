#!/bin/bash
# CLAUDE.MD SINGLE SOURCE ENFORCEMENT
# ZERO TOLERANCE FOR ALTERNATIVE CONFIGURATIONS

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "SINGLE SOURCE OF TRUTH ENFORCEMENT"
echo "========================================="

VIOLATIONS=0

# Check for ONLY ONE CLAUDE.md
MD_COUNT=$(find . -name "CLAUDE.md" -type f 2>/dev/null | wc -l)
if [ "$MD_COUNT" -eq "0" ]; then
    echo -e "${RED}✗ FATAL: CLAUDE.md not found in repository${NC}"
    echo -e "${RED}This is the ONLY accepted configuration file${NC}"
    exit 1
elif [ "$MD_COUNT" -gt "1" ]; then
    echo -e "${RED}✗ FATAL: Multiple CLAUDE.md files detected:${NC}"
    find . -name "CLAUDE.md" -type f
    echo -e "${RED}ONLY ./CLAUDE.md is permitted${NC}"
    exit 1
fi

# Verify it's in the root
if [ ! -f "./CLAUDE.md" ]; then
    echo -e "${RED}✗ FATAL: CLAUDE.md must be in repository root${NC}"
    echo "Found at: $(find . -name "CLAUDE.md" -type f)"
    exit 1
fi

# Check for forbidden alternative configs
echo -e "\n${YELLOW}Checking for forbidden alternative configurations...${NC}"

FORBIDDEN_PATTERNS=(
    "CLAUDE_*.md"
    "claude_*.md"
    "*_CLAUDE.md"
    "*_claude.md"
    "ALTERNATE*.md"
    "LOCAL_CLAUDE*.md"
    ".continue/**/*.md"
    "docs/CLAUDE*.md"
    "docs/claude*.md"
    "config/claude*"
    ".github/CLAUDE*"
)

for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    FOUND=$(find . -path "./.git" -prune -o -path "$pattern" -type f -print 2>/dev/null | head -5)
    if [ -n "$FOUND" ]; then
        echo -e "${RED}✗ FATAL: Forbidden configuration detected:${NC}"
        echo "$FOUND"
        echo -e "${RED}Pattern: $pattern${NC}"
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done

# Check for any instruction files in .continue
if [ -d ".continue" ]; then
    CONTINUE_FILES=$(find .continue -name "*.md" -o -name "*.txt" 2>/dev/null | wc -l)
    if [ "$CONTINUE_FILES" -gt "0" ]; then
        echo -e "${RED}✗ FATAL: .continue/ contains configuration files:${NC}"
        find .continue -name "*.md" -o -name "*.txt" | head -10
        echo -e "${RED}No .continue overrides are permitted${NC}"
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
fi

# Verify CLAUDE.md hasn't been tampered with
echo -e "\n${YELLOW}Verifying CLAUDE.md integrity...${NC}"

if ! grep -q "ABSOLUTE SINGLE SOURCE OF TRUTH" CLAUDE.md; then
    echo -e "${RED}✗ FATAL: CLAUDE.md header has been tampered with${NC}"
    echo -e "${RED}Missing: 'ABSOLUTE SINGLE SOURCE OF TRUTH'${NC}"
    exit 1
fi

if ! grep -q "THIS IS THE ONLY CLAUDE.MD FILE" CLAUDE.md; then
    echo -e "${RED}✗ FATAL: CLAUDE.md authority statement missing${NC}"
    exit 1
fi

if ! grep -q "89+ PLACEHOLDERS" CLAUDE.md; then
    echo -e "${RED}✗ FATAL: CLAUDE.md placeholder tracking section missing${NC}"
    exit 1
fi

# Check file permissions (should not be world-writable)
PERMS=$(stat -c %a CLAUDE.md 2>/dev/null || stat -f %A CLAUDE.md)
if [[ "$PERMS" == *"7" ]]; then
    echo -e "${YELLOW}⚠ WARNING: CLAUDE.md is world-writable. Fixing...${NC}"
    chmod 644 CLAUDE.md
fi

# Final report
echo -e "\n========================================="

if [ "$VIOLATIONS" -gt "0" ]; then
    echo -e "${RED}✗ SINGLE SOURCE ENFORCEMENT FAILED${NC}"
    echo -e "${RED}Found $VIOLATIONS violation(s)${NC}"
    echo -e "${RED}Remove all alternative configurations immediately${NC}"
    exit 1
else
    echo -e "${GREEN}✓ SINGLE SOURCE VERIFICATION PASSED${NC}"
    echo -e "${GREEN}./CLAUDE.md is the ONLY configuration${NC}"
    echo -e "${GREEN}No alternative configs detected${NC}"
    
    # Show current progress
    COMPLETE=$(grep -c "\[x\]" CLAUDE.md 2>/dev/null || echo "0")
    TOTAL=$(grep -c "\[ \]" CLAUDE.md 2>/dev/null || echo "89")
    PERCENT=$((COMPLETE * 100 / (COMPLETE + TOTAL)))
    
    echo -e "\n${YELLOW}Implementation Progress:${NC}"
    echo -e "Completed: ${GREEN}$COMPLETE${NC} / $((COMPLETE + TOTAL))"
    echo -e "Progress: ${YELLOW}${PERCENT}%${NC}"
    
    if [ "$PERCENT" -lt "100" ]; then
        echo -e "\n${YELLOW}⚠ $TOTAL placeholders remain to be implemented${NC}"
        echo -e "${YELLOW}Target: 5 placeholders/day for Proof of Power${NC}"
    fi
fi

echo "========================================="
exit 0