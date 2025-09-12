#!/bin/bash
# ARES Production Readiness Verification Script
# Zero tolerance for incomplete implementations

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILED=0
WARNINGS=0

echo "========================================="
echo "ARES PRODUCTION READINESS VERIFICATION"
echo "========================================="

# Check for forbidden patterns
echo -e "\n${YELLOW}Checking for forbidden code patterns...${NC}"

# Check for todo!, unimplemented!, etc
if rg -q "todo!\(\)|unimplemented!\(\)|unreachable!\(\)|panic!\(\"not.*implemented\"\)" crates/; then
    echo -e "${RED}✗ CRITICAL: Found incomplete implementations (todo!, unimplemented!)${NC}"
    rg "todo!\(\)|unimplemented!\(\)|unreachable!\(\)|panic!\(\"not.*implemented\"\)" crates/ || true
    FAILED=$((FAILED + 1))
else
    echo -e "${GREEN}✓ No incomplete implementations found${NC}"
fi

# Check for mock/stub/placeholder/dummy
if rg -iq "mock|stub|placeholder|dummy|fake.*data|test.*data|sample.*data" crates/ --type rust; then
    echo -e "${YELLOW}⚠ WARNING: Found potential mock/stub references${NC}"
    echo "Please verify these are legitimate uses:"
    rg -i "mock|stub|placeholder|dummy" crates/ --type rust | head -20 || true
    WARNINGS=$((WARNINGS + 1))
fi

# Build verification
echo -e "\n${YELLOW}Running build verification...${NC}"
if cargo build --release --workspace 2>&1 | grep -E "warning|error"; then
    echo -e "${RED}✗ Build has warnings or errors${NC}"
    FAILED=$((FAILED + 1))
else
    echo -e "${GREEN}✓ Build successful with no warnings${NC}"
fi

# Clippy strict mode
echo -e "\n${YELLOW}Running strict clippy analysis...${NC}"
if ! cargo clippy --workspace --all-features -- -D warnings 2>&1; then
    echo -e "${RED}✗ Clippy found issues${NC}"
    FAILED=$((FAILED + 1))
else
    echo -e "${GREEN}✓ Clippy analysis passed${NC}"
fi

# Test execution
echo -e "\n${YELLOW}Running test suite...${NC}"
if ! cargo test --workspace --all-features --release 2>&1 | tee /tmp/test_output.log; then
    echo -e "${RED}✗ Tests failed${NC}"
    FAILED=$((FAILED + 1))
else
    TEST_COUNT=$(grep -c "test result: ok" /tmp/test_output.log || echo "0")
    echo -e "${GREEN}✓ All tests passed (${TEST_COUNT} test suites)${NC}"
fi

# Check for proper error handling
echo -e "\n${YELLOW}Checking error handling...${NC}"
UNWRAP_COUNT=$(rg "\.unwrap\(\)" crates/ --type rust | wc -l || echo "0")
EXPECT_COUNT=$(rg "\.expect\(" crates/ --type rust | wc -l || echo "0")
if [ "$UNWRAP_COUNT" -gt "50" ]; then
    echo -e "${YELLOW}⚠ WARNING: Found ${UNWRAP_COUNT} unwrap() calls - consider proper error handling${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

# Check documentation coverage
echo -e "\n${YELLOW}Checking documentation...${NC}"
if cargo doc --workspace --no-deps 2>&1 | grep -q "warning"; then
    echo -e "${YELLOW}⚠ WARNING: Documentation has warnings${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓ Documentation complete${NC}"
fi

# Performance benchmark check
echo -e "\n${YELLOW}Checking for benchmarks...${NC}"
BENCH_COUNT=$(find crates/ -name "*.rs" -exec grep -l "#\[bench\]" {} \; | wc -l || echo "0")
if [ "$BENCH_COUNT" -lt "5" ]; then
    echo -e "${YELLOW}⚠ WARNING: Only ${BENCH_COUNT} benchmark files found - consider adding more${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓ Found ${BENCH_COUNT} benchmark files${NC}"
fi

# Check for mathematical validation comments
echo -e "\n${YELLOW}Checking mathematical validation...${NC}"
MATH_PROOF_COUNT=$(rg "proof:|theorem:|lemma:|invariant:" crates/ --type rust | wc -l || echo "0")
if [ "$MATH_PROOF_COUNT" -lt "10" ]; then
    echo -e "${YELLOW}⚠ WARNING: Limited mathematical proofs found (${MATH_PROOF_COUNT})${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓ Found ${MATH_PROOF_COUNT} mathematical validations${NC}"
fi

# Security audit
echo -e "\n${YELLOW}Running security audit...${NC}"
if command -v cargo-audit >/dev/null 2>&1; then
    if cargo audit 2>&1 | grep -q "Vulnerabilities"; then
        echo -e "${RED}✗ Security vulnerabilities found${NC}"
        FAILED=$((FAILED + 1))
    else
        echo -e "${GREEN}✓ No known security vulnerabilities${NC}"
    fi
else
    echo -e "${YELLOW}⚠ cargo-audit not installed - skipping security check${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

# Final report
echo -e "\n========================================="
echo "VERIFICATION COMPLETE"
echo "========================================="

if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}✗ CRITICAL FAILURES: ${FAILED}${NC}"
    echo -e "${RED}This codebase is NOT production ready${NC}"
    echo -e "${RED}Fix all critical issues before proceeding${NC}"
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warnings: ${WARNINGS}${NC}"
    echo -e "${YELLOW}Review warnings and improve where necessary${NC}"
    exit 0
else
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo -e "${GREEN}Codebase meets production standards${NC}"
    exit 0
fi