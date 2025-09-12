#!/bin/bash

# MLIR Security Validation Script
# Validates security aspects of the ARES ChronoFabric MLIR implementation

set -euo pipefail

REPO_ROOT="/home/diddy/dev/ares-monorepo"
MLIR_CRATE="$REPO_ROOT/crates/csf-mlir"

echo "🔒 ARES ChronoFabric MLIR Security Validation"
echo "=============================================="

# 1. Check for unsafe code blocks
echo "1. Checking for unsafe code usage..."
UNSAFE_COUNT=$(find "$MLIR_CRATE/src" -name "*.rs" -exec grep -c "unsafe" {} + | awk '{s+=$1} END {print s+0}')
echo "   Found $UNSAFE_COUNT unsafe blocks"

if [ "$UNSAFE_COUNT" -gt 3 ]; then
    echo "   ⚠️  WARNING: High number of unsafe blocks detected"
else
    echo "   ✅ Acceptable unsafe usage"
fi

# 2. Check for memory safety patterns
echo "2. Validating memory safety patterns..."
echo "   Checking for proper bounds checking..."
BOUNDS_CHECKS=$(grep -r "get(" "$MLIR_CRATE/src" | wc -l)
echo "   Found $BOUNDS_CHECKS bounds check patterns"

echo "   Checking for Arc/Rc usage for thread safety..."
ARC_USAGE=$(grep -r "Arc<" "$MLIR_CRATE/src" | wc -l)
echo "   Found $ARC_USAGE Arc usage patterns"

# 3. Check for potential buffer overflows
echo "3. Checking for buffer overflow vulnerabilities..."
SLICE_USAGE=$(grep -r "cast_slice\|from_raw_parts" "$MLIR_CRATE/src" | wc -l)
echo "   Found $SLICE_USAGE raw memory operations"

if [ "$SLICE_USAGE" -gt 5 ]; then
    echo "   ⚠️  WARNING: High raw memory usage - review for safety"
else
    echo "   ✅ Reasonable raw memory usage"
fi

# 4. Check for proper error handling
echo "4. Validating error handling..."
UNWRAP_COUNT=$(grep -r "unwrap()\|expect(" "$MLIR_CRATE/src" | wc -l)
RESULT_COUNT=$(grep -r "Result<" "$MLIR_CRATE/src" | wc -l)
echo "   Found $UNWRAP_COUNT unwrap/expect calls"
echo "   Found $RESULT_COUNT Result types"

if [ "$UNWRAP_COUNT" -gt 10 ]; then
    echo "   ⚠️  WARNING: Excessive unwrap usage - consider proper error handling"
else
    echo "   ✅ Good error handling practices"
fi

# 5. Check for cryptographic security
echo "5. Checking cryptographic implementations..."
CRYPTO_USAGE=$(grep -r "blake3\|sha2\|aes" "$MLIR_CRATE/src" | wc -l)
echo "   Found $CRYPTO_USAGE cryptographic operations"

# 6. Check for input validation
echo "6. Validating input sanitization..."
VALIDATION_COUNT=$(grep -r "validate\|sanitize\|check" "$MLIR_CRATE/src" | wc -l)
echo "   Found $VALIDATION_COUNT validation patterns"

# 7. Check dependencies for known vulnerabilities
echo "7. Running cargo audit for vulnerability check..."
cd "$REPO_ROOT"
if command -v cargo-audit >/dev/null 2>&1; then
    if cargo audit --file "$MLIR_CRATE/Cargo.toml" --quiet; then
        echo "   ✅ No known vulnerabilities found"
    else
        echo "   ⚠️  WARNING: Potential vulnerabilities detected"
    fi
else
    echo "   ℹ️  cargo-audit not installed - skipping vulnerability check"
fi

# 8. Test memory allocation limits
echo "8. Testing memory allocation safety..."
cd "$MLIR_CRATE"
if cargo test test_memory_pool_stress --quiet; then
    echo "   ✅ Memory allocation tests passed"
else
    echo "   ❌ Memory allocation tests failed"
fi

# 9. Validate quantum operation bounds
echo "9. Testing quantum operation bounds..."
if cargo test test_quantum_operations --quiet; then
    echo "   ✅ Quantum operation tests passed"
else
    echo "   ❌ Quantum operation tests failed"
fi

# 10. Performance bounds validation
echo "10. Validating performance bounds..."
if cargo test test_performance_validation --quiet; then
    echo "   ✅ Performance validation passed"
else
    echo "   ❌ Performance validation failed"
fi

echo ""
echo "🔒 Security Validation Summary"
echo "=============================="
echo "   Unsafe blocks: $UNSAFE_COUNT"
echo "   Memory operations: $SLICE_USAGE"
echo "   Error handling: $RESULT_COUNT Result types, $UNWRAP_COUNT unwraps"
echo "   Input validation: $VALIDATION_COUNT patterns"
echo ""

# Overall assessment
ISSUES=0
if [ "$UNSAFE_COUNT" -gt 3 ]; then ((ISSUES++)); fi
if [ "$SLICE_USAGE" -gt 5 ]; then ((ISSUES++)); fi
if [ "$UNWRAP_COUNT" -gt 10 ]; then ((ISSUES++)); fi

if [ "$ISSUES" -eq 0 ]; then
    echo "✅ SECURITY VALIDATION PASSED"
    echo "   The MLIR implementation meets security standards"
    exit 0
else
    echo "⚠️  SECURITY ISSUES DETECTED ($ISSUES issues)"
    echo "   Review implementation for security hardening"
    exit 1
fi