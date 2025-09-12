#!/bin/bash
set -euo pipefail

# ARES CSF Protocol Compliance Checker
# Enforces the protocol migration by preventing duplicate PhasePacket definitions
# and ensuring canonical type usage across the workspace
#
# This script is now automatically run in CI/CD pipeline (.github/workflows/ci.yml)
# and as a pre-commit hook (.git/hooks/pre-commit) to prevent protocol violations

echo "🔍 ARES CSF Protocol Compliance Check"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
VIOLATIONS=0
WARNINGS=0

# Function to report violation
report_violation() {
    local message="$1"
    local file="${2:-}"
    echo -e "${RED}❌ VIOLATION:${NC} $message"
    if [ -n "$file" ]; then
        echo -e "   📁 File: $file"
    fi
    ((VIOLATIONS++))
}

# Function to report warning
report_warning() {
    local message="$1"
    local file="${2:-}"
    echo -e "${YELLOW}⚠️  WARNING:${NC} $message"
    if [ -n "$file" ]; then
        echo -e "   📁 File: $file"
    fi
    ((WARNINGS++))
}

# Function to report success
report_success() {
    local message="$1"
    echo -e "${GREEN}✅ $message${NC}"
}

echo "1. Checking for duplicate PhasePacket struct definitions..."
echo "-----------------------------------------------------------"

# Find all PhasePacket struct definitions (excluding csf-protocol canonical definition)
PHASE_PACKET_DEFS=$(rg "struct PhasePacket" --type rust -n | grep -v "crates/csf-protocol/src/packet.rs" || true)

if [ -n "$PHASE_PACKET_DEFS" ]; then
    echo -e "${RED}❌ Found duplicate PhasePacket definitions:${NC}"
    echo "$PHASE_PACKET_DEFS"
    echo ""
    report_violation "Duplicate PhasePacket struct definitions found outside of csf-protocol"
    echo "   All components must use the canonical PhasePacket from csf-protocol"
    echo "   Use compatibility methods or re-exports instead of defining new structs"
else
    report_success "No duplicate PhasePacket definitions found"
fi

echo ""
echo "2. Checking for proper csf-protocol imports..."
echo "----------------------------------------------"

# Check that components are using csf-protocol types properly
COMPONENTS=("csf-bus" "csf-core" "csf-clogic" "csf-network" "csf-ffi")
CANONICAL_IMPORTS=("PhasePacket" "PacketFlags" "PacketHeader" "PacketPayload")

for component in "${COMPONENTS[@]}"; do
    if [ -d "crates/$component" ]; then
        echo "Checking $component..."
        
        # Check if component has proper imports
        for import_type in "${CANONICAL_IMPORTS[@]}"; do
            # Look for direct local definitions (bad)
            LOCAL_DEFS=$(rg "struct $import_type" --type rust "crates/$component" | grep -v "pub use" || true)
            if [ -n "$LOCAL_DEFS" ]; then
                report_violation "Component $component defines $import_type locally instead of using canonical type"
                echo "$LOCAL_DEFS"
            fi
            
            # Look for proper re-exports or use statements (good)
            PROPER_USAGE=$(rg "(use.*csf_protocol.*$import_type|pub use.*csf_protocol.*$import_type)" --type rust "crates/$component" || true)
            if [ -n "$PROPER_USAGE" ]; then
                echo -e "   ${GREEN}✓${NC} $component properly imports/re-exports $import_type"
            fi
        done
    else
        report_warning "Component directory crates/$component not found"
    fi
done

echo ""
echo "3. Checking for deprecated packet type patterns..."
echo "-------------------------------------------------"

# Check for old-style packet definitions
DEPRECATED_PATTERNS=(
    "struct.*Packet.*\{"
    "enum.*PacketType.*\{"
    "struct.*PacketFlags.*\{"
)

for pattern in "${DEPRECATED_PATTERNS[@]}"; do
    echo "Checking pattern: $pattern"
    MATCHES=$(rg "$pattern" --type rust crates/ | grep -v "crates/csf-protocol" | grep -v "pub use" || true)
    if [ -n "$MATCHES" ]; then
        report_warning "Found potential deprecated packet pattern"
        echo "$MATCHES"
    fi
done

echo ""
echo "4. Validating workspace compilation consistency..."
echo "-----------------------------------------------"

# Check that all core components compile
echo "Testing compilation of core components..."
CORE_COMPONENTS=("csf-shared-types" "csf-protocol" "csf-core" "csf-bus" "csf-clogic")

for component in "${CORE_COMPONENTS[@]}"; do
    echo -n "Checking $component compilation... "
    if cargo check -p "$component" --quiet 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        report_violation "Component $component failed to compile" "crates/$component"
    fi
done

echo ""
echo "5. Checking for proper feature flags..."
echo "--------------------------------------"

# Verify that optional features are properly gated
OPTIONAL_FEATURES=("mlir" "python" "cuda")
for feature in "${OPTIONAL_FEATURES[@]}"; do
    echo "Checking feature: $feature"
    # Look for ungated usage of optional dependencies
    case $feature in
        "mlir")
            UNGATED_USAGE=$(rg "csf_mlir::" --type rust crates/ | grep -v "#\[cfg(feature.*mlir" | head -5 || true)
            if [ -n "$UNGATED_USAGE" ]; then
                report_warning "Found potentially ungated csf_mlir usage:"
                echo "$UNGATED_USAGE"
            fi
            ;;
        "python")
            UNGATED_USAGE=$(rg "pyo3::|PyResult|PyModule" --type rust crates/ | grep -v "#\[cfg(feature.*python" | head -5 || true)
            if [ -n "$UNGATED_USAGE" ]; then
                report_warning "Found potentially ungated Python integration usage"
            fi
            ;;
        "cuda")
            UNGATED_USAGE=$(rg "cuda|nvidia" --type rust crates/ | grep -v "#\[cfg(feature.*cuda" | head -5 || true)
            if [ -n "$UNGATED_USAGE" ]; then
                report_warning "Found potentially ungated CUDA usage"
            fi
            ;;
    esac
done

echo ""
echo "6. Checking protocol version consistency..."
echo "-----------------------------------------"

# Make sure all references to protocol version are consistent
PROTOCOL_VERSION_REFS=$(rg "PROTOCOL_VERSION|protocol.*version" --type rust crates/ || true)
if [ -n "$PROTOCOL_VERSION_REFS" ]; then
    echo "Found protocol version references:"
    echo "$PROTOCOL_VERSION_REFS"
    # This is informational for now
fi

echo ""
echo "=========================================="
echo "📊 PROTOCOL COMPLIANCE SUMMARY"
echo "=========================================="

if [ $VIOLATIONS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED! 🎉${NC}"
    echo "The ARES CSF protocol migration is compliant."
    echo "All components properly use canonical types from csf-protocol."
    exit 0
elif [ $VIOLATIONS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  PASSED WITH WARNINGS ⚠️${NC}"
    echo "Found $WARNINGS warning(s) but no violations."
    echo "Consider addressing warnings for better code quality."
    exit 0
else
    echo -e "${RED}❌ COMPLIANCE CHECK FAILED ❌${NC}"
    echo "Found $VIOLATIONS violation(s) and $WARNINGS warning(s)."
    echo ""
    echo "Protocol migration compliance requirements:"
    echo "1. Only csf-protocol should define canonical PhasePacket struct"
    echo "2. All components must import/re-export from csf-protocol"
    echo "3. No local packet type definitions outside csf-protocol"
    echo "4. All core components must compile successfully"
    echo "5. Optional features must be properly gated"
    echo ""
    echo "Please fix violations before merging changes."
    exit 1
fi