#!/bin/bash

# Test ARES Resonance Trading System
# Author: Ididia Serfaty

echo "╔══════════════════════════════════════════════════╗"
echo "║   ARES Quantum Resonance Trading System Test    ║"
echo "║          Phase Correlation Analysis             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Set resonance parameters
export RUST_LOG=ares_trading=info

echo "🔮 Resonance Trading Features:"
echo "  ✓ Phase oscillator network"
echo "  ✓ Cross-asset correlation detection"
echo "  ✓ Kuramoto dynamics coupling"
echo "  ✓ Spectral frequency analysis"
echo "  ✓ Pattern memory learning"
echo ""

echo "📊 Trading Configuration:"
echo "  • Initial Balance: $100,000"
echo "  • Resonance Threshold: 0.75"
echo "  • Min Confidence: 0.7"
echo "  • Symbols: AAPL, GOOGL, MSFT, TSLA, AMZN, META, NVDA"
echo ""

echo "Starting resonance analysis..."
echo "Watch for 🔮 RESONANCE signals in the output!"
echo ""

cd /home/diddy/dev/ares-monorepo

# Run with timeout for testing
timeout 30 cargo run -p ares-trading 2>&1 | grep -E "resonance|RESONANCE|🔮|phase|coherence|Starting|Balance"

echo ""
echo "Test completed. Full system available with:"
echo "  cargo run -p ares-trading"
echo ""
echo "For long-running tests, use:"
echo "  ./start-trading.sh"