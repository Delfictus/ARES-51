#!/bin/bash

echo "Starting ARES Paper Trading System"
echo "==================================="
echo ""
echo "This system simulates stock trading using:"
echo "- Technical Analysis (RSI, SMA, Momentum)"
echo "- $100,000 starting capital"
echo "- Real-time simulated market data"
echo "- Automated buy/sell decisions"
echo ""
echo "Press Ctrl+C to stop the trading simulation"
echo ""

# Run the paper trading system
cargo run --release --bin paper-trader -p ares-trading