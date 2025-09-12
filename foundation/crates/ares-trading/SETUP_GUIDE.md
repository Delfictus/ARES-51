# ARES Paper Trading - Long Running Test Setup Guide

## Quick Start (5 Minutes)

### Option 1: FREE Yahoo Finance (Recommended for Testing)

```bash
# From the monorepo root directory
cd /home/diddy/dev/ares-monorepo

# Run with default configuration (Yahoo Finance, no API key needed)
cargo run -p ares-trading
```

This will start paper trading with:
- $100,000 virtual balance
- Yahoo Finance real market data (FREE)
- 5 major tech stocks (AAPL, GOOGL, MSFT, TSLA, AMZN)
- Technical analysis strategy (RSI, SMA, Momentum)

### Option 2: Cryptocurrency Trading (Binance - Also FREE)

```bash
# Edit the config to use Binance
cd crates/ares-trading
nano config.toml
```

Change these lines:
```toml
[market_data]
provider = "binance"  # Changed from yahoo_finance

[[symbols]]
symbol = "BTC-USD"
asset_type = "crypto"

[[symbols]]
symbol = "ETH-USD"
asset_type = "crypto"

[[symbols]]
symbol = "BNB-USD"
asset_type = "crypto"
```

Then run:
```bash
cargo run --bin paper-trader
```

## Long Running Test Setup (Production-Like)

### 1. Create a Dedicated Test Environment

```bash
# Create test directory
mkdir -p ~/ares-trading-test
cd ~/ares-trading-test

# Copy configuration
cp /home/diddy/dev/ares-monorepo/crates/ares-trading/config.toml ./

# Create log directory
mkdir logs
```

### 2. Configure for Long Running Test

Edit `config.toml`:

```toml
[trading]
initial_balance = 100000.0
max_position_size = 0.05      # 5% per position (more conservative)
stop_loss_percent = 1.5        # Tighter stop loss
take_profit_percent = 3.0      # Realistic profit target
max_open_positions = 20        # More positions for diversification

# Add more symbols for better testing
[[symbols]]
symbol = "AAPL"
asset_type = "stock"

[[symbols]]
symbol = "GOOGL"
asset_type = "stock"

[[symbols]]
symbol = "MSFT"
asset_type = "stock"

[[symbols]]
symbol = "AMZN"
asset_type = "stock"

[[symbols]]
symbol = "TSLA"
asset_type = "stock"

[[symbols]]
symbol = "META"
asset_type = "stock"

[[symbols]]
symbol = "NVDA"
asset_type = "stock"

[[symbols]]
symbol = "JPM"
asset_type = "stock"

[[symbols]]
symbol = "BAC"
asset_type = "stock"

[[symbols]]
symbol = "SPY"
asset_type = "stock"

[logging]
level = "info"
file = "logs/trading.log"
```

### 3. Create a Systemd Service (For 24/7 Operation)

```bash
# Create service file
sudo nano /etc/systemd/system/ares-trading.service
```

Add this content:
```ini
[Unit]
Description=ARES Paper Trading System
After=network.target

[Service]
Type=simple
User=diddy
WorkingDirectory=/home/diddy/ares-trading-test
ExecStart=/home/diddy/dev/ares-monorepo/target/release/paper-trader
Restart=always
RestartSec=10
StandardOutput=append:/home/diddy/ares-trading-test/logs/stdout.log
StandardError=append:/home/diddy/ares-trading-test/logs/stderr.log
Environment="RUST_LOG=ares_trading=info"
Environment="RUST_BACKTRACE=1"

[Install]
WantedBy=multi-user.target
```

Build release version and start service:
```bash
# Build optimized release version
cd /home/diddy/dev/ares-monorepo
cargo build --release -p ares-trading

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ares-trading
sudo systemctl start ares-trading

# Check status
sudo systemctl status ares-trading

# View logs
tail -f ~/ares-trading-test/logs/stdout.log
```

### 4. Using Screen/Tmux for Long Running Test (Alternative)

```bash
# Using screen
screen -S trading

# Inside screen session
cd /home/diddy/dev/ares-monorepo
RUST_LOG=ares_trading=info cargo run -p ares-trading 2>&1 | tee ~/ares-trading-test/logs/trading-$(date +%Y%m%d).log

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r trading
```

Or with tmux:
```bash
# Using tmux
tmux new -s trading

# Inside tmux session
cd /home/diddy/dev/ares-monorepo
RUST_LOG=ares_trading=info cargo run -p ares-trading 2>&1 | tee ~/ares-trading-test/logs/trading-$(date +%Y%m%d).log

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t trading
```

## Monitoring Your Test

### 1. Create Monitoring Script

```bash
nano ~/ares-trading-test/monitor.sh
```

Add this content:
```bash
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear
echo -e "${GREEN}=== ARES Paper Trading Monitor ===${NC}"
echo ""

# Check if service is running
if systemctl is-active --quiet ares-trading; then
    echo -e "Status: ${GREEN}RUNNING${NC}"
else
    echo -e "Status: ${RED}STOPPED${NC}"
fi

echo ""
echo "Recent Trades:"
echo "-------------"
grep "Executing trade" ~/ares-trading-test/logs/stdout.log | tail -5

echo ""
echo "Latest Portfolio Status:"
echo "----------------------"
grep "Current Balance" ~/ares-trading-test/logs/stdout.log | tail -1
grep "Total Equity" ~/ares-trading-test/logs/stdout.log | tail -1
grep "Realized P&L" ~/ares-trading-test/logs/stdout.log | tail -1

echo ""
echo "Error Count (last hour):"
echo "----------------------"
echo $(find ~/ares-trading-test/logs -name "*.log" -mmin -60 -exec grep -c ERROR {} \; | paste -sd+ | bc)

echo ""
echo "Market Data Updates (last 5):"
echo "---------------------------"
grep "Market Data Update" ~/ares-trading-test/logs/stdout.log | tail -5
```

Make it executable:
```bash
chmod +x ~/ares-trading-test/monitor.sh

# Run monitor
~/ares-trading-test/monitor.sh

# Or watch it continuously
watch -n 10 ~/ares-trading-test/monitor.sh
```

### 2. Performance Dashboard Script

```bash
nano ~/ares-trading-test/dashboard.py
```

```python
#!/usr/bin/env python3
import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict

def parse_logs(log_file):
    trades = []
    balances = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse trades
            if "Executing trade" in line:
                match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*symbol: (\w+).*side: (\w+).*quantity: ([\d.]+).*price: ([\d.]+)', line)
                if match:
                    trades.append({
                        'time': match.group(1),
                        'symbol': match.group(2),
                        'side': match.group(3),
                        'quantity': float(match.group(4)),
                        'price': float(match.group(5))
                    })
            
            # Parse balance updates
            if "Current Balance:" in line:
                match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*Current Balance: \$([\d.]+)', line)
                if match:
                    balances.append({
                        'time': match.group(1),
                        'balance': float(match.group(2))
                    })
    
    return trades, balances

def calculate_metrics(trades, balances):
    if not trades:
        return {}
    
    # Calculate trade statistics
    total_trades = len(trades)
    buy_trades = len([t for t in trades if t['side'].lower() == 'buy'])
    sell_trades = len([t for t in trades if t['side'].lower() == 'sell'])
    
    # Calculate P&L if we have balance history
    initial_balance = 100000.0
    current_balance = balances[-1]['balance'] if balances else initial_balance
    total_pnl = current_balance - initial_balance
    pnl_percent = (total_pnl / initial_balance) * 100
    
    # Symbol distribution
    symbol_counts = defaultdict(int)
    for trade in trades:
        symbol_counts[trade['symbol']] += 1
    
    return {
        'total_trades': total_trades,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'initial_balance': initial_balance,
        'current_balance': current_balance,
        'total_pnl': total_pnl,
        'pnl_percent': pnl_percent,
        'most_traded': max(symbol_counts.items(), key=lambda x: x[1]) if symbol_counts else ('N/A', 0)
    }

def print_dashboard(metrics):
    print("\n" + "="*60)
    print(" "*20 + "TRADING DASHBOARD")
    print("="*60)
    
    print(f"\nAccount Performance:")
    print(f"  Initial Balance:  ${metrics.get('initial_balance', 0):,.2f}")
    print(f"  Current Balance:  ${metrics.get('current_balance', 0):,.2f}")
    print(f"  Total P&L:        ${metrics.get('total_pnl', 0):+,.2f}")
    print(f"  Return:           {metrics.get('pnl_percent', 0):+.2f}%")
    
    print(f"\nTrading Activity:")
    print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
    print(f"  Buy Orders:       {metrics.get('buy_trades', 0)}")
    print(f"  Sell Orders:      {metrics.get('sell_trades', 0)}")
    
    symbol, count = metrics.get('most_traded', ('N/A', 0))
    print(f"  Most Traded:      {symbol} ({count} trades)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/home/diddy/ares-trading-test/logs/stdout.log"
    
    try:
        trades, balances = parse_logs(log_file)
        metrics = calculate_metrics(trades, balances)
        print_dashboard(metrics)
    except Exception as e:
        print(f"Error parsing logs: {e}")
```

Make it executable:
```bash
chmod +x ~/ares-trading-test/dashboard.py

# Run dashboard
python3 ~/ares-trading-test/dashboard.py
```

## Test Scenarios

### 1. 24-Hour Market Test (Weekday)
```bash
# Start Monday morning, run until Tuesday morning
# Best for testing with stock market data
systemctl start ares-trading
```

### 2. Weekend Crypto Test
```bash
# Configure for crypto (Binance) and run over weekend
# Crypto markets are 24/7
nano config.toml  # Set provider = "binance"
systemctl restart ares-trading
```

### 3. Stress Test (High Frequency)
```bash
# Edit config for aggressive trading
nano config.toml
```

Set aggressive parameters:
```toml
[trading]
max_open_positions = 50
stop_loss_percent = 1.0
take_profit_percent = 1.5

market_scan_interval_ms = 500  # Scan every 0.5 seconds
```

### 4. Conservative Test (Low Risk)
```toml
[trading]
max_position_size = 0.02  # 2% per position
stop_loss_percent = 0.5   # Very tight stop
max_open_positions = 5    # Few positions
```

## Collecting Test Results

### Daily Report Script
```bash
nano ~/ares-trading-test/daily_report.sh
```

```bash
#!/bin/bash

DATE=$(date +%Y-%m-%d)
REPORT_FILE="~/ares-trading-test/reports/report-$DATE.txt"

echo "ARES Trading Daily Report - $DATE" > $REPORT_FILE
echo "=================================" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# Extract key metrics
echo "Performance Metrics:" >> $REPORT_FILE
grep "Total Return" ~/ares-trading-test/logs/stdout.log | tail -1 >> $REPORT_FILE
grep "Win Rate" ~/ares-trading-test/logs/stdout.log | tail -1 >> $REPORT_FILE
grep "Sharpe Ratio" ~/ares-trading-test/logs/stdout.log | tail -1 >> $REPORT_FILE

echo "" >> $REPORT_FILE
echo "Trade Summary:" >> $REPORT_FILE
grep -c "Executing trade" ~/ares-trading-test/logs/stdout.log >> $REPORT_FILE

echo "" >> $REPORT_FILE
echo "Errors:" >> $REPORT_FILE
grep ERROR ~/ares-trading-test/logs/stderr.log | wc -l >> $REPORT_FILE

# Email report (optional)
# mail -s "Trading Report $DATE" your-email@example.com < $REPORT_FILE
```

### Add to crontab for daily reports:
```bash
crontab -e
# Add this line:
0 0 * * * /home/diddy/ares-trading-test/daily_report.sh
```

## Troubleshooting

### If the system stops trading:
```bash
# Check service status
systemctl status ares-trading

# Check logs for errors
tail -100 ~/ares-trading-test/logs/stderr.log

# Restart service
systemctl restart ares-trading
```

### If using Yahoo Finance during market hours:
- Trading only occurs during US market hours (9:30 AM - 4:00 PM EST)
- Use Binance provider for 24/7 testing

### Memory usage growing:
```bash
# Monitor memory
watch -n 10 'ps aux | grep paper-trader'

# Restart periodically (add to crontab)
0 */6 * * * systemctl restart ares-trading
```

## Expected Results

After 24 hours of testing, you should see:
- 50-200 trades executed (depending on market volatility)
- Portfolio performance between -5% to +5% (typical range)
- Detailed logs showing every decision
- Performance metrics calculated

After 1 week:
- 500+ trades
- Clear performance trends
- Identified profitable/unprofitable patterns
- System stability confirmed

## Next Steps

1. **Analyze Results**: Use the dashboard.py script to analyze performance
2. **Tune Parameters**: Adjust config based on results
3. **Add Symbols**: Test with different stocks/cryptocurrencies
4. **Upgrade Provider**: Get API keys for professional data if needed
5. **Implement Improvements**: Based on test results, enhance strategy

Remember: This is paper trading - no real money at risk!