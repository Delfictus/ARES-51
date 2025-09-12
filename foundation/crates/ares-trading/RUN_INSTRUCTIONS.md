# How to Run ARES Paper Trading System

## QUICK START (1 Minute)

```bash
# From anywhere on your system:
cd /home/diddy/dev/ares-monorepo
./start-trading.sh
```

Then select option 1 (Yahoo Finance) or 2 (Crypto).

## MANUAL START OPTIONS

### Option 1: Basic Test Run (FREE - Yahoo Finance)
```bash
cd /home/diddy/dev/ares-monorepo
cargo run -p ares-trading
```
- Uses Yahoo Finance (no API key needed)
- Works during US market hours (9:30 AM - 4:00 PM EST)
- Monitors AAPL, GOOGL, MSFT, TSLA, AMZN

### Option 2: 24/7 Cryptocurrency Trading (FREE - Binance)
```bash
cd /home/diddy/dev/ares-monorepo/crates/ares-trading

# Edit config.toml and change:
# provider = "binance"

cargo run --bin paper-trader
```
- Uses Binance (no API key needed)
- Works 24/7
- Trades BTC, ETH, BNB

### Option 3: Long Running Test (Production-like)
```bash
# Use the startup script
./start-trading.sh
# Select option 3

# Or manually:
cargo build --release -p ares-trading
./target/release/paper-trader
```

## MONITORING YOUR TEST

### Real-time Monitoring
```bash
# In a new terminal, watch the logs:
tail -f ~/ares-trading-test/logs/stdout.log

# Or use the startup script:
./start-trading.sh
# Select option 5 (Check Status)
```

### What You'll See
```
2024-01-15T10:30:00 INFO Starting ARES Paper Trading System
2024-01-15T10:30:01 INFO Configuration loaded:
  Provider: yahoo_finance
  Initial Balance: $100000
  Max Positions: 10
2024-01-15T10:30:02 INFO Subscribing to 5 symbols: ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
2024-01-15T10:30:03 INFO Market Data Update:
AAPL: $185.92 (+$1.35 +0.73%)
GOOGL: $139.65 (-$0.48 -0.34%)
MSFT: $388.47 (+$2.11 +0.55%)
2024-01-15T10:30:15 INFO Executing trade: BUY 50 shares of AAPL at $185.92
2024-01-15T10:30:30 INFO Portfolio Status Update:
  Current Balance: $90,704.00
  Open Positions: 1
  Unrealized P&L: +$12.50
```

## RUNNING A LONG TEST

### 24-Hour Test
```bash
# Start in screen/tmux for persistent session
screen -S trading
cd /home/diddy/dev/ares-monorepo
cargo run -p ares-trading

# Detach: Ctrl+A, D
# Reattach: screen -r trading
```

### Week-Long Test
```bash
# Use systemd service (created by startup script option 3)
./start-trading.sh
# Select option 3

# Check status anytime:
systemctl status ares-trading

# View performance:
./start-trading.sh
# Select option 7
```

## TEST CONFIGURATIONS

### Conservative (Low Risk)
Edit `crates/ares-trading/config.toml`:
```toml
max_position_size = 0.02   # 2% per trade
stop_loss_percent = 1.0     # 1% stop loss
max_open_positions = 5      # Max 5 positions
```

### Aggressive (High Activity)
```toml
max_position_size = 0.15   # 15% per trade
stop_loss_percent = 3.0     # 3% stop loss
max_open_positions = 20     # Max 20 positions
```

### Crypto Focus
```toml
provider = "binance"
[[symbols]]
symbol = "BTC-USD"
[[symbols]]
symbol = "ETH-USD"
```

## EXPECTED BEHAVIOR

### During Market Hours (Stocks)
- Scans market every 2 seconds
- Analyzes RSI, SMA, Momentum indicators
- Executes trades when signals align
- Applies stop-loss and take-profit automatically

### After Hours / Weekends
- Stock providers will show "market closed"
- Use Binance provider for 24/7 crypto trading
- System continues running but waits for market open

### Performance Expectations
- **First Hour**: 0-5 trades (system warming up)
- **First Day**: 10-50 trades (depending on volatility)
- **First Week**: 100-500 trades
- **Typical Returns**: -2% to +2% daily (paper trading)

## TROUBLESHOOTING

### "Market Closed" Message
- Normal for stocks outside 9:30 AM - 4:00 PM EST
- Switch to Binance for 24/7 trading

### No Trades Executing
- Check market volatility (low volatility = fewer signals)
- Verify symbols are correct
- Check logs for errors: `grep ERROR ~/ares-trading-test/logs/stderr.log`

### High CPU Usage
- Normal during market hours (analyzing data)
- Reduce symbol count if needed
- Increase scan interval in config

### Memory Growth
- Restart every 24-48 hours for long tests
- Use systemd service with auto-restart

## STOPPING THE SYSTEM

### Graceful Shutdown
```bash
# If running in terminal:
Ctrl+C

# If using startup script:
./start-trading.sh
# Select option 6

# If using systemd:
sudo systemctl stop ares-trading
```

## ANALYZING RESULTS

### View Summary
```bash
./start-trading.sh
# Select option 7 (View Performance)
```

### Export Logs
```bash
# Copy logs for analysis
cp ~/ares-trading-test/logs/*.log ~/my-analysis-folder/
```

### Key Metrics to Track
- Total P&L (profit/loss)
- Win Rate (profitable trades / total trades)
- Average Trade Duration
- Maximum Drawdown
- Sharpe Ratio (risk-adjusted returns)

## NEXT STEPS

1. **Run 24-hour test** with default settings
2. **Analyze results** using performance dashboard
3. **Adjust parameters** based on performance
4. **Run week-long test** with optimized settings
5. **Consider API upgrades** for better data:
   - Polygon.io for professional stock data
   - Finnhub for news sentiment
   - TwelveData for technical indicators

## SUPPORT

- Logs: `~/ares-trading-test/logs/`
- Config: `crates/ares-trading/config.toml`
- Status: `./start-trading.sh` → Option 5

Remember: This is PAPER TRADING - no real money involved!