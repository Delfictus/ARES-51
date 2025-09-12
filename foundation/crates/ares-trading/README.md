# ARES Paper Trading System

A sophisticated paper trading system that simulates stock market trading with real-world market data integration.

## Features

- **Paper Trading Simulation**: Practice trading with a virtual $100,000 portfolio
- **Real Market Data**: Integration with multiple market data providers
- **Technical Analysis**: RSI, SMA, and Momentum indicators
- **Risk Management**: Stop-loss, take-profit, and position sizing
- **Performance Analytics**: Track P&L, win rate, Sharpe ratio
- **Multiple Asset Classes**: Stocks, ETFs, and cryptocurrencies

## Market Data Providers

### Free Providers (No API Key Required)
- **Yahoo Finance**: Real-time stock quotes and historical data
- **Binance**: Cryptocurrency market data

### Professional Providers (API Key Required)
- **Polygon.io**: Professional stock data (5 calls/min free tier)
- **IEX Cloud**: Enterprise financial data (50,000 calls/month free)
- **Finnhub**: Real-time data with news sentiment (60 calls/min free)
- **Twelve Data**: Technical indicators included (800 calls/day free)

## Quick Start

### 1. Configuration

Edit `config.toml` to select your preferred market data provider:

```toml
[market_data]
provider = "yahoo_finance"  # Free, no API key needed

# For professional providers, add your API keys:
# polygon_api_key = "YOUR_KEY"
# iex_api_token = "YOUR_TOKEN"
# finnhub_api_key = "YOUR_KEY"
# twelve_data_api_key = "YOUR_KEY"
```

### 2. Run the Paper Trader

```bash
# From the monorepo root
cargo run -p ares-trading

# Or run directly
cd crates/ares-trading
cargo run --bin paper-trader
```

### 3. Monitor Performance

The system will display:
- Real-time market data updates every 10 seconds
- Portfolio status updates every 30 seconds
- Trade execution logs
- Performance metrics

## Trading Strategy

The system uses technical analysis with three main indicators:

1. **RSI (Relative Strength Index)**
   - Buy signal: RSI < 30 (oversold)
   - Sell signal: RSI > 70 (overbought)

2. **SMA Crossover**
   - Buy signal: Short SMA crosses above Long SMA
   - Sell signal: Short SMA crosses below Long SMA

3. **Momentum**
   - Confirms trend direction
   - Filters out false signals

## Configuration Options

### Trading Parameters
```toml
[trading]
initial_balance = 100000.0   # Starting capital
max_position_size = 0.1      # 10% of portfolio per position
stop_loss_percent = 2.0      # 2% stop loss
take_profit_percent = 5.0    # 5% take profit
max_open_positions = 10      # Maximum concurrent positions
```

### Risk Management
```toml
[risk_management]
max_daily_loss = 1000.0      # Maximum daily loss allowed
max_drawdown = 0.2           # 20% maximum drawdown
position_sizing = "kelly"     # Position sizing method
```

## API Key Setup

### Getting Free API Keys

1. **Polygon.io**: https://polygon.io/dashboard/signup
2. **IEX Cloud**: https://iexcloud.io/console/register
3. **Finnhub**: https://finnhub.io/register
4. **Twelve Data**: https://twelvedata.com/signup

### Configuring API Keys

Add your keys to `config.toml`:
```toml
[market_data]
provider = "polygon_io"
polygon_api_key = "YOUR_POLYGON_KEY"
```

## Architecture

```
ares-trading/
├── src/
│   ├── main.rs              # Application entry point
│   ├── account.rs           # Account management
│   ├── market_data.rs       # Market data interfaces
│   ├── trading_engine.rs    # Core trading logic
│   ├── orders.rs            # Order management
│   ├── portfolio.rs         # Portfolio tracking
│   ├── risk.rs              # Risk management
│   ├── strategies.rs        # Trading strategies
│   ├── analytics.rs         # Performance analytics
│   └── providers/           # Market data providers
│       ├── yahoo_finance.rs # Yahoo Finance (FREE)
│       ├── polygon_io.rs    # Polygon.io
│       ├── iex_cloud.rs     # IEX Cloud
│       ├── binance.rs       # Binance (FREE)
│       ├── finnhub.rs       # Finnhub
│       └── twelve_data.rs   # Twelve Data
├── config.toml              # Configuration file
└── README.md                # This file
```

## Performance Metrics

The system tracks:
- **Total P&L**: Realized and unrealized profit/loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Time held per position
- **Risk/Reward Ratio**: Average win vs average loss

## Safety Features

- **Paper Trading Only**: No real money at risk
- **Rate Limiting**: Respects API rate limits
- **Error Handling**: Graceful degradation on API failures
- **Stop Loss**: Automatic position exit on losses
- **Position Limits**: Maximum position size constraints

## Troubleshooting

### No Market Data
- Check your internet connection
- Verify API keys are correct
- Ensure market is open (for stock data)
- Check provider status pages

### High Latency
- Switch to a provider with better geographic proximity
- Use WebSocket connections when available
- Reduce the number of monitored symbols

### API Rate Limits
- Use free providers (Yahoo Finance, Binance)
- Implement caching for frequently accessed data
- Upgrade to paid API tiers for production use

## Future Enhancements

- [ ] WebSocket support for real-time data
- [ ] Advanced strategies (Options, Pairs trading)
- [ ] Machine learning integration
- [ ] Backtesting framework
- [ ] Web dashboard interface
- [ ] Multi-account support
- [ ] Social trading features

## License

Part of the ARES ChronoFabric system. See main repository for license details.

---
Author: Ididia Serfaty  
Contact: IS@delfictus.com