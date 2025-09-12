#!/bin/bash

# ARES Paper Trading Startup Script
# Author: Ididia Serfaty

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_DIR="/home/diddy/dev/ares-monorepo"
TRADING_DIR="/home/diddy/ares-trading-test"
LOG_DIR="$TRADING_DIR/logs"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     ARES Paper Trading System          ║${NC}"
echo -e "${BLUE}║     Virtual Trading Simulation         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Function to display menu
show_menu() {
    echo -e "${GREEN}Select Trading Mode:${NC}"
    echo "1) Quick Test - Yahoo Finance (FREE, stocks only during market hours)"
    echo "2) Crypto Trading - Binance (FREE, 24/7 cryptocurrency)"
    echo "3) Long Running Test - Systemd Service"
    echo "4) Development Mode - With detailed logging"
    echo "5) Check Status - View current trading status"
    echo "6) Stop Trading - Stop all trading processes"
    echo "7) View Performance - Show trading dashboard"
    echo "8) Exit"
    echo ""
    read -p "Enter choice [1-8]: " choice
}

# Setup test directory if it doesn't exist
setup_test_dir() {
    if [ ! -d "$TRADING_DIR" ]; then
        echo -e "${YELLOW}Setting up test directory...${NC}"
        mkdir -p "$TRADING_DIR"
        mkdir -p "$LOG_DIR"
        cp "$REPO_DIR/crates/ares-trading/config.toml" "$TRADING_DIR/"
        echo -e "${GREEN}Test directory created at $TRADING_DIR${NC}"
    fi
}

# Start Yahoo Finance trading
start_yahoo() {
    echo -e "${GREEN}Starting Paper Trading with Yahoo Finance...${NC}"
    echo -e "${YELLOW}Note: Stock trading only works during US market hours (9:30 AM - 4:00 PM EST)${NC}"
    echo ""
    
    cd "$REPO_DIR"
    RUST_LOG=ares_trading=info cargo run -p ares-trading 2>&1 | tee "$LOG_DIR/trading-$(date +%Y%m%d-%H%M%S).log"
}

# Start Binance crypto trading
start_crypto() {
    echo -e "${GREEN}Starting Crypto Paper Trading with Binance...${NC}"
    echo -e "${YELLOW}Configuring for cryptocurrency trading (24/7)...${NC}"
    
    # Update config for crypto
    cat > "$TRADING_DIR/config-crypto.toml" << 'EOF'
[trading]
initial_balance = 100000.0
max_position_size = 0.1
stop_loss_percent = 3.0
take_profit_percent = 5.0
max_open_positions = 10

[market_data]
provider = "binance"
rate_limit_per_minute = 60
enable_websocket = false

[[symbols]]
symbol = "BTC-USD"
asset_type = "crypto"

[[symbols]]
symbol = "ETH-USD"
asset_type = "crypto"

[[symbols]]
symbol = "BNB-USD"
asset_type = "crypto"

[[symbols]]
symbol = "SOL-USD"
asset_type = "crypto"

[[symbols]]
symbol = "ADA-USD"
asset_type = "crypto"

[logging]
level = "info"
EOF
    
    cd "$REPO_DIR"
    cp "$TRADING_DIR/config-crypto.toml" "$REPO_DIR/crates/ares-trading/config.toml"
    RUST_LOG=ares_trading=info cargo run -p ares-trading 2>&1 | tee "$LOG_DIR/crypto-$(date +%Y%m%d-%H%M%S).log"
}

# Start long running test
start_long_running() {
    echo -e "${GREEN}Setting up Long Running Test...${NC}"
    
    # Build release version
    echo -e "${YELLOW}Building optimized release version...${NC}"
    cd "$REPO_DIR"
    cargo build --release -p ares-trading
    
    # Create systemd service
    echo -e "${YELLOW}Creating systemd service...${NC}"
    sudo tee /etc/systemd/system/ares-trading.service > /dev/null << EOF
[Unit]
Description=ARES Paper Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$TRADING_DIR
ExecStart=$REPO_DIR/target/release/paper-trader
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/stdout.log
StandardError=append:$LOG_DIR/stderr.log
Environment="RUST_LOG=ares_trading=info"

[Install]
WantedBy=multi-user.target
EOF
    
    # Copy config
    cp "$REPO_DIR/crates/ares-trading/config.toml" "$TRADING_DIR/"
    
    # Start service
    sudo systemctl daemon-reload
    sudo systemctl enable ares-trading
    sudo systemctl restart ares-trading
    
    echo -e "${GREEN}Long running test started as systemd service${NC}"
    echo "Commands:"
    echo "  View status:  systemctl status ares-trading"
    echo "  View logs:    tail -f $LOG_DIR/stdout.log"
    echo "  Stop:         sudo systemctl stop ares-trading"
}

# Development mode with extra logging
start_dev_mode() {
    echo -e "${GREEN}Starting in Development Mode...${NC}"
    echo -e "${YELLOW}Full debug logging enabled${NC}"
    
    cd "$REPO_DIR"
    RUST_LOG=debug RUST_BACKTRACE=1 cargo run -p ares-trading 2>&1 | tee "$LOG_DIR/dev-$(date +%Y%m%d-%H%M%S).log"
}

# Check status
check_status() {
    echo -e "${BLUE}=== Trading System Status ===${NC}"
    echo ""
    
    # Check if systemd service is running
    if systemctl is-active --quiet ares-trading 2>/dev/null; then
        echo -e "Systemd Service: ${GREEN}RUNNING${NC}"
        systemctl status ares-trading --no-pager | head -10
    else
        echo -e "Systemd Service: ${RED}NOT RUNNING${NC}"
    fi
    
    echo ""
    
    # Check for running processes
    if pgrep -f "paper-trader" > /dev/null; then
        echo -e "Paper Trader Process: ${GREEN}RUNNING${NC}"
        ps aux | grep paper-trader | grep -v grep
    else
        echo -e "Paper Trader Process: ${RED}NOT RUNNING${NC}"
    fi
    
    echo ""
    
    # Show recent logs
    if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR)" ]; then
        echo -e "${BLUE}Recent Activity:${NC}"
        echo "Latest log file: $(ls -t $LOG_DIR/*.log 2>/dev/null | head -1)"
        
        latest_log=$(ls -t $LOG_DIR/*.log 2>/dev/null | head -1)
        if [ -f "$latest_log" ]; then
            echo ""
            echo "Last 5 trades:"
            grep "Executing trade" "$latest_log" 2>/dev/null | tail -5 || echo "No trades found"
            
            echo ""
            echo "Latest balance:"
            grep "Current Balance" "$latest_log" 2>/dev/null | tail -1 || echo "No balance info found"
        fi
    fi
}

# Stop all trading
stop_trading() {
    echo -e "${YELLOW}Stopping all trading processes...${NC}"
    
    # Stop systemd service
    if systemctl is-active --quiet ares-trading 2>/dev/null; then
        sudo systemctl stop ares-trading
        echo -e "${GREEN}Stopped systemd service${NC}"
    fi
    
    # Kill any remaining processes
    if pgrep -f "paper-trader" > /dev/null; then
        pkill -f "paper-trader"
        echo -e "${GREEN}Stopped paper-trader processes${NC}"
    fi
    
    echo -e "${GREEN}All trading processes stopped${NC}"
}

# View performance dashboard
view_performance() {
    echo -e "${BLUE}=== Trading Performance Dashboard ===${NC}"
    echo ""
    
    if [ ! -d "$LOG_DIR" ] || [ -z "$(ls -A $LOG_DIR 2>/dev/null)" ]; then
        echo -e "${RED}No log files found. Start trading first.${NC}"
        return
    fi
    
    latest_log=$(ls -t $LOG_DIR/*.log 2>/dev/null | head -1)
    
    if [ -f "$latest_log" ]; then
        echo "Analyzing: $latest_log"
        echo ""
        
        # Basic metrics
        total_trades=$(grep -c "Executing trade" "$latest_log" 2>/dev/null || echo 0)
        buy_orders=$(grep "Executing trade" "$latest_log" 2>/dev/null | grep -c "BUY" || echo 0)
        sell_orders=$(grep "Executing trade" "$latest_log" 2>/dev/null | grep -c "SELL" || echo 0)
        
        echo "Trade Statistics:"
        echo "  Total Trades: $total_trades"
        echo "  Buy Orders:   $buy_orders"
        echo "  Sell Orders:  $sell_orders"
        echo ""
        
        # Latest portfolio status
        echo "Latest Portfolio Status:"
        grep "Portfolio Status" "$latest_log" -A 10 2>/dev/null | tail -11 || echo "No portfolio status found"
        
        echo ""
        echo "Recent Errors:"
        grep ERROR "$latest_log" 2>/dev/null | tail -5 || echo "No errors found"
    else
        echo -e "${RED}No log files found${NC}"
    fi
}

# Main execution
setup_test_dir

while true; do
    show_menu
    
    case $choice in
        1)
            start_yahoo
            break
            ;;
        2)
            start_crypto
            break
            ;;
        3)
            start_long_running
            break
            ;;
        4)
            start_dev_mode
            break
            ;;
        5)
            check_status
            echo ""
            read -p "Press Enter to continue..."
            ;;
        6)
            stop_trading
            echo ""
            read -p "Press Enter to continue..."
            ;;
        7)
            view_performance
            echo ""
            read -p "Press Enter to continue..."
            ;;
        8)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
done