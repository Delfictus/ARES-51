#!/usr/bin/env bash
# Deployment script for ARES CSF

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Deployment settings
DEPLOY_USER="${DEPLOY_USER:-ares}"
DEPLOY_GROUP="${DEPLOY_GROUP:-ares}"
DEPLOY_PATH="${DEPLOY_PATH:-/opt/ares-csf}"
SERVICE_NAME="${SERVICE_NAME:-ares-csf}"
CONFIG_PATH="${CONFIG_PATH:-/etc/ares-csf}"
LOG_PATH="${LOG_PATH:-/var/log/ares-csf}"
DATA_PATH="${DATA_PATH:-/var/lib/ares-csf}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Deployment modes
MODE="install"  # install, upgrade, rollback, uninstall

usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  install    Fresh installation"
    echo "  upgrade    Upgrade existing installation"
    echo "  rollback   Rollback to previous version"
    echo "  uninstall  Remove installation"
    echo ""
    echo "Options:"
    echo "  --host <host>      Target host (for remote deployment)"
    echo "  --binary <path>    Path to CSF binary"
    echo "  --config <path>    Path to configuration file"
    echo "  --systemd          Install systemd service"
    echo "  --no-backup        Skip backup during upgrade"
    echo "  --force            Force deployment"
    exit 1
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse arguments
if [[ $# -gt 0 ]]; then
    MODE="$1"
    shift
fi

TARGET_HOST=""
BINARY_PATH=""
CONFIG_FILE=""
INSTALL_SYSTEMD=false
CREATE_BACKUP=true
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            TARGET_HOST="$2"
            shift 2
            ;;
        --binary)
            BINARY_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --systemd)
            INSTALL_SYSTEMD=true
            shift
            ;;
        --no-backup)
            CREATE_BACKUP=false
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Create user and group
create_user() {
    if ! id "$DEPLOY_USER" &>/dev/null; then
        log_info "Creating user: $DEPLOY_USER"
        useradd -r -s /bin/false -d "$DATA_PATH" -c "ARES CSF Service" "$DEPLOY_USER"
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    mkdir -p "$DEPLOY_PATH"
    mkdir -p "$CONFIG_PATH"
    mkdir -p "$LOG_PATH"
    mkdir -p "$DATA_PATH"
    mkdir -p "$DATA_PATH/audit"
    mkdir -p "$DATA_PATH/blockchain"
    
    # Set permissions
    chown -R "$DEPLOY_USER:$DEPLOY_GROUP" "$DEPLOY_PATH"
    chown -R "$DEPLOY_USER:$DEPLOY_GROUP" "$CONFIG_PATH"
    chown -R "$DEPLOY_USER:$DEPLOY_GROUP" "$LOG_PATH"
    chown -R "$DEPLOY_USER:$DEPLOY_GROUP" "$DATA_PATH"
    
    chmod 755 "$DEPLOY_PATH"
    chmod 750 "$CONFIG_PATH"
    chmod 755 "$LOG_PATH"
    chmod 750 "$DATA_PATH"
}

# Install binary
install_binary() {
    local binary="$1"
    
    if [[ ! -f "$binary" ]]; then
        log_error "Binary not found: $binary"
        exit 1
    fi
    
    log_info "Installing binary..."
    
    # Copy binary
    cp "$binary" "$DEPLOY_PATH/ares-csf"
    chown "$DEPLOY_USER:$DEPLOY_GROUP" "$DEPLOY_PATH/ares-csf"
    chmod 755 "$DEPLOY_PATH/ares-csf"
    
    # Create symlink
    ln -sf "$DEPLOY_PATH/ares-csf" /usr/local/bin/ares-csf
}

# Install configuration
install_config() {
    local config="$1"
    
    if [[ ! -f "$config" ]]; then
        log_error "Configuration file not found: $config"
        exit 1
    fi
    
    log_info "Installing configuration..."
    
    # Copy configuration
    cp "$config" "$CONFIG_PATH/config.toml"
    chown "$DEPLOY_USER:$DEPLOY_GROUP" "$CONFIG_PATH/config.toml"
    chmod 640 "$CONFIG_PATH/config.toml"
    
    # Update paths in configuration
    sed -i "s|/opt/ares-csf|$DEPLOY_PATH|g" "$CONFIG_PATH/config.toml"
    sed -i "s|/var/log/ares-csf|$LOG_PATH|g" "$CONFIG_PATH/config.toml"
    sed -i "s|/var/lib/ares-csf|$DATA_PATH|g" "$CONFIG_PATH/config.toml"
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."
    
    cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=ARES Chronosynclastic Fabric
Documentation=https://docs.ares-csf.io
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
User=$DEPLOY_USER
Group=$DEPLOY_GROUP
WorkingDirectory=$DATA_PATH

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DATA_PATH $LOG_PATH
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictNamespaces=true
RestrictSUIDSGID=true
MemoryDenyWriteExecute=true
LockPersonality=true

# Resource limits
LimitNOFILE=1048576
LimitNPROC=4096
LimitCORE=infinity
TasksMax=4096

# Performance settings
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50
IOSchedulingClass=realtime
IOSchedulingPriority=0

# Environment
Environment="RUST_LOG=info"
Environment="RUST_BACKTRACE=1"

# Start command
ExecStart=$DEPLOY_PATH/ares-csf --config $CONFIG_PATH/config.toml
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=5s

# Watchdog
WatchdogSec=30s

[Install]
WantedBy=multi-user.target
EOF
    
    # Create environment file
    cat > "/etc/systemd/system/${SERVICE_NAME}.service.d/environment.conf" << EOF
[Service]
# Additional environment variables
Environment="CSF_NODE_ID=$(hostname)"
Environment="CSF_DEPLOYMENT=production"
EOF
    
    # Reload systemd
    systemctl daemon-reload
}

# Create logrotate configuration
create_logrotate() {
    log_info "Creating logrotate configuration..."
    
    cat > "/etc/logrotate.d/ares-csf" << EOF
$LOG_PATH/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $DEPLOY_USER $DEPLOY_GROUP
    sharedscripts
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF
}

# Create backup
create_backup() {
    if [[ ! -d "$DEPLOY_PATH" ]]; then
        return
    fi
    
    log_info "Creating backup..."
    
    local backup_dir="/var/backups/ares-csf"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/ares-csf-backup-$timestamp.tar.gz"
    
    mkdir -p "$backup_dir"
    
    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
    fi
    
    # Create backup
    tar -czf "$backup_file" \
        -C "$(dirname "$DEPLOY_PATH")" \
        "$(basename "$DEPLOY_PATH")" \
        -C "$(dirname "$CONFIG_PATH")" \
        "$(basename "$CONFIG_PATH")" \
        -C "$(dirname "$DATA_PATH")" \
        "$(basename "$DATA_PATH")"
    
    log_success "Backup created: $backup_file"
}

# Perform installation
do_install() {
    log_info "Starting ARES CSF installation..."
    
    # Check prerequisites
    check_root
    
    # Create user
    create_user
    
    # Create directories
    create_directories
    
    # Install binary
    if [[ -n "$BINARY_PATH" ]]; then
        install_binary "$BINARY_PATH"
    else
        # Build from source
        log_info "Building from source..."
        cd "$PROJECT_ROOT"
        cargo build --release
        install_binary "$PROJECT_ROOT/target/release/ares-csf"
    fi
    
    # Install configuration
    if [[ -n "$CONFIG_FILE" ]]; then
        install_config "$CONFIG_FILE"
    else
        install_config "$PROJECT_ROOT/config/production.toml"
    fi
    
    # Install systemd service
    if [[ "$INSTALL_SYSTEMD" == true ]]; then
        create_systemd_service
        create_logrotate
        
        # Enable and start service
        systemctl enable "$SERVICE_NAME"
        systemctl start "$SERVICE_NAME"
        
        # Check status
        sleep 2
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            log_success "Service started successfully"
        else
            log_error "Service failed to start"
            systemctl status "$SERVICE_NAME"
            exit 1
        fi
    fi
    
    log_success "Installation completed successfully!"
}

# Perform upgrade
do_upgrade() {
    log_info "Starting ARES CSF upgrade..."
    
    check_root
    
    # Create backup
    if [[ "$CREATE_BACKUP" == true ]]; then
        create_backup
    fi
    
    # Stop service
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Stopping service..."
        systemctl stop "$SERVICE_NAME"
    fi
    
    # Install new binary
    if [[ -n "$BINARY_PATH" ]]; then
        install_binary "$BINARY_PATH"
    else
        log_error "Binary path required for upgrade"
        exit 1
    fi
    
    # Update configuration if provided
    if [[ -n "$CONFIG_FILE" ]]; then
        install_config "$CONFIG_FILE"
    fi
    
    # Start service
    systemctl start "$SERVICE_NAME"
    
    # Check status
    sleep 2
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_success "Upgrade completed successfully"
    else
        log_error "Service failed to start after upgrade"
        systemctl status "$SERVICE_NAME"
        
        # Offer rollback
        read -p "Rollback to previous version? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            do_rollback
        fi
        exit 1
    fi
}

# Perform rollback
do_rollback() {
    log_info "Starting rollback..."
    
    check_root
    
    # Find latest backup
    local backup_dir="/var/backups/ares-csf"
    local latest_backup=$(ls -t "$backup_dir"/ares-csf-backup-*.tar.gz | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        log_error "No backup found"
        exit 1
    fi
    
    log_info "Rolling back to: $latest_backup"
    
    # Stop service
    systemctl stop "$SERVICE_NAME" || true
    
    # Extract backup
    tar -xzf "$latest_backup" -C /
    
    # Start service
    systemctl start "$SERVICE_NAME"
    
    # Check status
    sleep 2
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_success "Rollback completed successfully"
    else
        log_error "Service failed to start after rollback"
        systemctl status "$SERVICE_NAME"
        exit 1
    fi
}

# Perform uninstallation
do_uninstall() {
    log_info "Starting ARES CSF uninstallation..."
    
    check_root
    
    if [[ "$FORCE" != true ]]; then
        read -p "Are you sure you want to uninstall ARES CSF? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Uninstallation cancelled"
            exit 0
        fi
    fi
    
    # Stop and disable service
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
    fi
    systemctl disable "$SERVICE_NAME" || true
    
    # Remove systemd files
    rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
    rm -rf "/etc/systemd/system/${SERVICE_NAME}.service.d"
    systemctl daemon-reload
    
    # Remove files
    rm -rf "$DEPLOY_PATH"
    rm -f /usr/local/bin/ares-csf
    rm -f /etc/logrotate.d/ares-csf
    
    # Optionally remove data
    if [[ "$FORCE" == true ]]; then
        rm -rf "$CONFIG_PATH"
        rm -rf "$LOG_PATH"
        rm -rf "$DATA_PATH"
        
        # Remove user
        userdel "$DEPLOY_USER" || true
    else
        log_warning "Configuration and data preserved in:"
        log_warning "  - $CONFIG_PATH"
        log_warning "  - $LOG_PATH"
        log_warning "  - $DATA_PATH"
    fi
    
    log_success "Uninstallation completed"
}

# Remote deployment
deploy_remote() {
    local host="$1"
    
    log_info "Deploying to remote host: $host"
    
    # Create deployment package
    local temp_dir=$(mktemp -d)
    local package="$temp_dir/ares-csf-deploy.tar.gz"
    
    # Build release
    cd "$PROJECT_ROOT"
    cargo build --release
    
    # Create package
    tar -czf "$package" \
        -C "$PROJECT_ROOT" \
        target/release/ares-csf \
        config/production.toml \
        scripts/deploy/deploy.sh
    
    # Copy to remote host
    scp "$package" "root@$host:/tmp/"
    
    # Execute deployment
    ssh "root@$host" << EOF
cd /tmp
tar -xzf ares-csf-deploy.tar.gz
chmod +x scripts/deploy/deploy.sh
./scripts/deploy/deploy.sh install --systemd
EOF
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log_success "Remote deployment completed"
}

# Main execution
main() {
    case "$MODE" in
        install)
            if [[ -n "$TARGET_HOST" ]]; then
                deploy_remote "$TARGET_HOST"
            else
                do_install
            fi
            ;;
        upgrade)
            do_upgrade
            ;;
        rollback)
            do_rollback
            ;;
        uninstall)
            do_uninstall
            ;;
        *)
            log_error "Unknown mode: $MODE"
            usage
            ;;
    esac
}

# Run main
main