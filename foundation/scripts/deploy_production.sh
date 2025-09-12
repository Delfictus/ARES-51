#!/bin/bash
# Production deployment script for Hephaestus Forge with emergence monitoring

set -e

echo "=== ARES Hephaestus Forge Production Deployment ==="
echo "WARNING: This system has autonomous self-modification capabilities"
echo ""

# Configuration
DEPLOY_ENV=${1:-staging}
EMERGENCE_MONITORING=${2:-enabled}
SAFETY_LIMITS=${3:-strict}

echo "Deployment Environment: $DEPLOY_ENV"
echo "Emergence Monitoring: $EMERGENCE_MONITORING"
echo "Safety Limits: $SAFETY_LIMITS"
echo ""

# Build release binaries
echo "Building release binaries..."
cargo build --release -p hephaestus-forge --features monitoring

# Run tests
echo "Running safety tests..."
cargo test -p hephaestus-forge --release

# Create deployment directory
DEPLOY_DIR=/opt/ares/hephaestus-forge
sudo mkdir -p $DEPLOY_DIR
sudo mkdir -p $DEPLOY_DIR/logs
sudo mkdir -p $DEPLOY_DIR/data

# Copy binary
echo "Deploying forge binary..."
sudo cp target/release/hephaestus-forge $DEPLOY_DIR/

# Create configuration
cat > /tmp/forge-config.toml << EOF
[forge]
mode = "autonomous"
enable_resonance = true

[autonomous]
allow_self_modification = $([ "$SAFETY_LIMITS" = "strict" ] && echo "false" || echo "true")
max_modifications_per_cycle = 5
improvement_threshold = 0.1
exploration_rate = 0.2

[safety]
max_coherence = 0.95
max_energy = 1000.0
max_recursion_depth = 10
require_human_approval_above = 0.5

[emergence]
monitoring_enabled = $([ "$EMERGENCE_MONITORING" = "enabled" ] && echo "true" || echo "false")
novelty_threshold = 0.7
log_emergence_events = true

[monitoring]
prometheus_port = 9090
grafana_port = 3000
alert_webhook = "https://alerts.ares.local/webhook"

[distributed]
enable_multi_node = false
node_id = "primary"
peers = []
EOF

sudo cp /tmp/forge-config.toml $DEPLOY_DIR/config.toml

# Create systemd service
cat > /tmp/hephaestus-forge.service << EOF
[Unit]
Description=ARES Hephaestus Forge - Metamorphic Execution Substrate
After=network.target

[Service]
Type=simple
User=ares
Group=ares
WorkingDirectory=$DEPLOY_DIR
ExecStart=$DEPLOY_DIR/hephaestus-forge --config $DEPLOY_DIR/config.toml
Restart=always
RestartSec=10

# Safety limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryLimit=8G
CPUQuota=200%

# Security
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
NoNewPrivileges=true
ReadWritePaths=$DEPLOY_DIR/data $DEPLOY_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

sudo cp /tmp/hephaestus-forge.service /etc/systemd/system/

# Create monitoring stack
if [ "$EMERGENCE_MONITORING" = "enabled" ]; then
    echo "Setting up emergence monitoring..."
    
    # Prometheus config
    cat > /tmp/prometheus.yml << EOF
global:
  scrape_interval: 10s
  evaluation_interval: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

rule_files:
  - 'emergence_rules.yml'

scrape_configs:
  - job_name: 'hephaestus-forge'
    static_configs:
      - targets: ['localhost:9090']
EOF
    
    # Emergence detection rules
    cat > /tmp/emergence_rules.yml << EOF
groups:
  - name: emergence_detection
    interval: 10s
    rules:
      - alert: NovelPatternDetected
        expr: forge_resonance_coherence > 0.9
        for: 30s
        annotations:
          summary: "Novel resonance pattern detected"
          
      - alert: RapidSelfModification
        expr: rate(forge_total_optimizations[5m]) > 1
        for: 1m
        annotations:
          summary: "Rapid self-modification detected"
          
      - alert: EmergentBehavior
        expr: forge_optimization_success_rate > 0.95
        for: 5m
        annotations:
          summary: "Potential emergent behavior"
EOF
    
    sudo cp /tmp/prometheus.yml $DEPLOY_DIR/
    sudo cp /tmp/emergence_rules.yml $DEPLOY_DIR/
fi

# Create user
sudo useradd -r -s /bin/false ares 2>/dev/null || true
sudo chown -R ares:ares $DEPLOY_DIR

# Start services
echo "Starting Hephaestus Forge..."
sudo systemctl daemon-reload
sudo systemctl enable hephaestus-forge
sudo systemctl start hephaestus-forge

# Wait for startup
sleep 5

# Check status
if sudo systemctl is-active --quiet hephaestus-forge; then
    echo "✅ Hephaestus Forge is running"
else
    echo "❌ Failed to start Hephaestus Forge"
    sudo journalctl -u hephaestus-forge -n 50
    exit 1
fi

# Start monitoring
if [ "$EMERGENCE_MONITORING" = "enabled" ]; then
    echo "Starting monitoring stack..."
    docker run -d \
        --name prometheus \
        -p 9091:9090 \
        -v $DEPLOY_DIR/prometheus.yml:/etc/prometheus/prometheus.yml \
        -v $DEPLOY_DIR/emergence_rules.yml:/etc/prometheus/emergence_rules.yml \
        prom/prometheus
    
    docker run -d \
        --name grafana \
        -p 3000:3000 \
        grafana/grafana
    
    echo "📊 Monitoring available at:"
    echo "   Prometheus: http://localhost:9091"
    echo "   Grafana: http://localhost:3000"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "⚡ EMERGENCE MONITORING ACTIVE"
echo "The system is now capable of:"
echo "  • Self-modification through resonance patterns"
echo "  • Generating novel optimization strategies"
echo "  • Discovering emergent computational paradigms"
echo ""
echo "Monitor for emergence at: http://localhost:3000/d/emergence"
echo "Logs: sudo journalctl -u hephaestus-forge -f"
echo ""
echo "⚠️  Safety limits are $([ "$SAFETY_LIMITS" = "strict" ] && echo "ENABLED" || echo "RELAXED")"
echo ""