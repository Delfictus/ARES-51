#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MONITORING_DIR="$REPO_ROOT/deployments/monitoring"

NAMESPACE="ares-monitoring"
RELEASE_NAME="ares-monitoring"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error "helm is required but not installed"
    
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"
    
    log "Prerequisites check passed"
}

create_namespace() {
    log "Creating monitoring namespace..."
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite
    
    log "Namespace $NAMESPACE ready"
}

add_helm_repositories() {
    log "Adding Helm repositories..."
    
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add datadog https://helm.datadoghq.com
    helm repo update
    
    log "Helm repositories updated"
}

generate_monitoring_secrets() {
    log "Generating monitoring secrets..."
    
    # Create secrets for monitoring stack
    kubectl create secret generic prometheus-config \
        --from-file="$MONITORING_DIR/prometheus-config.yaml" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic alert-rules \
        --from-file="$MONITORING_DIR/alert-rules.yml" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Datadog API key (if provided)
    if [ -n "${DATADOG_API_KEY:-}" ]; then
        kubectl create secret generic datadog-secrets \
            --from-literal=api-key="$DATADOG_API_KEY" \
            --from-literal=app-key="${DATADOG_APP_KEY:-}" \
            --namespace="$NAMESPACE" \
            --dry-run=client -o yaml | kubectl apply -f -
        log "Datadog secrets configured"
    fi
    
    # Slack webhook for alerting
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        kubectl create secret generic alerting-secrets \
            --from-literal=slack-webhook="$SLACK_WEBHOOK_URL" \
            --from-literal=pagerduty-key="${PAGERDUTY_INTEGRATION_KEY:-}" \
            --namespace="$NAMESPACE" \
            --dry-run=client -o yaml | kubectl apply -f -
        log "Alerting secrets configured"
    fi
    
    log "Monitoring secrets created"
}

deploy_prometheus_stack() {
    log "Deploying Prometheus monitoring stack..."
    
    # Create Prometheus values file
    cat > /tmp/prometheus-values.yaml << EOF
prometheus:
  prometheusSpec:
    configMaps:
      - prometheus-config
    secrets:
      - alert-rules
    retention: 30d
    retentionSize: 100GB
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 150Gi
    resources:
      requests:
        memory: 4Gi
        cpu: 2
      limits:
        memory: 8Gi
        cpu: 4
    nodeSelector:
      node-type: monitoring
    tolerations:
      - key: "monitoring-node"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"

alertmanager:
  alertmanagerSpec:
    configSecret: alerting-secrets
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi
    resources:
      requests:
        memory: 512Mi
        cpu: 100m
      limits:
        memory: 1Gi
        cpu: 500m

grafana:
  adminPassword: $(openssl rand -base64 32)
  persistence:
    enabled: true
    storageClassName: fast-ssd
    size: 20Gi
  resources:
    requests:
      memory: 1Gi
      cpu: 500m
    limits:
      memory: 2Gi
      cpu: 1
  nodeSelector:
    node-type: monitoring
  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
        - name: Prometheus
          type: prometheus
          url: http://prometheus-operated:9090
          access: proxy
          isDefault: true
        - name: Alertmanager
          type: alertmanager
          url: http://alertmanager-operated:9093
          access: proxy

nodeExporter:
  enabled: true
  hostNetwork: true
  hostPID: true
  
kubeStateMetrics:
  enabled: true

defaultRules:
  create: true
  rules:
    alertmanager: true
    etcd: true
    configReloaders: true
    general: true
    k8s: true
    kubeApiserver: true
    kubeApiserverAvailability: true
    kubeApiserverSlos: true
    kubelet: true
    kubeProxy: true
    kubePrometheusGeneral: true
    kubePrometheusNodeRecording: true
    kubernetesApps: true
    kubernetesResources: true
    kubernetesStorage: true
    kubernetesSystem: true
    network: true
    node: true
    nodeExporterAlerting: true
    nodeExporterRecording: true
    prometheus: true
    prometheusOperator: true
EOF

    helm upgrade --install "$RELEASE_NAME-prometheus" prometheus-community/kube-prometheus-stack \
        --namespace="$NAMESPACE" \
        --values /tmp/prometheus-values.yaml \
        --wait \
        --timeout=10m
    
    log "Prometheus stack deployed successfully"
}

deploy_datadog_agent() {
    if [ -z "${DATADOG_API_KEY:-}" ]; then
        log "Skipping Datadog deployment (no API key provided)"
        return 0
    fi
    
    log "Deploying Datadog agent..."
    
    cat > /tmp/datadog-values.yaml << EOF
datadog:
  apiKeyExistingSecret: datadog-secrets
  site: datadoghq.com
  logs:
    enabled: true
    containerCollectAll: true
  apm:
    enabled: true
    portEnabled: true
  processAgent:
    enabled: true
  systemProbe:
    enabled: true
  kubeStateMetricsEnabled: true
  kubeStateMetricsCore:
    enabled: true
  prometheusScrapeEnabled: true
  prometheusScrapeServiceEndpoints: true
  prometheusScrapeChecks: true
  
clusterAgent:
  enabled: true
  metricsProvider:
    enabled: true
  resources:
    requests:
      memory: 512Mi
      cpu: 200m
    limits:
      memory: 1Gi
      cpu: 500m

agents:
  containers:
    agent:
      resources:
        requests:
          memory: 256Mi
          cpu: 200m
        limits:
          memory: 512Mi
          cpu: 500m
    processAgent:
      resources:
        requests:
          memory: 64Mi
          cpu: 100m
        limits:
          memory: 128Mi
          cpu: 200m
    traceAgent:
      resources:
        requests:
          memory: 128Mi
          cpu: 100m
        limits:
          memory: 256Mi
          cpu: 200m
    systemProbe:
      resources:
        requests:
          memory: 64Mi
          cpu: 100m
        limits:
          memory: 128Mi
          cpu: 200m
EOF

    helm upgrade --install "$RELEASE_NAME-datadog" datadog/datadog \
        --namespace="$NAMESPACE" \
        --values /tmp/datadog-values.yaml \
        --wait \
        --timeout=5m
    
    log "Datadog agent deployed successfully"
}

setup_grafana_dashboards() {
    log "Setting up Grafana dashboards..."
    
    # Wait for Grafana to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana -n "$NAMESPACE" --timeout=300s
    
    # Import ARES ChronoFabric dashboard
    kubectl create configmap ares-dashboard \
        --from-file="$MONITORING_DIR/grafana-dashboard.json" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply dashboard configmap to Grafana
    kubectl label configmap ares-dashboard \
        grafana_dashboard=1 \
        --namespace="$NAMESPACE"
    
    log "Grafana dashboards configured"
}

create_service_monitors() {
    log "Creating ServiceMonitor resources..."
    
    cat << EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ares-csf-core
  namespace: $NAMESPACE
  labels:
    app: ares-csf
    prometheus: kube-prometheus
spec:
  selector:
    matchLabels:
      app: ares-csf
  endpoints:
  - port: metrics
    interval: 5s
    path: /metrics
    scheme: https
    tlsConfig:
      insecureSkipVerify: false
      ca:
        secret:
          name: ares-tls-ca
          key: ca.crt
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ares-business-metrics
  namespace: $NAMESPACE
  labels:
    app: ares-business-metrics
    prometheus: kube-prometheus
spec:
  selector:
    matchLabels:
      app: ares-business-metrics
  endpoints:
  - port: http
    interval: 60s
    path: /business-metrics
    scheme: https
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ares-quantum-metrics
  namespace: $NAMESPACE
  labels:
    app: ares-quantum-metrics
    prometheus: kube-prometheus
spec:
  selector:
    matchLabels:
      app: ares-quantum-metrics
  endpoints:
  - port: metrics
    interval: 5s
    path: /quantum-metrics
    scheme: https
EOF

    log "ServiceMonitor resources created"
}

configure_alertmanager() {
    log "Configuring Alertmanager..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-main
  namespace: $NAMESPACE
type: Opaque
stringData:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'smtp.ares-internal.com:587'
      smtp_from: 'alerts@ares-systems.com'
      smtp_auth_username: 'alerts@ares-systems.com'
      smtp_auth_password: '$SMTP_PASSWORD'
      
    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'default'
      routes:
        - match:
            severity: emergency
          receiver: 'emergency-response'
          group_wait: 5s
          repeat_interval: 5m
        - match:
            severity: critical
          receiver: 'critical-response'
          group_wait: 10s
          repeat_interval: 15m
        - match:
            team: business
          receiver: 'business-team'
          group_wait: 30s
          repeat_interval: 2h
        - match:
            team: security
          receiver: 'security-team'
          group_wait: 5s
          repeat_interval: 30m
    
    receivers:
      - name: 'default'
        email_configs:
          - to: 'ops-team@ares-systems.com'
            subject: 'ARES Alert: {{ .GroupLabels.alertname }}'
            body: |
              {{ range .Alerts }}
              Alert: {{ .Annotations.summary }}
              Description: {{ .Annotations.description }}
              Severity: {{ .Labels.severity }}
              {{ end }}
      
      - name: 'emergency-response'
        slack_configs:
          - api_url: '$SLACK_WEBHOOK_URL'
            channel: '#ares-emergency'
            title: '🚨 EMERGENCY: {{ .GroupLabels.alertname }}'
            text: |
              {{ range .Alerts }}
              *EMERGENCY ALERT*
              Summary: {{ .Annotations.summary }}
              Description: {{ .Annotations.description }}
              Action Required: {{ .Annotations.action_required }}
              {{ end }}
        pagerduty_configs:
          - routing_key: '$PAGERDUTY_INTEGRATION_KEY'
            description: 'ARES Emergency: {{ .GroupLabels.alertname }}'
            severity: 'critical'
      
      - name: 'critical-response'
        slack_configs:
          - api_url: '$SLACK_WEBHOOK_URL'
            channel: '#ares-alerts'
            title: '⚠️ CRITICAL: {{ .GroupLabels.alertname }}'
            text: |
              {{ range .Alerts }}
              *Critical Alert*
              Summary: {{ .Annotations.summary }}
              Description: {{ .Annotations.description }}
              Runbook: {{ .Annotations.runbook_url }}
              {{ end }}
      
      - name: 'business-team'
        email_configs:
          - to: 'business-team@ares-systems.com'
            subject: 'ARES Business Alert: {{ .GroupLabels.alertname }}'
            body: |
              {{ range .Alerts }}
              Business Impact Alert
              Summary: {{ .Annotations.summary }}
              Business Impact: {{ .Annotations.business_impact }}
              Revenue Impact: {{ .Annotations.estimated_revenue_impact }}
              {{ end }}
      
      - name: 'security-team'
        email_configs:
          - to: 'security-team@ares-systems.com'
            subject: 'ARES Security Alert: {{ .GroupLabels.alertname }}'
            body: |
              {{ range .Alerts }}
              Security Alert
              Summary: {{ .Annotations.summary }}
              Description: {{ .Annotations.description }}
              Compliance Impact: {{ .Labels.category }}
              {{ end }}
        slack_configs:
          - api_url: '$SLACK_WEBHOOK_URL'
            channel: '#ares-security'
            title: '🔒 SECURITY: {{ .GroupLabels.alertname }}'
    
    inhibit_rules:
      - source_match:
          severity: 'emergency'
        target_match:
          severity: 'critical'
        equal: ['alertname', 'cluster', 'service']
      - source_match:
          severity: 'critical'
        target_match:
          severity: 'warning'
        equal: ['alertname', 'cluster', 'service']
EOF

    log "Alertmanager configuration applied"
}

deploy_business_metrics_collector() {
    log "Deploying business metrics collector..."
    
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: business-metrics-collector
  namespace: $NAMESPACE
  labels:
    app: ares-business-metrics
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ares-business-metrics
  template:
    metadata:
      labels:
        app: ares-business-metrics
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/business-metrics"
    spec:
      containers:
      - name: business-metrics
        image: ghcr.io/ares-systems/business-metrics-collector:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: DATADOG_API_KEY
          valueFrom:
            secretKeyRef:
              name: datadog-secrets
              key: api-key
              optional: true
        - name: PROMETHEUS_ENDPOINT
          value: "http://prometheus-operated:9090"
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
            cpu: 500m
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: business-metrics-collector
  namespace: $NAMESPACE
  labels:
    app: ares-business-metrics
spec:
  selector:
    app: ares-business-metrics
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  type: ClusterIP
EOF

    log "Business metrics collector deployed"
}

deploy_custom_exporters() {
    log "Deploying custom ARES exporters..."
    
    # Quantum metrics exporter
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-metrics-exporter
  namespace: $NAMESPACE
  labels:
    app: ares-quantum-metrics
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ares-quantum-metrics
  template:
    metadata:
      labels:
        app: ares-quantum-metrics
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9100"
        prometheus.io/path: "/quantum-metrics"
    spec:
      containers:
      - name: quantum-exporter
        image: ghcr.io/ares-systems/quantum-metrics-exporter:v1.0.0
        ports:
        - containerPort: 9100
          name: metrics
        env:
        - name: QUANTUM_COHERENCE_THRESHOLD
          value: "0.95"
        - name: GATE_FIDELITY_TARGET
          value: "0.99"
        resources:
          requests:
            memory: 128Mi
            cpu: 100m
          limits:
            memory: 256Mi
            cpu: 200m
---
apiVersion: v1
kind: Service
metadata:
  name: quantum-metrics-exporter
  namespace: $NAMESPACE
  labels:
    app: ares-quantum-metrics
spec:
  selector:
    app: ares-quantum-metrics
  ports:
  - name: metrics
    port: 9100
    targetPort: 9100
EOF

    # Security scanner exporter
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-scanner-exporter
  namespace: $NAMESPACE
  labels:
    app: ares-security-scanner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ares-security-scanner
  template:
    metadata:
      labels:
        app: ares-security-scanner
    spec:
      containers:
      - name: security-scanner
        image: ghcr.io/ares-systems/security-scanner:v1.0.0
        ports:
        - containerPort: 9200
          name: metrics
        env:
        - name: SCAN_INTERVAL
          value: "300"
        - name: COMPLIANCE_FRAMEWORKS
          value: "soc2,iso27001,pci-dss"
        resources:
          requests:
            memory: 512Mi
            cpu: 200m
          limits:
            memory: 1Gi
            cpu: 500m
        volumeMounts:
        - name: scan-results
          mountPath: /var/lib/security-scans
      volumes:
      - name: scan-results
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: security-scanner-exporter
  namespace: $NAMESPACE
  labels:
    app: ares-security-scanner
spec:
  selector:
    app: ares-security-scanner
  ports:
  - name: metrics
    port: 9200
    targetPort: 9200
EOF

    log "Custom exporters deployed"
}

configure_network_policies() {
    log "Configuring network policies for monitoring..."
    
    cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: monitoring-network-policy
  namespace: $NAMESPACE
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ares-production
    - namespaceSelector:
        matchLabels:
          name: istio-system
    ports:
    - protocol: TCP
      port: 9090
    - protocol: TCP
      port: 3000
    - protocol: TCP
      port: 9093
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
EOF

    log "Network policies configured"
}

setup_rbac() {
    log "Setting up RBAC for monitoring stack..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ares-monitoring
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ares-monitoring
rules:
- apiGroups: [""]
  resources: ["nodes", "nodes/proxy", "services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["extensions"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "daemonsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["monitoring.coreos.com"]
  resources: ["servicemonitors", "prometheusrules"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ares-monitoring
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ares-monitoring
subjects:
- kind: ServiceAccount
  name: ares-monitoring
  namespace: $NAMESPACE
EOF

    log "RBAC configured"
}

validate_deployment() {
    log "Validating monitoring stack deployment..."
    
    # Check if all pods are running
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=alertmanager -n "$NAMESPACE" --timeout=300s
    
    # Verify metric collection
    PROMETHEUS_URL="http://$(kubectl get svc prometheus-operated -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}'):9090"
    
    log "Testing Prometheus connectivity..."
    kubectl run prometheus-test --image=curlimages/curl --rm -i --restart=Never -- \
        curl -s "$PROMETHEUS_URL/api/v1/query?query=up" | grep -q '"status":"success"' || \
        error "Prometheus is not responding correctly"
    
    # Get Grafana admin password
    GRAFANA_PASSWORD=$(kubectl get secret prometheus-grafana -n "$NAMESPACE" -o jsonpath="{.data.admin-password}" | base64 --decode)
    log "Grafana admin password: $GRAFANA_PASSWORD"
    
    # Display service endpoints
    log "Monitoring stack endpoints:"
    echo "  Prometheus: http://$(kubectl get svc prometheus-operated -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090"
    echo "  Grafana: http://$(kubectl get svc prometheus-grafana -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):3000"
    echo "  Alertmanager: http://$(kubectl get svc alertmanager-operated -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9093"
    
    log "Monitoring stack deployment validation completed successfully"
}

create_monitoring_ingress() {
    log "Creating ingress for monitoring services..."
    
    cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ares-monitoring-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: monitoring-auth
    nginx.ingress.kubernetes.io/auth-realm: 'ARES Monitoring'
spec:
  tls:
  - hosts:
    - prometheus.ares-internal.com
    - grafana.ares-internal.com
    - alertmanager.ares-internal.com
    secretName: monitoring-tls
  rules:
  - host: prometheus.ares-internal.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prometheus-operated
            port:
              number: 9090
  - host: grafana.ares-internal.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prometheus-grafana
            port:
              number: 3000
  - host: alertmanager.ares-internal.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: alertmanager-operated
            port:
              number: 9093
EOF

    log "Monitoring ingress configured"
}

main() {
    log "Starting ARES ChronoFabric Enterprise Monitoring Stack deployment..."
    
    check_prerequisites
    create_namespace
    add_helm_repositories
    generate_monitoring_secrets
    setup_rbac
    deploy_prometheus_stack
    deploy_datadog_agent
    configure_alertmanager
    create_service_monitors
    deploy_business_metrics_collector
    deploy_custom_exporters
    configure_network_policies
    setup_grafana_dashboards
    create_monitoring_ingress
    validate_deployment
    
    log "Enterprise monitoring stack deployment completed successfully!"
    log ""
    log "Next steps:"
    log "1. Access Grafana at https://grafana.ares-internal.com (admin/$GRAFANA_PASSWORD)"
    log "2. Configure Datadog dashboards if API key was provided"
    log "3. Test alerting by triggering a test alert"
    log "4. Review compliance scores in the enterprise dashboard"
    log ""
    log "Monitoring stack is now collecting business KPIs, quantum metrics, and compliance data."
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi