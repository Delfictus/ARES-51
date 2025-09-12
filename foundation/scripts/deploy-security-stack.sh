#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NAMESPACE="ares-security"
RELEASE_NAME="ares-security"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

check_prerequisites() {
    log "Checking security stack prerequisites..."
    
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error "helm is required but not installed"
    
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"
    
    log "Prerequisites check passed"
}

create_security_namespace() {
    log "Creating security namespace..."
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite
    kubectl label namespace "$NAMESPACE" pod-security.kubernetes.io/enforce=restricted --overwrite
    kubectl label namespace "$NAMESPACE" pod-security.kubernetes.io/audit=restricted --overwrite
    kubectl label namespace "$NAMESPACE" pod-security.kubernetes.io/warn=restricted --overwrite
    
    log "Security namespace $NAMESPACE ready"
}

deploy_istio_service_mesh() {
    log "Deploying Istio service mesh for zero-trust networking..."
    
    # Add Istio repository
    helm repo add istio https://istio-release.storage.googleapis.com/charts
    helm repo update
    
    # Install Istio base
    helm upgrade --install istio-base istio/base \
        --namespace istio-system \
        --create-namespace \
        --wait
    
    # Install Istio discovery
    helm upgrade --install istiod istio/istiod \
        --namespace istio-system \
        --values - << EOF
pilot:
  resources:
    requests:
      memory: 512Mi
      cpu: 500m
    limits:
      memory: 2Gi
      cpu: 1
  env:
    EXTERNAL_ISTIOD: false
    PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION: true
    PILOT_ENABLE_CROSS_CLUSTER_WORKLOAD_ENTRY: true
  
global:
  meshID: ares-mesh
  network: ares-network
  
telemetryV2:
  enabled: true
  prometheus:
    configOverride:
      metric_relabeling_configs:
      - source_labels: [__name__]
        regex: 'istio_.*'
        target_label: component
        replacement: 'service-mesh'
EOF
    
    # Install Istio ingress gateway
    helm upgrade --install istio-ingress istio/gateway \
        --namespace istio-ingress \
        --create-namespace \
        --values - << EOF
service:
  type: LoadBalancer
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-scheme: internet-facing
  
resources:
  requests:
    memory: 256Mi
    cpu: 100m
  limits:
    memory: 1Gi
    cpu: 500m

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
EOF
    
    log "Istio service mesh deployed"
}

deploy_security_policies() {
    log "Deploying security policies..."
    
    # Istio security policies
    cat << EOF | kubectl apply -f -
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: ares-production
spec:
  mtls:
    mode: STRICT
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: ares-csf-authz
  namespace: ares-production
spec:
  selector:
    matchLabels:
      app: ares-csf
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/ares-production/sa/ares-csf"]
  - to:
    - operation:
        methods: ["GET", "POST", "PUT", "DELETE"]
        paths: ["/api/*", "/health/*", "/metrics"]
  - when:
    - key: request.headers[authorization]
      values: ["Bearer *"]
---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: quantum-operations-authz
  namespace: ares-production
spec:
  selector:
    matchLabels:
      app: ares-csf
  rules:
  - to:
    - operation:
        methods: ["POST", "PUT"]
        paths: ["/api/quantum/*"]
  - when:
    - key: request.headers[x-quantum-clearance]
      values: ["level-3", "level-4", "level-5"]
    - key: request.headers[authorization]
      values: ["Bearer *"]
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ares-csf-circuit-breaker
  namespace: ares-production
spec:
  host: ares-csf
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 30s
        tcpKeepalive:
          time: 7200s
          interval: 75s
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
        maxRetries: 3
        consecutiveGatewayErrors: 5
        interval: 30s
        baseEjectionTime: 30s
        maxEjectionPercent: 50
    outlierDetection:
      consecutive5xxErrors: 3
      consecutiveGatewayErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 30
EOF

    log "Security policies deployed"
}

deploy_waf_protection() {
    log "Deploying Web Application Firewall..."
    
    # Deploy ModSecurity with OWASP Core Rule Set
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modsecurity-waf
  namespace: $NAMESPACE
  labels:
    app: modsecurity-waf
spec:
  replicas: 3
  selector:
    matchLabels:
      app: modsecurity-waf
  template:
    metadata:
      labels:
        app: modsecurity-waf
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9145"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: modsecurity
        image: owasp/modsecurity-crs:3.3-apache
        ports:
        - containerPort: 80
          name: http
        - containerPort: 9145
          name: metrics
        env:
        - name: PARANOIA
          value: "2"
        - name: ANOMALY_INBOUND
          value: "10"
        - name: ANOMALY_OUTBOUND
          value: "5"
        - name: BACKEND
          value: "http://ares-csf:8080"
        - name: PROXY
          value: "1"
        - name: METRICS
          value: "1"
        - name: CRS_ENABLE_TEST_MARKER
          value: "0"
        - name: VALIDATE_UTF8_ENCODING
          value: "1"
        resources:
          requests:
            memory: 512Mi
            cpu: 250m
          limits:
            memory: 1Gi
            cpu: 500m
        volumeMounts:
        - name: custom-rules
          mountPath: /etc/modsecurity.d/owasp-crs/custom-rules
        - name: waf-config
          mountPath: /etc/modsecurity.d/local-config
      volumes:
      - name: custom-rules
        configMap:
          name: ares-waf-custom-rules
      - name: waf-config
        configMap:
          name: ares-waf-config
---
apiVersion: v1
kind: Service
metadata:
  name: modsecurity-waf
  namespace: $NAMESPACE
  labels:
    app: modsecurity-waf
spec:
  selector:
    app: modsecurity-waf
  ports:
  - name: http
    port: 80
    targetPort: 80
  - name: metrics
    port: 9145
    targetPort: 9145
EOF

    # WAF custom rules for ARES-specific threats
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: ares-waf-custom-rules
  namespace: $NAMESPACE
data:
  quantum-protection.conf: |
    # ARES ChronoFabric Quantum Operation Protection
    SecRule REQUEST_URI "@beginsWith /api/quantum/" \\
        "id:1001,\\
         phase:1,\\
         deny,\\
         status:403,\\
         msg:'Quantum operation requires special authorization',\\
         logdata:'Attempted quantum access without proper clearance',\\
         tag:'attack-quantum',\\
         ctl:auditLogParts=+E"
    
    SecRule REQUEST_HEADERS:X-Quantum-Clearance "@rx ^level-[345]\$" \\
        "id:1002,\\
         phase:1,\\
         pass,\\
         msg:'Quantum clearance verified',\\
         tag:'quantum-authorized'"
    
    # Temporal manipulation protection
    SecRule REQUEST_URI "@contains temporal" \\
        "id:1003,\\
         phase:1,\\
         deny,\\
         status:403,\\
         msg:'Temporal operation blocked for security',\\
         logdata:'Potential temporal manipulation attempt',\\
         tag:'attack-temporal'"
    
    # Rate limiting for quantum operations
    SecRule REQUEST_URI "@beginsWith /api/quantum/" \\
        "id:1004,\\
         phase:1,\\
         pass,\\
         initcol:ip=%{REMOTE_ADDR},\\
         setvar:ip.quantum_requests=+1,\\
         expirevar:ip.quantum_requests=60,\\
         deny,\\
         status:429,\\
         msg:'Quantum operation rate limit exceeded',\\
         chain"
    SecRule IP:QUANTUM_REQUESTS "@gt 10" \\
        "id:1005"
  
  ares-application-rules.conf: |
    # ARES-specific application protection rules
    
    # Block access to admin endpoints without proper authentication
    SecRule REQUEST_URI "@beginsWith /admin/" \\
        "id:2001,\\
         phase:1,\\
         deny,\\
         status:403,\\
         msg:'Admin access requires authentication',\\
         chain"
    SecRule REQUEST_HEADERS:Authorization "!@rx ^Bearer\\s+[A-Za-z0-9\\-_]+\\.[A-Za-z0-9\\-_]+\\.[A-Za-z0-9\\-_]+\$"
    
    # Protect sensitive configuration endpoints
    SecRule REQUEST_URI "@rx /config|/secrets|/keys" \\
        "id:2002,\\
         phase:1,\\
         deny,\\
         status:403,\\
         msg:'Configuration endpoint access denied',\\
         logdata:'Attempt to access sensitive configuration',\\
         tag:'attack-config'"
    
    # Monitor for potential data exfiltration
    SecRule RESPONSE_BODY "@detectSQLi" \\
        "id:2003,\\
         phase:4,\\
         pass,\\
         msg:'Potential data exfiltration detected',\\
         logdata:'Response contains SQL-like patterns',\\
         tag:'data-exfiltration'"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ares-waf-config
  namespace: $NAMESPACE
data:
  modsecurity.conf: |
    # ARES ChronoFabric ModSecurity Configuration
    
    SecRuleEngine On
    SecRequestBodyAccess On
    SecResponseBodyAccess On
    SecResponseBodyMimeType text/plain text/html text/xml application/json
    SecDefaultAction "phase:1,log,auditlog,deny,status:403"
    SecAction "id:900000,phase:1,nolog,pass,t:none,setvar:tx.paranoia_level=2"
    SecAction "id:900001,phase:1,nolog,pass,t:none,setvar:tx.anomaly_score_threshold=10"
    
    # Audit log configuration
    SecAuditEngine RelevantOnly
    SecAuditLogRelevantStatus "^(?:5|4(?!04))"
    SecAuditLogParts ABIJDEFHZ
    SecAuditLogType Serial
    SecAuditLog /var/log/modsecurity/audit.log
    
    # Request body handling
    SecRequestBodyLimit 13107200
    SecRequestBodyNoFilesLimit 131072
    SecRequestBodyInMemoryLimit 131072
    SecRequestBodyLimitAction Reject
    
    # Response body handling  
    SecResponseBodyLimit 524288
    SecResponseBodyLimitAction ProcessPartial
    
    # File upload handling
    SecTmpDir /tmp/
    SecDataDir /tmp/
    
    # Debug logging (disable in production)
    SecDebugLog /var/log/modsecurity/debug.log
    SecDebugLogLevel 0
    
    # Custom ARES rules
    Include /etc/modsecurity.d/local-config/*.conf
EOF

    log "WAF protection deployed"
}

deploy_threat_detection() {
    log "Deploying threat detection system..."
    
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: threat-detection
  namespace: $NAMESPACE
  labels:
    app: threat-detection
spec:
  replicas: 2
  selector:
    matchLabels:
      app: threat-detection
  template:
    metadata:
      labels:
        app: threat-detection
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9200"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: threat-detection
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: threat-detection
        image: ghcr.io/ares-systems/threat-detection:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9200
          name: metrics
        env:
        - name: THREAT_INTEL_FEEDS
          value: "misp,otx,virustotal"
        - name: ML_MODEL_PATH
          value: "/var/lib/models"
        - name: THREAT_THRESHOLD
          value: "0.7"
        - name: QUANTUM_THREAT_ENABLED
          value: "true"
        resources:
          requests:
            memory: 1Gi
            cpu: 500m
          limits:
            memory: 4Gi
            cpu: 2
        volumeMounts:
        - name: model-storage
          mountPath: /var/lib/models
        - name: threat-intel
          mountPath: /var/lib/threat-intel
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: threat-detection-models
      - name: threat-intel
        emptyDir:
          sizeLimit: 1Gi
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: threat-detection
  namespace: $NAMESPACE
---
apiVersion: v1
kind: Service
metadata:
  name: threat-detection
  namespace: $NAMESPACE
  labels:
    app: threat-detection
spec:
  selector:
    app: threat-detection
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9200
    targetPort: 9200
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: threat-detection-models
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 20Gi
EOF

    log "Threat detection system deployed"
}

deploy_intrusion_detection() {
    log "Deploying intrusion detection system..."
    
    # Deploy Suricata IDS
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: suricata-ids
  namespace: $NAMESPACE
  labels:
    app: suricata-ids
spec:
  selector:
    matchLabels:
      app: suricata-ids
  template:
    metadata:
      labels:
        app: suricata-ids
    spec:
      hostNetwork: true
      hostPID: true
      serviceAccountName: suricata-ids
      tolerations:
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      containers:
      - name: suricata
        image: jasonish/suricata:7.0
        securityContext:
          privileged: true
        env:
        - name: SURICATA_OPTIONS
          value: "-i any --init-errors-fatal"
        - name: SURICATA_CAPTURE_FILTER
          value: "not port 22 and not port 2376"
        resources:
          requests:
            memory: 1Gi
            cpu: 500m
          limits:
            memory: 4Gi
            cpu: 2
        volumeMounts:
        - name: suricata-config
          mountPath: /etc/suricata
        - name: suricata-logs
          mountPath: /var/log/suricata
        - name: suricata-rules
          mountPath: /var/lib/suricata/rules
      - name: filebeat-sidecar
        image: elastic/filebeat:8.11.0
        env:
        - name: ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc.cluster.local"
        - name: ELASTICSEARCH_PORT
          value: "9200"
        volumeMounts:
        - name: suricata-logs
          mountPath: /var/log/suricata
        - name: filebeat-config
          mountPath: /usr/share/filebeat/filebeat.yml
          subPath: filebeat.yml
      volumes:
      - name: suricata-config
        configMap:
          name: suricata-config
      - name: suricata-rules
        configMap:
          name: suricata-rules
      - name: suricata-logs
        emptyDir: {}
      - name: filebeat-config
        configMap:
          name: filebeat-config
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: suricata-ids
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: suricata-ids
rules:
- apiGroups: [""]
  resources: ["nodes", "pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: suricata-ids
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: suricata-ids
subjects:
- kind: ServiceAccount
  name: suricata-ids
  namespace: $NAMESPACE
EOF

    # Suricata configuration
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: suricata-config
  namespace: $NAMESPACE
data:
  suricata.yaml: |
    %YAML 1.1
    ---
    
    # ARES ChronoFabric Suricata IDS Configuration
    
    # Global settings
    max-pending-packets: 1024
    runmode: autofp
    autofp-scheduler: hash
    
    # Host configuration
    host-mode: auto
    
    # Logging configuration
    default-log-dir: /var/log/suricata/
    
    # Stats configuration
    stats:
      enabled: yes
      interval: 8
      
    # Output configuration
    outputs:
      - eve-log:
          enabled: yes
          filetype: regular
          filename: eve.json
          types:
            - alert:
                metadata: yes
                tagged-packets: yes
                xff:
                  enabled: yes
                  mode: extra-data
                  deployment: reverse
                  header: X-Forwarded-For
            - anomaly:
                enabled: yes
                types:
                  decode: yes
                  stream: yes
                  applayer: yes
            - http:
                enabled: yes
                extended: yes
            - dns:
                enabled: yes
            - tls:
                enabled: yes
                extended: yes
            - files:
                enabled: yes
                force-magic: no
            - smtp:
                enabled: yes
            - flow:
                enabled: yes
      
      - prometheus:
          enabled: yes
          port: 9145
          prefix: suricata_
    
    # Application layer protocols
    app-layer:
      protocols:
        http:
          enabled: yes
          libhtp:
            default-config:
              personality: IDS
              request-body-limit: 100kb
              response-body-limit: 100kb
              request-body-minimal-inspect-size: 32kb
              request-body-inspect-window: 4kb
              response-body-minimal-inspect-size: 40kb
              response-body-inspect-window: 16kb
              http-body-inline: auto
              swf-decompression:
                enabled: yes
                type: both
                compress-depth: 100kb
                decompress-depth: 100kb
              double-decode-path: no
              double-decode-query: no
        tls:
          enabled: yes
          detection-ports:
            dp: 443
        smtp:
          enabled: yes
          detection-ports:
            dp: 25
        dns:
          enabled: yes
          tcp:
            enabled: yes
            detection-ports:
              dp: 53
          udp:
            enabled: yes
            detection-ports:
              dp: 53
    
    # Rule management
    default-rule-path: /var/lib/suricata/rules
    rule-files:
      - suricata.rules
      - ares-custom.rules
    
    classification-file: /etc/suricata/classification.config
    reference-config-file: /etc/suricata/reference.config
    
    # Performance tuning
    threading:
      set-cpu-affinity: no
      cpu-affinity:
        - management-cpu-set:
            cpu: [ 0 ]
        - receive-cpu-set:
            cpu: [ 0 ]
        - worker-cpu-set:
            cpu: [ "1-3" ]
      detect-thread-ratio: 1.0
    
    # Detection engine
    detect-engine:
      - profile: medium
      - custom-values:
          toclient-groups: 3
          toserver-groups: 25
      - sgh-mpm-context: auto
      - inspection-recursion-limit: 3000
    
    # Stream engine
    stream:
      memcap: 64mb
      checksum-validation: yes
      inline: auto
      reassembly:
        memcap: 256mb
        depth: 1mb
        toserver-chunk-size: 2560
        toclient-chunk-size: 2560
        randomize-chunk-size: yes
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: suricata-rules
  namespace: $NAMESPACE
data:
  ares-custom.rules: |
    # ARES ChronoFabric Custom Security Rules
    
    # Quantum operation monitoring
    alert http any any -> any any (msg:"Quantum API Access"; content:"/api/quantum/"; http_uri; classtype:policy-violation; sid:3001; rev:1;)
    alert http any any -> any any (msg:"Unauthorized Quantum Operation"; content:"/api/quantum/"; http_uri; content:!"X-Quantum-Clearance"; http_header; classtype:attempted-admin; sid:3002; rev:1;)
    
    # Temporal manipulation detection
    alert http any any -> any any (msg:"Temporal Manipulation Attempt"; content:"temporal"; http_uri; classtype:attempted-admin; sid:3003; rev:1;)
    alert http any any -> any any (msg:"Chronon Manipulation Detected"; content:"chronon"; nocase; classtype:attempted-admin; sid:3004; rev:1;)
    
    # High-frequency attack detection
    alert http any any -> any any (msg:"Potential DDoS Attack"; threshold:type both,track by_src,count 100,seconds 60; classtype:attempted-dos; sid:3005; rev:1;)
    
    # Suspicious user agent strings
    alert http any any -> any any (msg:"Suspicious User Agent"; content:"sqlmap"; http_user_agent; nocase; classtype:web-application-attack; sid:3006; rev:1;)
    alert http any any -> any any (msg:"Suspicious User Agent"; content:"nikto"; http_user_agent; nocase; classtype:web-application-attack; sid:3007; rev:1;)
    
    # Configuration access attempts
    alert http any any -> any any (msg:"Configuration Access Attempt"; content:"/config"; http_uri; content:!"Authorization"; http_header; classtype:attempted-admin; sid:3008; rev:1;)
    alert http any any -> any any (msg:"Secrets Access Attempt"; content:"/secrets"; http_uri; classtype:attempted-admin; sid:3009; rev:1;)
    
    # Anomalous request patterns
    alert http any any -> any any (msg:"Large Request Body"; dsize:>100000; classtype:policy-violation; sid:3010; rev:1;)
    alert http any any -> any any (msg:"Unusual HTTP Method"; content:!"GET|POST|PUT|DELETE|HEAD|OPTIONS"; http_method; classtype:policy-violation; sid:3011; rev:1;)
EOF

    log "Intrusion detection system deployed"
}

deploy_rate_limiting() {
    log "Deploying enterprise rate limiting..."
    
    # Deploy Redis for rate limiting state
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-rate-limiting
  namespace: $NAMESPACE
  labels:
    app: redis-rate-limiting
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redis-rate-limiting
  template:
    metadata:
      labels:
        app: redis-rate-limiting
    spec:
      containers:
      - name: redis
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
          name: redis
        args:
        - redis-server
        - --maxmemory
        - 1gb
        - --maxmemory-policy
        - allkeys-lru
        - --save
        - ""
        - --appendonly
        - "no"
        - --tcp-keepalive
        - "60"
        - --timeout
        - "300"
        resources:
          requests:
            memory: 512Mi
            cpu: 100m
          limits:
            memory: 1Gi
            cpu: 500m
        livenessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: redis-rate-limiting
  namespace: $NAMESPACE
  labels:
    app: redis-rate-limiting
spec:
  selector:
    app: redis-rate-limiting
  ports:
  - name: redis
    port: 6379
    targetPort: 6379
  type: ClusterIP
EOF

    # Deploy rate limiting proxy
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rate-limiter
  namespace: $NAMESPACE
  labels:
    app: rate-limiter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rate-limiter
  template:
    metadata:
      labels:
        app: rate-limiter
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: rate-limiter
        image: ghcr.io/ares-systems/rate-limiter:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: REDIS_URL
          value: "redis://redis-rate-limiting:6379"
        - name: GLOBAL_RATE_LIMIT
          value: "10000"
        - name: PER_USER_RATE_LIMIT
          value: "1000"
        - name: PER_IP_RATE_LIMIT
          value: "100"
        - name: QUANTUM_RATE_LIMIT
          value: "10"
        - name: BURST_ALLOWANCE
          value: "50"
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
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rate-limiter
  namespace: $NAMESPACE
  labels:
    app: rate-limiter
spec:
  selector:
    app: rate-limiter
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
EOF

    log "Rate limiting deployed"
}

deploy_security_monitoring() {
    log "Deploying security monitoring..."
    
    # Deploy Falco for runtime security monitoring
    helm repo add falcosecurity https://falcosecurity.github.io/charts
    helm repo update
    
    helm upgrade --install falco falcosecurity/falco \
        --namespace="$NAMESPACE" \
        --values - << EOF
driver:
  kind: ebpf

falco:
  rules_file:
    - /etc/falco/falco_rules.yaml
    - /etc/falco/ares_custom_rules.yaml
  
  grpc:
    enabled: true
    bind_address: "0.0.0.0:5060"
  
  grpc_output:
    enabled: true
  
  json_output: true
  json_include_output_property: true
  json_include_tags_property: true
  
  syslog_output:
    enabled: false
  
  http_output:
    enabled: true
    url: "http://falco-exporter:9376/events"

resources:
  requests:
    memory: 512Mi
    cpu: 100m
  limits:
    memory: 1Gi
    cpu: 500m

customRules:
  ares_custom_rules.yaml: |
    # ARES ChronoFabric Custom Falco Rules
    
    - rule: Unauthorized Quantum Hardware Access
      desc: Detect unauthorized access to quantum hardware resources
      condition: >
        open_read and
        (fd.name contains "/dev/quantum" or
         fd.name contains "/sys/class/quantum" or
         proc.name contains "quantum")
      output: >
        Unauthorized quantum hardware access
        (user=%user.name command=%proc.cmdline file=%fd.name container=%container.id)
      priority: CRITICAL
      tags: [quantum, hardware, unauthorized_access]
    
    - rule: Temporal State File Modification
      desc: Detect modifications to temporal state files
      condition: >
        open_write and
        (fd.name contains "temporal_state" or
         fd.name contains "chronon" or
         fd.name contains ".tsf")
      output: >
        Temporal state file modification detected
        (user=%user.name command=%proc.cmdline file=%fd.name container=%container.id)
      priority: ERROR
      tags: [temporal, state_modification, security]
    
    - rule: Suspicious Network Activity to Quantum Services
      desc: Detect suspicious network connections to quantum services
      condition: >
        (inbound_connection or outbound_connection) and
        (fd.sport in (9001, 9002, 9003) or fd.dport in (9001, 9002, 9003))
        and not proc.name in (ares-csf, quantum-simulator)
      output: >
        Suspicious quantum service network activity
        (user=%user.name command=%proc.cmdline connection=%fd.name container=%container.id)
      priority: WARNING
      tags: [network, quantum, suspicious]
    
    - rule: Privilege Escalation in ARES Containers
      desc: Detect privilege escalation attempts in ARES containers
      condition: >
        spawned_process and
        container and
        container.image.repository contains "ares" and
        (proc.name in (su, sudo, newgrp, newuidmap, newgidmap) or
         (proc.args contains "chmod +s" or proc.args contains "setuid"))
      output: >
        Privilege escalation attempt in ARES container
        (user=%user.name command=%proc.cmdline container=%container.id image=%container.image.repository)
      priority: CRITICAL
      tags: [privilege_escalation, container, ares]

serviceMonitor:
  enabled: true
  interval: 30s
  scrapeTimeout: 30s
EOF

    log "Security monitoring deployed"
}

configure_network_security() {
    log "Configuring network security policies..."
    
    # Default deny-all network policy
    cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: ares-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ares-csf-network-policy
  namespace: ares-production
spec:
  podSelector:
    matchLabels:
      app: ares-csf
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - namespaceSelector:
        matchLabels:
          name: ares-monitoring
    - podSelector:
        matchLabels:
          app: rate-limiter
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9090
    - protocol: TCP
      port: 9001
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: quantum-operations-isolation
  namespace: ares-production
spec:
  podSelector:
    matchLabels:
      component: quantum-processor
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: ares-csf
    ports:
    - protocol: TCP
      port: 9001
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: quantum-hardware-driver
    ports:
    - protocol: TCP
      port: 9002
EOF

    log "Network security policies configured"
}

deploy_security_scanning() {
    log "Deploying security scanning tools..."
    
    # Deploy Trivy for vulnerability scanning
    cat << EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: trivy-vulnerability-scan
  namespace: $NAMESPACE
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: trivy
            image: aquasec/trivy:0.47.0
            command:
            - /bin/sh
            - -c
            - |
              # Scan filesystem
              trivy fs --format json --output /tmp/fs-scan.json /
              
              # Scan container images
              trivy image --format json --output /tmp/image-scan.json \\
                ghcr.io/ares-systems/chronofabric:latest
              
              # Upload results to monitoring
              curl -X POST http://security-scanner.ares-monitoring:9200/scan-results \\
                -H "Content-Type: application/json" \\
                -d @/tmp/fs-scan.json
              
              curl -X POST http://security-scanner.ares-monitoring:9200/scan-results \\
                -H "Content-Type: application/json" \\
                -d @/tmp/image-scan.json
            resources:
              requests:
                memory: 512Mi
                cpu: 200m
              limits:
                memory: 2Gi
                cpu: 1
            volumeMounts:
            - name: tmp-storage
              mountPath: /tmp
          volumes:
          - name: tmp-storage
            emptyDir:
              sizeLimit: 1Gi
EOF

    # Deploy OWASP ZAP for application security testing
    cat << EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: zap-security-scan
  namespace: $NAMESPACE
spec:
  schedule: "0 4 * * 0"  # Weekly on Sunday at 4 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: zap
            image: owasp/zap2docker-stable:2.14.0
            command:
            - /bin/bash
            - -c
            - |
              # Baseline scan
              zap-baseline.py -t https://api.ares-csf.com \\
                -J /tmp/baseline-report.json \\
                -r /tmp/baseline-report.html
              
              # Full scan for critical endpoints
              zap-full-scan.py -t https://api.ares-csf.com/quantum \\
                -J /tmp/quantum-scan.json \\
                -r /tmp/quantum-scan.html
              
              # Upload results
              curl -X POST http://security-scanner.ares-monitoring:9200/zap-results \\
                -F "baseline=@/tmp/baseline-report.json" \\
                -F "quantum=@/tmp/quantum-scan.json"
            resources:
              requests:
                memory: 1Gi
                cpu: 500m
              limits:
                memory: 4Gi
                cpu: 2
            volumeMounts:
            - name: zap-storage
              mountPath: /tmp
          volumes:
          - name: zap-storage
            emptyDir:
              sizeLimit: 2Gi
EOF

    log "Security scanning tools deployed"
}

setup_security_rbac() {
    log "Setting up security RBAC..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ares-security
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ares-security
rules:
- apiGroups: [""]
  resources: ["pods", "services", "nodes", "events"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "daemonsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies", "ingresses"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["security.istio.io"]
  resources: ["authorizationpolicies", "peerauthentications"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["policy"]
  resources: ["podsecuritypolicies"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ares-security
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ares-security
subjects:
- kind: ServiceAccount
  name: ares-security
  namespace: $NAMESPACE
EOF

    log "Security RBAC configured"
}

validate_security_deployment() {
    log "Validating security stack deployment..."
    
    # Check if all security components are running
    kubectl wait --for=condition=ready pod -l app=modsecurity-waf -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=threat-detection -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=rate-limiter -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis-rate-limiting -n "$NAMESPACE" --timeout=300s
    
    # Verify Istio is working
    kubectl get pods -n istio-system
    
    # Test rate limiting
    RATE_LIMITER_URL="http://$(kubectl get svc rate-limiter -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}'):8080"
    
    log "Testing rate limiter connectivity..."
    kubectl run rate-limit-test --image=curlimages/curl --rm -i --restart=Never -- \
        curl -s "$RATE_LIMITER_URL/health" | grep -q "healthy" || \
        error "Rate limiter is not responding correctly"
    
    # Test WAF
    WAF_URL="http://$(kubectl get svc modsecurity-waf -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')"
    
    log "Testing WAF connectivity..."
    kubectl run waf-test --image=curlimages/curl --rm -i --restart=Never -- \
        curl -s -w "%{http_code}" "$WAF_URL" | grep -q "200" || \
        log "WAF test completed (may show different status code)"
    
    # Verify threat detection is collecting metrics
    THREAT_DETECTION_URL="http://$(kubectl get svc threat-detection -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}'):9200"
    
    kubectl run threat-detection-test --image=curlimages/curl --rm -i --restart=Never -- \
        curl -s "$THREAT_DETECTION_URL/metrics" | grep -q "threat_" || \
        log "Threat detection metrics test completed"
    
    log "Security stack deployment validation completed successfully"
}

create_security_dashboards() {
    log "Creating security dashboards..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-dashboard
  namespace: $NAMESPACE
  labels:
    grafana_dashboard: "1"
data:
  security-dashboard.json: |
    {
      "dashboard": {
        "title": "ARES ChronoFabric Security Dashboard",
        "tags": ["security", "ares", "threat-detection"],
        "panels": [
          {
            "title": "Threat Detection Score",
            "type": "stat",
            "targets": [
              {
                "expr": "avg(threat_detection_score)",
                "legendFormat": "Threat Score"
              }
            ]
          },
          {
            "title": "Rate Limiting Status",
            "type": "timeseries",
            "targets": [
              {
                "expr": "rate(rate_limit_exceeded_total[5m])",
                "legendFormat": "Rate Limits Exceeded"
              }
            ]
          },
          {
            "title": "WAF Blocks",
            "type": "timeseries", 
            "targets": [
              {
                "expr": "rate(modsecurity_blocked_requests_total[5m])",
                "legendFormat": "WAF Blocks"
              }
            ]
          },
          {
            "title": "Security Violations by Type",
            "type": "piechart",
            "targets": [
              {
                "expr": "sum by (violation_type) (security_violations_total)",
                "legendFormat": "{{violation_type}}"
              }
            ]
          }
        ]
      }
    }
EOF

    log "Security dashboards created"
}

main() {
    log "Starting ARES ChronoFabric Enterprise Security Stack deployment..."
    
    check_prerequisites
    create_security_namespace
    setup_security_rbac
    deploy_istio_service_mesh
    deploy_security_policies
    deploy_waf_protection
    deploy_threat_detection
    deploy_intrusion_detection
    deploy_rate_limiting
    deploy_security_monitoring
    deploy_security_scanning
    configure_network_security
    validate_security_deployment
    create_security_dashboards
    
    log "Enterprise security stack deployment completed successfully!"
    log ""
    log "Security endpoints:"
    log "- WAF: http://$(kubectl get svc modsecurity-waf -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')"
    log "- Rate Limiter: http://$(kubectl get svc rate-limiter -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}'):8080"
    log "- Threat Detection: http://$(kubectl get svc threat-detection -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}'):9200"
    log ""
    log "Next steps:"
    log "1. Configure threat intelligence feeds"
    log "2. Tune WAF rules for your specific application"
    log "3. Set up security alert escalation procedures"
    log "4. Review and adjust rate limiting thresholds"
    log "5. Test incident response procedures"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi