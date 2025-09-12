#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SECRETS_DIR="$REPO_ROOT/deployments/secrets"

VAULT_ADDR="${VAULT_ADDR:-https://vault.ares-internal.com:8200}"
VAULT_NAMESPACE="${VAULT_NAMESPACE:-ares-production}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

check_vault_prerequisites() {
    log "Checking Vault prerequisites..."
    
    command -v vault >/dev/null 2>&1 || error "vault CLI is required but not installed"
    
    if [ -z "${VAULT_TOKEN:-}" ]; then
        error "VAULT_TOKEN environment variable is required"
    fi
    
    # Test Vault connectivity
    vault status >/dev/null 2>&1 || error "Cannot connect to Vault at $VAULT_ADDR"
    
    log "Vault prerequisites check passed"
}

enable_secret_engines() {
    log "Enabling secret engines..."
    
    # Key-Value v2 secret engine for application secrets
    vault secrets enable -path=secret kv-v2 || log "KV-v2 engine already enabled"
    
    # Database secrets engine for dynamic database credentials
    vault secrets enable database || log "Database engine already enabled"
    
    # PKI secret engine for certificate management
    vault secrets enable -path=pki pki || log "PKI engine already enabled"
    vault secrets tune -max-lease-ttl=8760h pki || log "PKI TTL already configured"
    
    # Development PKI engine
    vault secrets enable -path=pki_dev pki || log "Development PKI engine already enabled"
    vault secrets tune -max-lease-ttl=720h pki_dev || log "Development PKI TTL already configured"
    
    # Transit secrets engine for encryption as a service
    vault secrets enable transit || log "Transit engine already enabled"
    
    log "Secret engines enabled successfully"
}

configure_pki_infrastructure() {
    log "Configuring PKI infrastructure..."
    
    # Generate root CA certificate
    vault write pki/root/generate/internal \
        common_name="ARES ChronoFabric Root CA" \
        organization="ARES Systems" \
        country="US" \
        locality="San Francisco" \
        province="California" \
        ttl=8760h || log "Root CA already exists"
    
    # Configure CA and CRL URLs
    vault write pki/config/urls \
        issuing_certificates="$VAULT_ADDR/v1/pki/ca" \
        crl_distribution_points="$VAULT_ADDR/v1/pki/crl"
    
    # Create role for ARES services
    vault write pki/roles/ares-csf \
        allowed_domains="ares-internal.com,ares-csf.com,localhost" \
        allow_subdomains=true \
        allow_localhost=true \
        allow_ip_sans=true \
        max_ttl=720h \
        ttl=168h \
        generate_lease=true
    
    # Development PKI setup
    vault write pki_dev/root/generate/internal \
        common_name="ARES ChronoFabric Dev CA" \
        organization="ARES Systems Development" \
        ttl=720h || log "Dev Root CA already exists"
    
    vault write pki_dev/config/urls \
        issuing_certificates="$VAULT_ADDR/v1/pki_dev/ca" \
        crl_distribution_points="$VAULT_ADDR/v1/pki_dev/crl"
    
    vault write pki_dev/roles/ares-csf-dev \
        allowed_domains="localhost,127.0.0.1,dev.ares-internal.com" \
        allow_subdomains=true \
        allow_localhost=true \
        allow_ip_sans=true \
        max_ttl=168h \
        ttl=24h
    
    log "PKI infrastructure configured"
}

configure_database_secrets() {
    log "Configuring database secret engine..."
    
    # PostgreSQL connection for main database
    vault write database/config/ares-csf-postgres \
        plugin_name=postgresql-database-plugin \
        connection_url="postgresql://{{username}}:{{password}}@postgres.ares-internal.com:5432/ares_chronofabric?sslmode=require" \
        allowed_roles="ares-csf-readonly,ares-csf-readwrite,ares-csf-admin" \
        username="vault_admin" \
        password="$POSTGRES_VAULT_PASSWORD"
    
    # Read-only database role
    vault write database/roles/ares-csf-readonly \
        db_name=ares-csf-postgres \
        creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
                           GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
        default_ttl="1h" \
        max_ttl="24h"
    
    # Read-write database role
    vault write database/roles/ares-csf-readwrite \
        db_name=ares-csf-postgres \
        creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
                           GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
        default_ttl="4h" \
        max_ttl="12h"
    
    # Admin database role (limited use)
    vault write database/roles/ares-csf-admin \
        db_name=ares-csf-postgres \
        creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}' SUPERUSER;" \
        default_ttl="15m" \
        max_ttl="1h"
    
    # Redis connection
    vault write database/config/ares-csf-redis \
        plugin_name=redis-database-plugin \
        connection_url="redis://redis.ares-internal.com:6379" \
        allowed_roles="ares-csf-redis" \
        username="vault_admin" \
        password="$REDIS_VAULT_PASSWORD"
    
    vault write database/roles/ares-csf-redis \
        db_name=ares-csf-redis \
        creation_statements='["ACL SETUSER \"{{name}}\" ON >\"{{password}}\" +@read +@write -@dangerous ~* &*"]' \
        default_ttl="2h" \
        max_ttl="8h"
    
    log "Database secrets configured"
}

configure_transit_encryption() {
    log "Configuring transit encryption..."
    
    # Create encryption key for ARES ChronoFabric
    vault write transit/keys/ares-csf \
        type=aes256-gcm96 \
        exportable=false \
        allow_plaintext_backup=false \
        deletion_allowed=false
    
    # Create quantum-specific encryption key
    vault write transit/keys/ares-quantum \
        type=chacha20-poly1305 \
        exportable=false \
        allow_plaintext_backup=false \
        deletion_allowed=false
    
    # Create customer data encryption key
    vault write transit/keys/ares-customer-data \
        type=aes256-gcm96 \
        exportable=false \
        allow_plaintext_backup=false \
        deletion_allowed=false \
        min_decryption_version=1 \
        min_encryption_version=0
    
    log "Transit encryption configured"
}

setup_authentication_methods() {
    log "Setting up authentication methods..."
    
    # Kubernetes authentication
    vault auth enable kubernetes || log "Kubernetes auth already enabled"
    
    vault write auth/kubernetes/config \
        token_reviewer_jwt="$KUBERNETES_SA_TOKEN" \
        kubernetes_host="https://kubernetes.default.svc.cluster.local" \
        kubernetes_ca_cert="$KUBERNETES_CA_CERT"
    
    # Create Kubernetes roles for different ARES services
    vault write auth/kubernetes/role/ares-csf-core \
        bound_service_account_names=ares-csf \
        bound_service_account_namespaces=ares-production \
        policies=ares-application \
        ttl=1h \
        max_ttl=4h
    
    vault write auth/kubernetes/role/ares-csf-admin \
        bound_service_account_names=ares-csf-admin \
        bound_service_account_namespaces=ares-production \
        policies=ares-administration \
        ttl=30m \
        max_ttl=2h
    
    vault write auth/kubernetes/role/ares-csf-monitoring \
        bound_service_account_names=ares-monitoring \
        bound_service_account_namespaces=ares-monitoring \
        policies=ares-read-only \
        ttl=2h \
        max_ttl=8h
    
    # JWT/OIDC authentication for user access
    vault auth enable jwt || log "JWT auth already enabled"
    
    vault write auth/jwt/config \
        bound_issuer="https://auth.ares-systems.com" \
        oidc_discovery_url="https://auth.ares-systems.com/.well-known/openid_configuration"
    
    # JWT role for ARES employees
    vault write auth/jwt/role/ares-employee \
        user_claim="email" \
        bound_claims='{"groups":["ares-employees"]}' \
        policies=ares-read-only \
        ttl=8h \
        max_ttl=24h
    
    # JWT role for ARES administrators
    vault write auth/jwt/role/ares-admin \
        user_claim="email" \
        bound_claims='{"groups":["ares-admins"]}' \
        policies=ares-administration \
        ttl=4h \
        max_ttl=12h
    
    log "Authentication methods configured"
}

create_vault_policies() {
    log "Creating Vault policies..."
    
    # Application policy
    vault policy write ares-application - << EOF
# ARES Application Policy
path "secret/data/ares-csf/*" {
  capabilities = ["read"]
}

path "secret/metadata/ares-csf/*" {
  capabilities = ["read", "list"]
}

path "database/creds/ares-csf-readonly" {
  capabilities = ["read"]
}

path "database/creds/ares-csf-readwrite" {
  capabilities = ["read"]
}

path "pki/issue/ares-csf" {
  capabilities = ["update"]
}

path "transit/encrypt/ares-csf" {
  capabilities = ["update"]
}

path "transit/decrypt/ares-csf" {
  capabilities = ["update"]
}
EOF

    # Administration policy
    vault policy write ares-administration - << EOF
# ARES Administration Policy
path "secret/data/ares-csf/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/ares-csf/*" {
  capabilities = ["read", "list", "delete"]
}

path "database/config/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "database/roles/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "pki/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "transit/keys/ares-csf" {
  capabilities = ["create", "read", "update"]
}

path "auth/kubernetes/role/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
EOF

    # Security team policy
    vault policy write ares-security - << EOF
# ARES Security Team Policy
path "secret/data/ares-csf/security/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "pki/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "transit/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "identity/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "auth/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "sys/audit" {
  capabilities = ["read", "list"]
}

path "sys/audit/*" {
  capabilities = ["create", "read", "update", "delete"]
}
EOF

    # Read-only policy for monitoring
    vault policy write ares-read-only - << EOF
# ARES Read-Only Policy
path "secret/data/ares-csf/*" {
  capabilities = ["read"]
}

path "secret/metadata/ares-csf/*" {
  capabilities = ["read", "list"]
}

path "sys/health" {
  capabilities = ["read"]
}

path "sys/metrics" {
  capabilities = ["read"]
}
EOF

    log "Vault policies created"
}

setup_ares_secrets() {
    log "Setting up ARES ChronoFabric secrets..."
    
    # Monitoring secrets
    vault kv put secret/ares-csf/monitoring/datadog \
        api_key="$DATADOG_API_KEY" \
        app_key="$DATADOG_APP_KEY" \
        site="datadoghq.com"
    
    vault kv put secret/ares-csf/monitoring/alerting \
        slack_webhook="$SLACK_WEBHOOK_URL" \
        pagerduty_key="$PAGERDUTY_INTEGRATION_KEY"
    
    # Database secrets
    vault kv put secret/ares-csf/database/postgres \
        host="postgres.ares-internal.com" \
        port="5432" \
        database="ares_chronofabric" \
        ssl_mode="require"
    
    vault kv put secret/ares-csf/database/redis \
        host="redis.ares-internal.com" \
        port="6379" \
        ssl_enabled="true"
    
    # API secrets
    vault kv put secret/ares-csf/api/jwt \
        signing_key="$(openssl rand -base64 64)" \
        algorithm="HS256" \
        expiry="24h"
    
    vault kv put secret/ares-csf/api/encryption \
        master_key="$(openssl rand -base64 32)" \
        algorithm="AES-256-GCM"
    
    # External service credentials
    vault kv put secret/ares-csf/external/github \
        api_token="$GITHUB_API_TOKEN" \
        webhook_secret="$(openssl rand -hex 32)"
    
    # Container registry credentials
    vault kv put secret/ares-csf/registry/ghcr \
        username="ares-systems" \
        token="$GITHUB_CONTAINER_TOKEN"
    
    # Cloud provider credentials
    if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
        vault kv put secret/ares-csf/aws/credentials \
            access_key_id="$AWS_ACCESS_KEY_ID" \
            secret_access_key="$AWS_SECRET_ACCESS_KEY" \
            region="us-east-1"
    fi
    
    if [ -n "${GCP_SERVICE_ACCOUNT_KEY:-}" ]; then
        vault kv put secret/ares-csf/gcp/credentials \
            service_account_key="$GCP_SERVICE_ACCOUNT_KEY" \
            project_id="ares-chronofabric"
    fi
    
    # Quantum hardware access (if available)
    if [ -n "${IBM_QUANTUM_TOKEN:-}" ]; then
        vault kv put secret/ares-csf/quantum-hardware/ibm \
            api_token="$IBM_QUANTUM_TOKEN" \
            hub="ibm-q" \
            group="open" \
            project="main"
    fi
    
    if [ -n "${AWS_BRAKET_ACCESS_KEY:-}" ]; then
        vault kv put secret/ares-csf/quantum-hardware/aws-braket \
            access_key_id="$AWS_BRAKET_ACCESS_KEY" \
            secret_access_key="$AWS_BRAKET_SECRET_KEY" \
            region="us-east-1"
    fi
    
    # Security scanning tokens
    vault kv put secret/ares-csf/security/scanners \
        snyk_token="$SNYK_TOKEN" \
        sonarqube_token="$SONARQUBE_TOKEN" \
        veracode_api_id="$VERACODE_API_ID" \
        veracode_api_key="$VERACODE_API_KEY"
    
    log "ARES secrets configured"
}

setup_transit_keys() {
    log "Setting up transit encryption keys..."
    
    # Main application encryption key
    vault write transit/keys/ares-csf \
        type=aes256-gcm96 \
        exportable=false \
        allow_plaintext_backup=false \
        deletion_allowed=false
    
    # Quantum state encryption key
    vault write transit/keys/ares-quantum \
        type=chacha20-poly1305 \
        exportable=false \
        allow_plaintext_backup=false \
        deletion_allowed=false
    
    # Customer data encryption key (with versioning)
    vault write transit/keys/ares-customer-data \
        type=aes256-gcm96 \
        exportable=false \
        allow_plaintext_backup=false \
        deletion_allowed=false \
        min_decryption_version=1 \
        min_encryption_version=0
    
    # Audit log encryption key
    vault write transit/keys/ares-audit \
        type=aes256-gcm96 \
        exportable=false \
        allow_plaintext_backup=false \
        deletion_allowed=false
    
    log "Transit encryption keys created"
}

configure_secret_rotation() {
    log "Configuring automatic secret rotation..."
    
    # Configure rotation for database credentials
    vault write sys/rotate/config \
        enabled=true \
        rotation_period="720h"  # 30 days
    
    # Set up rotation for specific secrets
    vault write sys/rotate/database/ares-csf-postgres \
        auto_rotate=true \
        rotation_period="2160h"  # 90 days
    
    # Set up certificate rotation
    vault write pki/config/auto-tidy \
        enabled=true \
        tidy_cert_store=true \
        tidy_revoked_certs=true \
        safety_buffer="72h"
    
    log "Secret rotation configured"
}

setup_audit_logging() {
    log "Setting up audit logging..."
    
    # Enable file audit logging
    vault audit enable file file_path=/var/log/vault/audit.log || log "File audit already enabled"
    
    # Enable syslog audit logging (optional)
    vault audit enable syslog tag="vault" facility="LOCAL0" || log "Syslog audit already enabled"
    
    log "Audit logging configured"
}

validate_vault_setup() {
    log "Validating Vault setup..."
    
    # Test secret read/write
    vault kv put secret/ares-csf/test/validation test_key="test_value"
    TEST_VALUE=$(vault kv get -field=test_key secret/ares-csf/test/validation)
    
    if [ "$TEST_VALUE" != "test_value" ]; then
        error "Vault secret read/write validation failed"
    fi
    
    # Clean up test secret
    vault kv delete secret/ares-csf/test/validation
    
    # Test database credentials
    DB_CREDS=$(vault read database/creds/ares-csf-readonly -format=json)
    if [ -z "$DB_CREDS" ]; then
        error "Database credential generation failed"
    fi
    
    # Test certificate generation
    CERT=$(vault write pki/issue/ares-csf common_name="test.ares-internal.com" -format=json)
    if [ -z "$CERT" ]; then
        error "Certificate generation failed"
    fi
    
    # Test transit encryption
    ENCRYPTED=$(vault write transit/encrypt/ares-csf plaintext=$(echo -n "test data" | base64) -field=ciphertext)
    DECRYPTED=$(vault write transit/decrypt/ares-csf ciphertext="$ENCRYPTED" -field=plaintext | base64 -d)
    
    if [ "$DECRYPTED" != "test data" ]; then
        error "Transit encryption validation failed"
    fi
    
    log "Vault setup validation completed successfully"
}

generate_vault_operator_init_script() {
    log "Generating Vault operator initialization script..."
    
    cat > "$REPO_ROOT/scripts/vault-operator-init.sh" << 'EOF'
#!/bin/bash

# ARES ChronoFabric Vault Operator Initialization Script
# Run this script to initialize Vault in a production environment

set -euo pipefail

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

if [ ! -f "/tmp/vault-init.json" ]; then
    log "Initializing Vault cluster..."
    
    vault operator init \
        -key-shares=5 \
        -key-threshold=3 \
        -format=json > /tmp/vault-init.json
    
    log "Vault initialized. CRITICAL: Securely store /tmp/vault-init.json"
    log "Root token and unseal keys are in this file"
fi

if [ -f "/tmp/vault-init.json" ]; then
    log "Unsealing Vault with stored keys..."
    
    UNSEAL_KEY_1=$(jq -r '.unseal_keys_b64[0]' /tmp/vault-init.json)
    UNSEAL_KEY_2=$(jq -r '.unseal_keys_b64[1]' /tmp/vault-init.json)
    UNSEAL_KEY_3=$(jq -r '.unseal_keys_b64[2]' /tmp/vault-init.json)
    
    vault operator unseal "$UNSEAL_KEY_1"
    vault operator unseal "$UNSEAL_KEY_2"
    vault operator unseal "$UNSEAL_KEY_3"
    
    export VAULT_TOKEN=$(jq -r '.root_token' /tmp/vault-init.json)
    
    log "Vault unsealed and ready for configuration"
    log "Run setup-vault-secrets.sh to complete the setup"
fi
EOF
    
    chmod +x "$REPO_ROOT/scripts/vault-operator-init.sh"
    
    log "Vault operator initialization script created"
}

create_kubernetes_vault_resources() {
    log "Creating Kubernetes resources for Vault..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vault
  namespace: $VAULT_NAMESPACE
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: vault-config
  namespace: $VAULT_NAMESPACE
data:
  vault.hcl: |
$(cat "$SECRETS_DIR/vault-config.hcl" | sed 's/^/    /')
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vault
  namespace: $VAULT_NAMESPACE
  labels:
    app: vault
spec:
  serviceName: vault
  replicas: 3
  selector:
    matchLabels:
      app: vault
  template:
    metadata:
      labels:
        app: vault
    spec:
      serviceAccountName: vault
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: vault
        image: hashicorp/vault:1.15
        args:
        - "vault"
        - "server"
        - "-config=/vault/config/vault.hcl"
        ports:
        - containerPort: 8200
          name: api
        - containerPort: 8201
          name: cluster
        env:
        - name: VAULT_ADDR
          value: "https://127.0.0.1:8200"
        - name: VAULT_API_ADDR
          value: "https://vault.ares-internal.com:8200"
        - name: VAULT_CLUSTER_ADDR
          value: "https://vault.ares-internal.com:8201"
        - name: VAULT_LOG_LEVEL
          value: "INFO"
        - name: VAULT_LOG_FORMAT
          value: "json"
        resources:
          requests:
            memory: 1Gi
            cpu: 500m
          limits:
            memory: 2Gi
            cpu: 1
        volumeMounts:
        - name: vault-config
          mountPath: /vault/config
        - name: vault-data
          mountPath: /vault/data
        - name: vault-tls
          mountPath: /etc/vault/tls
        livenessProbe:
          httpGet:
            path: /v1/sys/health
            port: 8200
            scheme: HTTPS
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/sys/health?standbyok=true
            port: 8200
            scheme: HTTPS
          initialDelaySeconds: 30
          periodSeconds: 5
      volumes:
      - name: vault-config
        configMap:
          name: vault-config
      - name: vault-tls
        secret:
          secretName: vault-tls
  volumeClaimTemplates:
  - metadata:
      name: vault-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vault
  namespace: $VAULT_NAMESPACE
  labels:
    app: vault
spec:
  type: LoadBalancer
  ports:
  - name: api
    port: 8200
    targetPort: 8200
  - name: cluster
    port: 8201
    targetPort: 8201
  selector:
    app: vault
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vault-ingress
  namespace: $VAULT_NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/backend-protocol: HTTPS
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - vault.ares-internal.com
    secretName: vault-tls-ingress
  rules:
  - host: vault.ares-internal.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vault
            port:
              number: 8200
EOF

    log "Kubernetes Vault resources created"
}

main() {
    log "Starting ARES ChronoFabric Vault secrets management setup..."
    
    check_vault_prerequisites
    enable_secret_engines
    configure_pki_infrastructure
    configure_database_secrets
    configure_transit_encryption
    setup_authentication_methods
    create_vault_policies
    setup_ares_secrets
    setup_audit_logging
    configure_secret_rotation
    validate_vault_setup
    generate_vault_operator_init_script
    create_kubernetes_vault_resources
    
    log "Vault secrets management setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Securely store the Vault initialization keys from /tmp/vault-init.json"
    log "2. Set up automated backup of Vault data"
    log "3. Configure monitoring alerts for Vault health"
    log "4. Test secret rotation procedures"
    log "5. Train operators on emergency unseal procedures"
    log ""
    log "Vault is now ready for enterprise secret management."
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi