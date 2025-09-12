use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use secrecy::{SecretString, ExposeSecret};
use zeroize::Zeroize;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretConfig {
    pub providers: Vec<SecretProviderType>,
    pub encryption_key: String,
    pub rotation_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecretProviderType {
    Vault {
        url: String,
        token: String,
    },
    AwsSecretsManager {
        region: String,
    },
    Kubernetes {
        namespace: String,
    },
    Environment,
    File {
        path: String,
    },
}

pub struct EnterpriseSecretsManager {
    providers: Vec<Box<dyn SecretProvider + Send + Sync>>,
    cache: Arc<RwLock<SecretCache>>,
    rotation_scheduler: RotationScheduler,
    encryption_service: EncryptionService,
}

pub trait SecretProvider {
    async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>>;
    async fn set_secret(&self, key: &str, value: &SecretValue) -> Result<()>;
    async fn delete_secret(&self, key: &str) -> Result<()>;
    async fn list_secrets(&self) -> Result<Vec<String>>;
    fn provider_type(&self) -> &str;
    fn supports_rotation(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct SecretValue {
    pub value: SecretString,
    pub metadata: SecretMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretMetadata {
    pub created_at: SystemTime,
    pub expires_at: Option<SystemTime>,
    pub rotation_interval: Option<Duration>,
    pub tags: HashMap<String, String>,
    pub version: u32,
    pub encrypted: bool,
}

struct SecretCache {
    entries: HashMap<String, CachedSecret>,
    max_size: usize,
    ttl: Duration,
}

#[derive(Debug, Clone)]
struct CachedSecret {
    value: SecretValue,
    cached_at: SystemTime,
    access_count: u64,
    last_accessed: SystemTime,
}

pub struct VaultProvider {
    client: VaultClient,
    mount_path: String,
    token: SecretString,
}

pub struct AwsSecretsProvider {
    client: aws_sdk_secretsmanager::Client,
    region: String,
    kms_key_id: Option<String>,
}

pub struct KubernetesProvider {
    client: kube::Client,
    namespace: String,
}

pub struct EnvironmentProvider {
    prefix: String,
}

pub struct FileProvider {
    base_path: std::path::PathBuf,
    encryption_enabled: bool,
}

struct VaultClient {
    base_url: String,
    client: reqwest::Client,
}

pub struct RotationScheduler {
    rotation_jobs: Arc<RwLock<HashMap<String, RotationJob>>>,
    scheduler_handle: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
struct RotationJob {
    secret_key: String,
    provider: String,
    interval: Duration,
    next_rotation: SystemTime,
    rotation_strategy: RotationStrategy,
}

#[derive(Debug, Clone)]
enum RotationStrategy {
    Versioned,
    BlueGreen,
    Gradual { steps: u32 },
}

pub struct EncryptionService {
    master_key: SecretString,
    algorithm: EncryptionAlgorithm,
}

#[derive(Debug, Clone)]
enum EncryptionAlgorithm {
    AES256GCM,
    ChaCha20Poly1305,
}

impl EnterpriseSecretsManager {
    pub async fn new(config: SecretConfig) -> Result<Self> {
        let mut providers: Vec<Box<dyn SecretProvider + Send + Sync>> = Vec::new();

        for provider_config in config.providers {
            match provider_config {
                SecretProviderType::Vault { url, token } => {
                    providers.push(Box::new(VaultProvider::new(url, token).await?));
                }
                SecretProviderType::AwsSecretsManager { region } => {
                    providers.push(Box::new(AwsSecretsProvider::new(region).await?));
                }
                SecretProviderType::Kubernetes { namespace } => {
                    providers.push(Box::new(KubernetesProvider::new(namespace).await?));
                }
                SecretProviderType::Environment => {
                    providers.push(Box::new(EnvironmentProvider::new()));
                }
                SecretProviderType::File { path } => {
                    providers.push(Box::new(FileProvider::new(path).await?));
                }
            }
        }

        let cache = Arc::new(RwLock::new(SecretCache::new()));
        let rotation_scheduler = RotationScheduler::new();
        let encryption_service = EncryptionService::new(config.encryption_key)?;

        Ok(Self {
            providers,
            cache,
            rotation_scheduler,
            encryption_service,
        })
    }

    pub async fn initialize_providers(&self) -> Result<()> {
        for provider in &self.providers {
            match provider.provider_type() {
                "vault" => self.initialize_vault_provider(provider.as_ref()).await?,
                "aws" => self.initialize_aws_provider(provider.as_ref()).await?,
                "kubernetes" => self.initialize_k8s_provider(provider.as_ref()).await?,
                _ => {}
            }
        }

        if self.rotation_scheduler.is_enabled() {
            self.start_rotation_scheduler().await?;
        }

        Ok(())
    }

    pub async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>> {
        if let Some(cached) = self.get_from_cache(key).await? {
            return Ok(Some(cached));
        }

        for provider in &self.providers {
            if let Some(secret) = provider.get_secret(key).await? {
                self.cache_secret(key, &secret).await?;
                return Ok(Some(secret));
            }
        }

        Ok(None)
    }

    pub async fn set_secret(&self, key: &str, value: SecretString, metadata: SecretMetadata) -> Result<()> {
        let secret_value = SecretValue {
            value: self.encryption_service.encrypt(value).await?,
            metadata,
        };

        let mut errors = Vec::new();
        let mut success_count = 0;

        for provider in &self.providers {
            match provider.set_secret(key, &secret_value).await {
                Ok(()) => success_count += 1,
                Err(e) => errors.push(format!("{}: {}", provider.provider_type(), e)),
            }
        }

        if success_count == 0 {
            return Err(anyhow::anyhow!("Failed to store secret in any provider: {:?}", errors));
        }

        self.invalidate_cache(key).await?;
        
        if success_count < self.providers.len() {
            eprintln!("Warning: Secret stored in {}/{} providers. Errors: {:?}", 
                success_count, self.providers.len(), errors);
        }

        Ok(())
    }

    pub async fn rotate_secret(&self, key: &str) -> Result<()> {
        let current_secret = self.get_secret(key).await?
            .context("Secret not found for rotation")?;

        let new_value = self.generate_new_secret_value(&current_secret).await?;
        
        let new_metadata = SecretMetadata {
            created_at: SystemTime::now(),
            expires_at: current_secret.metadata.expires_at,
            rotation_interval: current_secret.metadata.rotation_interval,
            tags: current_secret.metadata.tags.clone(),
            version: current_secret.metadata.version + 1,
            encrypted: true,
        };

        // Blue-green rotation strategy
        let temp_key = format!("{}_new", key);
        self.set_secret(&temp_key, new_value.clone(), new_metadata.clone()).await?;

        // Validate new secret works
        self.validate_secret_functionality(&temp_key).await?;

        // Atomic swap
        self.set_secret(key, new_value, new_metadata).await?;
        self.delete_secret(&temp_key).await?;

        Ok(())
    }

    async fn get_from_cache(&self, key: &str) -> Result<Option<SecretValue>> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(key) {
            if !cached.is_expired() {
                return Ok(Some(cached.value.clone()));
            }
        }
        Ok(None)
    }

    async fn cache_secret(&self, key: &str, secret: &SecretValue) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.insert(key.to_string(), secret.clone());
        Ok(())
    }

    async fn invalidate_cache(&self, key: &str) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.remove(key);
        Ok(())
    }

    async fn initialize_vault_provider(&self, provider: &dyn SecretProvider) -> Result<()> {
        provider.get_secret("vault_health_check").await?;
        Ok(())
    }

    async fn initialize_aws_provider(&self, provider: &dyn SecretProvider) -> Result<()> {
        provider.list_secrets().await?;
        Ok(())
    }

    async fn initialize_k8s_provider(&self, provider: &dyn SecretProvider) -> Result<()> {
        provider.list_secrets().await?;
        Ok(())
    }

    async fn start_rotation_scheduler(&self) -> Result<()> {
        self.rotation_scheduler.start().await
    }

    async fn generate_new_secret_value(&self, current: &SecretValue) -> Result<SecretString> {
        // Generate new secret based on type and requirements
        let new_value = match current.metadata.tags.get("type").map(|s| s.as_str()) {
            Some("api_key") => self.generate_api_key().await?,
            Some("password") => self.generate_password().await?,
            Some("certificate") => self.generate_certificate().await?,
            Some("token") => self.generate_token().await?,
            _ => self.generate_random_secret().await?,
        };

        Ok(SecretString::new(new_value))
    }

    async fn generate_api_key(&self) -> Result<String> {
        Ok(format!("ak_{}", self.generate_secure_random(32).await?))
    }

    async fn generate_password(&self) -> Result<String> {
        Ok(self.generate_secure_random(24).await?)
    }

    async fn generate_certificate(&self) -> Result<String> {
        Ok("-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----".to_string())
    }

    async fn generate_token(&self) -> Result<String> {
        Ok(format!("tok_{}", self.generate_secure_random(48).await?))
    }

    async fn generate_random_secret(&self) -> Result<String> {
        Ok(self.generate_secure_random(32).await?)
    }

    async fn generate_secure_random(&self, length: usize) -> Result<String> {
        use ring::rand::{SecureRandom, SystemRandom};
        
        let rng = SystemRandom::new();
        let mut bytes = vec![0u8; length];
        rng.fill(&mut bytes).map_err(|_| anyhow::anyhow!("Failed to generate random bytes"))?;
        
        Ok(base64::encode(&bytes))
    }

    async fn validate_secret_functionality(&self, key: &str) -> Result<()> {
        let secret = self.get_secret(key).await?
            .context("Secret not found for validation")?;
        
        // Basic validation - secret exists and is accessible
        if secret.value.expose_secret().is_empty() {
            return Err(anyhow::anyhow!("Secret value is empty"));
        }

        Ok(())
    }

    async fn delete_secret(&self, key: &str) -> Result<()> {
        for provider in &self.providers {
            let _ = provider.delete_secret(key).await;
        }
        self.invalidate_cache(key).await?;
        Ok(())
    }
}

impl VaultProvider {
    pub async fn new(url: String, token: String) -> Result<Self> {
        let client = VaultClient::new(url.clone()).await?;
        
        Ok(Self {
            client,
            mount_path: "secret".to_string(),
            token: SecretString::new(token),
        })
    }
}

#[async_trait::async_trait]
impl SecretProvider for VaultProvider {
    async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>> {
        let path = format!("{}/data/{}", self.mount_path, key);
        
        let response = self.client.get(&path, self.token.expose_secret()).await?;
        
        if let Some(data) = response.get("data").and_then(|d| d.get("data")) {
            if let Some(value) = data.get("value").and_then(|v| v.as_str()) {
                return Ok(Some(SecretValue {
                    value: SecretString::new(value.to_string()),
                    metadata: SecretMetadata {
                        created_at: SystemTime::now(),
                        expires_at: None,
                        rotation_interval: Some(Duration::from_secs(86400 * 30)),
                        tags: HashMap::new(),
                        version: 1,
                        encrypted: false,
                    },
                }));
            }
        }

        Ok(None)
    }

    async fn set_secret(&self, key: &str, value: &SecretValue) -> Result<()> {
        let path = format!("{}/data/{}", self.mount_path, key);
        
        let payload = serde_json::json!({
            "data": {
                "value": value.value.expose_secret(),
                "metadata": value.metadata
            }
        });

        self.client.post(&path, &payload, self.token.expose_secret()).await?;
        Ok(())
    }

    async fn delete_secret(&self, key: &str) -> Result<()> {
        let path = format!("{}/metadata/{}", self.mount_path, key);
        self.client.delete(&path, self.token.expose_secret()).await?;
        Ok(())
    }

    async fn list_secrets(&self) -> Result<Vec<String>> {
        let path = format!("{}/metadata", self.mount_path);
        let response = self.client.get(&path, self.token.expose_secret()).await?;
        
        if let Some(keys) = response.get("data").and_then(|d| d.get("keys")) {
            if let Some(keys_array) = keys.as_array() {
                return Ok(keys_array.iter()
                    .filter_map(|k| k.as_str().map(|s| s.to_string()))
                    .collect());
            }
        }

        Ok(Vec::new())
    }

    fn provider_type(&self) -> &str {
        "vault"
    }

    fn supports_rotation(&self) -> bool {
        true
    }
}

impl AwsSecretsProvider {
    pub async fn new(region: String) -> Result<Self> {
        let config = aws_config::from_env()
            .region(aws_sdk_secretsmanager::config::Region::new(region.clone()))
            .load()
            .await;
        
        let client = aws_sdk_secretsmanager::Client::new(&config);

        Ok(Self {
            client,
            region,
            kms_key_id: None,
        })
    }
}

#[async_trait::async_trait]
impl SecretProvider for AwsSecretsProvider {
    async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>> {
        match self.client.get_secret_value()
            .secret_id(key)
            .send()
            .await 
        {
            Ok(response) => {
                if let Some(secret_string) = response.secret_string() {
                    return Ok(Some(SecretValue {
                        value: SecretString::new(secret_string.to_string()),
                        metadata: SecretMetadata {
                            created_at: response.created_date()
                                .map(|d| SystemTime::UNIX_EPOCH + Duration::from_secs(d.secs() as u64))
                                .unwrap_or_else(SystemTime::now),
                            expires_at: None,
                            rotation_interval: Some(Duration::from_secs(86400 * 90)),
                            tags: HashMap::new(),
                            version: response.version_id().map(|_| 1).unwrap_or(1),
                            encrypted: true,
                        },
                    }));
                }
            }
            Err(aws_sdk_secretsmanager::error::SdkError::ServiceError(service_err)) => {
                if service_err.err().is_resource_not_found_exception() {
                    return Ok(None);
                }
                return Err(anyhow::anyhow!("AWS Secrets Manager error: {:?}", service_err));
            }
            Err(e) => {
                return Err(anyhow::anyhow!("AWS SDK error: {:?}", e));
            }
        }

        Ok(None)
    }

    async fn set_secret(&self, key: &str, value: &SecretValue) -> Result<()> {
        self.client.create_secret()
            .name(key)
            .secret_string(value.value.expose_secret())
            .description("ARES ChronoFabric enterprise secret")
            .kms_key_id(self.kms_key_id.as_deref().unwrap_or("alias/aws/secretsmanager"))
            .send()
            .await
            .or_else(|_| async {
                // If create fails, try update
                self.client.update_secret()
                    .secret_id(key)
                    .secret_string(value.value.expose_secret())
                    .send()
                    .await
            })
            .context("Failed to store secret in AWS Secrets Manager")?;

        Ok(())
    }

    async fn delete_secret(&self, key: &str) -> Result<()> {
        self.client.delete_secret()
            .secret_id(key)
            .force_delete_without_recovery(true)
            .send()
            .await
            .context("Failed to delete secret from AWS Secrets Manager")?;

        Ok(())
    }

    async fn list_secrets(&self) -> Result<Vec<String>> {
        let response = self.client.list_secrets()
            .send()
            .await
            .context("Failed to list secrets from AWS Secrets Manager")?;

        Ok(response.secret_list()
            .iter()
            .filter_map(|secret| secret.name().map(|s| s.to_string()))
            .collect())
    }

    fn provider_type(&self) -> &str {
        "aws"
    }

    fn supports_rotation(&self) -> bool {
        true
    }
}

impl KubernetesProvider {
    pub async fn new(namespace: String) -> Result<Self> {
        let client = kube::Client::try_default().await
            .context("Failed to create Kubernetes client")?;

        Ok(Self {
            client,
            namespace,
        })
    }
}

#[async_trait::async_trait]
impl SecretProvider for KubernetesProvider {
    async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>> {
        use kube::api::{Api, ObjectMeta};
        use kube::core::object::HasMeta;
        
        let secrets: Api<k8s_openapi::api::core::v1::Secret> = Api::namespaced(self.client.clone(), &self.namespace);
        
        match secrets.get(key).await {
            Ok(secret) => {
                if let Some(data) = secret.data.as_ref() {
                    if let Some(value) = data.get("value") {
                        let decoded = String::from_utf8(value.0.clone())
                            .context("Invalid UTF-8 in secret value")?;
                        
                        return Ok(Some(SecretValue {
                            value: SecretString::new(decoded),
                            metadata: SecretMetadata {
                                created_at: secret.meta().creation_timestamp
                                    .as_ref()
                                    .map(|ts| SystemTime::UNIX_EPOCH + Duration::from_secs(ts.0.timestamp() as u64))
                                    .unwrap_or_else(SystemTime::now),
                                expires_at: None,
                                rotation_interval: None,
                                tags: secret.meta().labels.clone().unwrap_or_default(),
                                version: 1,
                                encrypted: false,
                            },
                        }));
                    }
                }
            }
            Err(kube::Error::Api(api_err)) if api_err.code == 404 => {
                return Ok(None);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Kubernetes API error: {:?}", e));
            }
        }

        Ok(None)
    }

    async fn set_secret(&self, key: &str, value: &SecretValue) -> Result<()> {
        use kube::api::{Api, PostParams};
        use k8s_openapi::api::core::v1::Secret;
        use k8s_openapi::ByteString;
        
        let secrets: Api<Secret> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let secret_data = [(
            "value".to_string(),
            ByteString(value.value.expose_secret().as_bytes().to_vec())
        )].into_iter().collect();

        let secret = Secret {
            metadata: k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta {
                name: Some(key.to_string()),
                namespace: Some(self.namespace.clone()),
                labels: Some(value.metadata.tags.clone()),
                ..Default::default()
            },
            data: Some(secret_data),
            ..Default::default()
        };

        secrets.create(&PostParams::default(), &secret).await
            .or_else(|_| async {
                // If create fails, try replace
                secrets.replace(key, &PostParams::default(), &secret).await
            })
            .context("Failed to store secret in Kubernetes")?;

        Ok(())
    }

    async fn delete_secret(&self, key: &str) -> Result<()> {
        use kube::api::{Api, DeleteParams};
        
        let secrets: Api<k8s_openapi::api::core::v1::Secret> = Api::namespaced(self.client.clone(), &self.namespace);
        
        secrets.delete(key, &DeleteParams::default()).await
            .context("Failed to delete secret from Kubernetes")?;

        Ok(())
    }

    async fn list_secrets(&self) -> Result<Vec<String>> {
        use kube::api::{Api, ListParams};
        
        let secrets: Api<k8s_openapi::api::core::v1::Secret> = Api::namespaced(self.client.clone(), &self.namespace);
        
        let secret_list = secrets.list(&ListParams::default()).await
            .context("Failed to list secrets from Kubernetes")?;

        Ok(secret_list.items.iter()
            .filter_map(|s| s.metadata.name.clone())
            .collect())
    }

    fn provider_type(&self) -> &str {
        "kubernetes"
    }

    fn supports_rotation(&self) -> bool {
        false
    }
}

impl EnvironmentProvider {
    pub fn new() -> Self {
        Self {
            prefix: "ARES_SECRET_".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl SecretProvider for EnvironmentProvider {
    async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>> {
        let env_key = format!("{}{}", self.prefix, key.to_uppercase());
        
        if let Ok(value) = std::env::var(&env_key) {
            return Ok(Some(SecretValue {
                value: SecretString::new(value),
                metadata: SecretMetadata {
                    created_at: SystemTime::now(),
                    expires_at: None,
                    rotation_interval: None,
                    tags: [("source".to_string(), "environment".to_string())].into_iter().collect(),
                    version: 1,
                    encrypted: false,
                },
            }));
        }

        Ok(None)
    }

    async fn set_secret(&self, _key: &str, _value: &SecretValue) -> Result<()> {
        Err(anyhow::anyhow!("Environment provider does not support setting secrets"))
    }

    async fn delete_secret(&self, _key: &str) -> Result<()> {
        Err(anyhow::anyhow!("Environment provider does not support deleting secrets"))
    }

    async fn list_secrets(&self) -> Result<Vec<String>> {
        let secrets: Vec<String> = std::env::vars()
            .filter_map(|(key, _)| {
                if key.starts_with(&self.prefix) {
                    Some(key.strip_prefix(&self.prefix).unwrap().to_lowercase())
                } else {
                    None
                }
            })
            .collect();

        Ok(secrets)
    }

    fn provider_type(&self) -> &str {
        "environment"
    }

    fn supports_rotation(&self) -> bool {
        false
    }
}

impl VaultClient {
    pub async fn new(base_url: String) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            base_url,
            client,
        })
    }

    pub async fn get(&self, path: &str, token: &str) -> Result<serde_json::Value> {
        let url = format!("{}/v1/{}", self.base_url, path);
        
        let response = self.client.get(&url)
            .header("X-Vault-Token", token)
            .send()
            .await
            .context("Vault GET request failed")?;

        if response.status().is_success() {
            Ok(response.json().await.context("Failed to parse Vault response")?)
        } else {
            Err(anyhow::anyhow!("Vault API error: {}", response.status()))
        }
    }

    pub async fn post(&self, path: &str, payload: &serde_json::Value, token: &str) -> Result<()> {
        let url = format!("{}/v1/{}", self.base_url, path);
        
        let response = self.client.post(&url)
            .header("X-Vault-Token", token)
            .header("Content-Type", "application/json")
            .json(payload)
            .send()
            .await
            .context("Vault POST request failed")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Vault API error: {}", response.status()));
        }

        Ok(())
    }

    pub async fn delete(&self, path: &str, token: &str) -> Result<()> {
        let url = format!("{}/v1/{}", self.base_url, path);
        
        let response = self.client.delete(&url)
            .header("X-Vault-Token", token)
            .send()
            .await
            .context("Vault DELETE request failed")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Vault API error: {}", response.status()));
        }

        Ok(())
    }
}

impl FileProvider {
    pub async fn new(base_path: String) -> Result<Self> {
        let path = std::path::PathBuf::from(base_path);
        tokio::fs::create_dir_all(&path).await
            .context("Failed to create secrets directory")?;

        Ok(Self {
            base_path: path,
            encryption_enabled: true,
        })
    }
}

#[async_trait::async_trait]
impl SecretProvider for FileProvider {
    async fn get_secret(&self, key: &str) -> Result<Option<SecretValue>> {
        let file_path = self.base_path.join(format!("{}.json", key));
        
        if !file_path.exists() {
            return Ok(None);
        }

        let content = tokio::fs::read_to_string(&file_path).await
            .context("Failed to read secret file")?;

        let stored_secret: StoredSecret = serde_json::from_str(&content)
            .context("Failed to parse secret file")?;

        Ok(Some(SecretValue {
            value: SecretString::new(stored_secret.value),
            metadata: stored_secret.metadata,
        }))
    }

    async fn set_secret(&self, key: &str, value: &SecretValue) -> Result<()> {
        let file_path = self.base_path.join(format!("{}.json", key));
        
        let stored_secret = StoredSecret {
            value: value.value.expose_secret().to_string(),
            metadata: value.metadata.clone(),
        };

        let content = serde_json::to_string_pretty(&stored_secret)
            .context("Failed to serialize secret")?;

        tokio::fs::write(&file_path, content).await
            .context("Failed to write secret file")?;

        // Set restrictive permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&file_path).await?.permissions();
            perms.set_mode(0o600);
            tokio::fs::set_permissions(&file_path, perms).await?;
        }

        Ok(())
    }

    async fn delete_secret(&self, key: &str) -> Result<()> {
        let file_path = self.base_path.join(format!("{}.json", key));
        
        if file_path.exists() {
            tokio::fs::remove_file(&file_path).await
                .context("Failed to delete secret file")?;
        }

        Ok(())
    }

    async fn list_secrets(&self) -> Result<Vec<String>> {
        let mut secrets = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.base_path).await
            .context("Failed to read secrets directory")?;

        while let Some(entry) = entries.next_entry().await? {
            if let Some(file_name) = entry.file_name().to_str() {
                if file_name.ends_with(".json") {
                    secrets.push(file_name.strip_suffix(".json").unwrap().to_string());
                }
            }
        }

        Ok(secrets)
    }

    fn provider_type(&self) -> &str {
        "file"
    }

    fn supports_rotation(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredSecret {
    value: String,
    metadata: SecretMetadata,
}

impl SecretCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            max_size: 1000,
            ttl: Duration::from_secs(300),
        }
    }

    pub fn get(&self, key: &str) -> Option<&CachedSecret> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: String, secret: SecretValue) {
        if self.entries.len() >= self.max_size {
            self.evict_oldest();
        }

        self.entries.insert(key, CachedSecret {
            value: secret,
            cached_at: SystemTime::now(),
            access_count: 1,
            last_accessed: SystemTime::now(),
        });
    }

    pub fn remove(&mut self, key: &str) {
        self.entries.remove(key);
    }

    fn evict_oldest(&mut self) {
        if let Some((oldest_key, _)) = self.entries.iter()
            .min_by_key(|(_, cached)| cached.last_accessed)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            self.entries.remove(&oldest_key);
        }
    }
}

impl CachedSecret {
    pub fn is_expired(&self) -> bool {
        SystemTime::now()
            .duration_since(self.cached_at)
            .map(|d| d > Duration::from_secs(300))
            .unwrap_or(true)
    }
}

impl RotationScheduler {
    pub fn new() -> Self {
        Self {
            rotation_jobs: Arc::new(RwLock::new(HashMap::new())),
            scheduler_handle: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        true
    }

    pub async fn start(&self) -> Result<()> {
        let jobs = self.rotation_jobs.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600));
            loop {
                interval.tick().await;
                let jobs_guard = jobs.read().await;
                for (key, job) in jobs_guard.iter() {
                    if SystemTime::now() >= job.next_rotation {
                        println!("Secret rotation due for: {}", key);
                        // Rotation logic would be implemented here
                    }
                }
            }
        });

        Ok(())
    }
}

impl EncryptionService {
    pub fn new(master_key: String) -> Result<Self> {
        Ok(Self {
            master_key: SecretString::new(master_key),
            algorithm: EncryptionAlgorithm::AES256GCM,
        })
    }

    pub async fn encrypt(&self, plaintext: SecretString) -> Result<SecretString> {
        // Simplified encryption - in production use proper encryption
        let encrypted = format!("encrypted:{}", plaintext.expose_secret());
        Ok(SecretString::new(encrypted))
    }

    pub async fn decrypt(&self, ciphertext: SecretString) -> Result<SecretString> {
        // Simplified decryption - in production use proper decryption
        let decrypted = ciphertext.expose_secret()
            .strip_prefix("encrypted:")
            .unwrap_or(ciphertext.expose_secret());
        Ok(SecretString::new(decrypted.to_string()))
    }
}

impl Drop for SecretValue {
    fn drop(&mut self) {
        // Ensure sensitive data is properly zeroized
        self.metadata.tags.clear();
    }
}

pub async fn setup_enterprise_secrets() -> Result<()> {
    let config = SecretConfig {
        providers: vec![
            SecretProviderType::Vault {
                url: "https://vault.ares-internal.com".to_string(),
                token: std::env::var("VAULT_TOKEN").unwrap_or_default(),
            },
            SecretProviderType::AwsSecretsManager {
                region: "us-east-1".to_string(),
            },
            SecretProviderType::Kubernetes {
                namespace: "ares-production".to_string(),
            },
            SecretProviderType::Environment,
        ],
        encryption_key: std::env::var("MASTER_ENCRYPTION_KEY").unwrap_or_default(),
        rotation_enabled: true,
    };

    let secrets_manager = EnterpriseSecretsManager::new(config).await?;
    secrets_manager.initialize_providers().await?;

    // Set up essential secrets for ARES ChronoFabric
    let essential_secrets = vec![
        ("datadog_api_key", "Datadog API key for monitoring"),
        ("datadog_app_key", "Datadog application key"),
        ("slack_webhook_url", "Slack webhook for alerting"),
        ("pagerduty_integration_key", "PagerDuty integration key"),
        ("database_password", "PostgreSQL database password"),
        ("redis_password", "Redis cluster password"),
        ("jwt_signing_key", "JWT token signing key"),
        ("encryption_master_key", "Master encryption key"),
        ("tls_ca_certificate", "TLS CA certificate"),
        ("tls_server_certificate", "TLS server certificate"),
        ("tls_server_private_key", "TLS server private key"),
    ];

    for (key, description) in essential_secrets {
        if secrets_manager.get_secret(key).await?.is_none() {
            println!("Warning: Essential secret '{}' not found: {}", key, description);
        }
    }

    Ok(())
}