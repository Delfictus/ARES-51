//! Security implementation for network communication

use super::*;
use ring::signature::{self, KeyPair};
use ring::{aead, rand};
use rustls::Certificate;

/// Security manager
pub struct SecurityManager {
    config: SecurityConfig,
    private_key: Option<signature::Ed25519KeyPair>,
    certificate: Option<Certificate>,
    encryption_key: Option<aead::LessSafeKey>,
}

impl SecurityManager {
    /// Create new security manager
    pub fn new(config: &SecurityConfig) -> NetworkResult<Self> {
        let private_key = if config.enable_auth {
            Some(generate_keypair()?)
        } else {
            None
        };

        let certificate = if config.enable_tls {
            config
                .cert_path
                .as_ref()
                .map(|path| load_certificate(path))
                .transpose()?
        } else {
            None
        };

        let encryption_key = if config.enable_encryption {
            Some(generate_encryption_key()?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            private_key,
            certificate,
            encryption_key,
        })
    }

    /// Sign data
    pub fn sign(&self, data: &[u8]) -> NetworkResult<Vec<u8>> {
        if let Some(key) = &self.private_key {
            Ok(key.sign(data).as_ref().to_vec())
        } else {
            Err(anyhow::anyhow!("Signing not enabled"))
        }
    }

    /// Verify signature
    pub fn verify(&self, data: &[u8], signature: &[u8], public_key: &[u8]) -> NetworkResult<bool> {
        let peer_public_key = signature::UnparsedPublicKey::new(&signature::ED25519, public_key);

        match peer_public_key.verify(data, signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Encrypt data
    pub fn encrypt(&self, data: &[u8]) -> NetworkResult<Vec<u8>> {
        if let Some(key) = &self.encryption_key {
            let mut in_out = data.to_vec();
            let nonce_bytes = generate_nonce()?;

            let nonce = aead::Nonce::try_assume_unique_for_key(&nonce_bytes)
                .map_err(|e| anyhow::anyhow!("Invalid nonce: {:?}", e))?;

            key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut in_out)
                .map_err(|e| anyhow::anyhow!("Encryption failed: {:?}", e))?;

            // Prepend nonce
            let mut result = nonce_bytes.to_vec();
            result.extend_from_slice(&in_out);

            Ok(result)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Decrypt data
    pub fn decrypt(&self, data: &[u8]) -> NetworkResult<Vec<u8>> {
        if let Some(key) = &self.encryption_key {
            if data.len() < 12 {
                return Err(anyhow::anyhow!("Invalid encrypted data"));
            }

            let nonce = &data[..12];
            let mut in_out = data[12..].to_vec();

            let nonce_obj = aead::Nonce::try_assume_unique_for_key(nonce)
                .map_err(|e| anyhow::anyhow!("Invalid nonce: {:?}", e))?;

            let plain_data = key
                .open_in_place(nonce_obj, aead::Aad::empty(), &mut in_out)
                .map_err(|e| anyhow::anyhow!("Decryption failed: {:?}", e))?;

            let plain_len = plain_data.len();

            in_out.truncate(plain_len);
            Ok(in_out)
        } else {
            Ok(data.to_vec())
        }
    }

    /// Get public key
    pub fn public_key(&self) -> Option<Vec<u8>> {
        self.private_key
            .as_ref()
            .map(|k| k.public_key().as_ref().to_vec())
    }
}

fn generate_keypair() -> NetworkResult<signature::Ed25519KeyPair> {
    let rng = rand::SystemRandom::new();
    let pkcs8_bytes = signature::Ed25519KeyPair::generate_pkcs8(&rng)?;
    Ok(signature::Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())?)
}

fn generate_encryption_key() -> NetworkResult<aead::LessSafeKey> {
    let rng = rand::SystemRandom::new();
    let mut key_bytes = [0u8; 32];
    rand::SecureRandom::fill(&rng, &mut key_bytes)?;

    let unbound_key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, &key_bytes)?;
    Ok(aead::LessSafeKey::new(unbound_key))
}

fn generate_nonce() -> NetworkResult<[u8; 12]> {
    let mut nonce = [0u8; 12];
    let rng = rand::SystemRandom::new();
    rand::SecureRandom::fill(&rng, &mut nonce)
        .map_err(|e| anyhow::anyhow!("Failed to generate secure nonce: {:?}", e))?;
    Ok(nonce)
}

fn load_certificate(path: &str) -> NetworkResult<Certificate> {
    let cert_bytes = std::fs::read(path)?;
    Ok(Certificate(cert_bytes))
}
