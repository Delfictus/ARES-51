use ed25519_dalek::{Signature, Signer, SigningKey};
use rand::rngs::OsRng;
use rand::RngCore;

/// Generate a cryptographically secure Ed25519 keypair
///
/// # Security
/// Uses the OS random number generator for secure key generation
///
/// # Security
/// Uses the OS random number generator for secure key generation
///
/// # Returns
/// * `Ok(SigningKey)` on success
/// * `Err(ed25519_dalek::SignatureError)` if the key bytes are invalid (should not occur)
pub fn generate_keypair() -> Result<SigningKey, ed25519_dalek::SignatureError> {
    let mut secret_key_bytes = [0u8; 32];
    OsRng.fill_bytes(&mut secret_key_bytes);
    Ok(SigningKey::from_bytes(&secret_key_bytes))
}

/// Sign data with Ed25519 digital signature
///
/// # Arguments
/// * `key` - The signing key to use
/// * `data` - The data to sign
///
/// # Returns
/// A cryptographically valid Ed25519 signature
pub fn sign(key: &SigningKey, data: &[u8]) -> Signature {
    key.sign(data)
}
