//! Compression utilities

use super::*;

/// Compress data using specified algorithm
pub fn compress(
    data: &[u8],
    algorithm: CompressionAlgorithm,
    level: u32,
) -> NetworkResult<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::Lz4 => compress_lz4(data, level),
        CompressionAlgorithm::Zstd => compress_zstd(data, level),
        CompressionAlgorithm::None => Ok(data.to_vec()),
    }
}

/// Decompress data using specified algorithm
pub fn decompress(data: &[u8], algorithm: CompressionAlgorithm) -> NetworkResult<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::Lz4 => decompress_lz4(data),
        CompressionAlgorithm::Zstd => decompress_zstd(data),
        CompressionAlgorithm::None => Ok(data.to_vec()),
    }
}

fn compress_lz4(data: &[u8], _level: u32) -> NetworkResult<Vec<u8>> {
    Ok(lz4::block::compress(
        data,
        Some(lz4::block::CompressionMode::DEFAULT),
        true,
    )?)
}

fn decompress_lz4(data: &[u8]) -> NetworkResult<Vec<u8>> {
    Ok(lz4::block::decompress(data, None)?)
}

fn compress_zstd(data: &[u8], level: u32) -> NetworkResult<Vec<u8>> {
    Ok(zstd::encode_all(data, level as i32)?)
}

fn decompress_zstd(data: &[u8]) -> NetworkResult<Vec<u8>> {
    Ok(zstd::decode_all(data)?)
}
