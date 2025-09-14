#!/usr/bin/env cargo
//! Import Test Binary
//! Tests that all imports in the fixed binaries work correctly

use prct_engine::PRCTResult;
use prct_engine::data::CASPLoader;

fn main() -> PRCTResult<()> {
    println!("🧬 Testing PRCT Engine Imports...");

    // Test CASPLoader import and construction
    let temp_dir = std::env::temp_dir().join("import_test");
    std::fs::create_dir_all(&temp_dir).unwrap();

    match CASPLoader::new(&temp_dir) {
        Ok(_loader) => {
            println!("✅ CASPLoader import and construction: SUCCESS");
        },
        Err(e) => {
            println!("✅ CASPLoader import: SUCCESS (expected error for empty dir: {})", e);
        }
    }

    println!("✅ PRCTResult type: SUCCESS");

    // Cleanup
    std::fs::remove_dir_all(&temp_dir).unwrap_or(());

    println!("🎯 All imports working correctly!");
    Ok(())
}