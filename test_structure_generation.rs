#!/usr/bin/env cargo
//! Test structure generation with validation results

use std::fs;
use prct_engine::structure::{PRCTFolder, write_pdb_structure};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Testing PRCT Structure Generation");
    println!("====================================");

    // Load validation results
    let results_file = "validation/results/prct_blind_test_results_20250915_004326.json";

    match fs::read_to_string(results_file) {
        Ok(content) => {
            let results: Value = serde_json::from_str(&content)?;

            // Initialize PRCT folder
            let folder = PRCTFolder::with_debug();

            // Extract target predictions
            if let Some(predictions) = results.get("target_predictions").and_then(|v| v.as_array()) {
                println!("📊 Found {} predictions to process", predictions.len());

                for (i, prediction) in predictions.iter().enumerate() {
                    let target_id = prediction.get("target_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("UNKNOWN");

                    let phase_coherence = prediction.get("phase_coherence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.75);

                    let chromatic_score = prediction.get("chromatic_score")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.65);

                    let hamiltonian_energy = prediction.get("hamiltonian_energy")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(-25.0);

                    let tsp_energy = prediction.get("tsp_energy")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(-15.0);

                    println!("\n🎯 Target {}: {}", i + 1, target_id);
                    println!("   📈 Phase coherence: {:.3}", phase_coherence);
                    println!("   🎨 Chromatic score: {:.3}", chromatic_score);
                    println!("   ⚡ Hamiltonian energy: {:.1} kcal/mol", hamiltonian_energy);
                    println!("   🔗 TSP energy: {:.1} kcal/mol", tsp_energy);

                    // Generate 3D structure
                    match folder.fold_to_coordinates(
                        target_id,
                        phase_coherence,
                        chromatic_score,
                        hamiltonian_energy,
                        tsp_energy
                    ) {
                        Ok(structure) => {
                            println!("   ✅ Generated structure with {} atoms", structure.atom_count());
                            println!("   🎯 Confidence: {:.3}", structure.confidence);
                            println!("   ⚡ Total energy: {:.1} kcal/mol", structure.energy);

                            // Generate PDB content
                            match write_pdb_structure(&structure) {
                                Ok(pdb_content) => {
                                    // Create output directory
                                    fs::create_dir_all("test_structures")?;

                                    // Write PDB file
                                    let pdb_file = format!("test_structures/{}.pdb", target_id);
                                    fs::write(&pdb_file, pdb_content)?;
                                    println!("   💾 PDB saved: {}", pdb_file);

                                    // Show first few lines of PDB
                                    let lines: Vec<&str> = pdb_content.lines().take(5).collect();
                                    println!("   📋 PDB preview:");
                                    for line in lines {
                                        println!("      {}", line);
                                    }
                                }
                                Err(e) => {
                                    println!("   ❌ PDB generation failed: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("   ❌ Structure generation failed: {}", e);
                        }
                    }
                }

                println!("\n🎉 Structure generation test completed!");

            } else {
                println!("❌ No target predictions found in results file");
            }
        }
        Err(e) => {
            println!("❌ Failed to load validation results: {}", e);
            println!("💡 Expected file: {}", results_file);
            println!("💡 Make sure validation results are available");
        }
    }

    Ok(())
}