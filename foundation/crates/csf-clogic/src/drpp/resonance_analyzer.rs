//! Resonance analysis for oscillator networks

use super::NeuralOscillator;
use ndarray::Array2;
use num_traits::Float;

/// Resonance analyzer for detecting coupling patterns
pub struct ResonanceAnalyzer {
    /// Frequency resolution for analysis
    frequency_resolution: f64,

    /// Frequency range
    frequency_range: (f64, f64),
}

impl ResonanceAnalyzer {
    /// Create a new resonance analyzer
    pub fn new(config: &super::DrppConfig) -> Self {
        Self {
            frequency_resolution: 0.1, // 0.1 Hz resolution
            frequency_range: config.frequency_range,
        }
    }

    /// Analyze resonance patterns in oscillator network
    pub fn analyze(&self, oscillators: &[NeuralOscillator]) -> Array2<f64> {
        let n = oscillators.len();
        let mut resonance_map = Array2::zeros((n, n));

        // Calculate pairwise resonance strength
        for i in 0..n {
            for j in i + 1..n {
                let resonance = self.calculate_resonance(&oscillators[i], &oscillators[j]);
                resonance_map[[i, j]] = resonance;
                resonance_map[[j, i]] = resonance; // Symmetric
            }
        }

        // Normalize by row to get relative resonance strengths
        for i in 0..n {
            let row_sum = resonance_map.row(i).sum();
            if row_sum > 0.0 {
                resonance_map.row_mut(i).mapv_inplace(|x| x / row_sum);
            }
        }

        resonance_map
    }

    /// Calculate resonance between two oscillators
    fn calculate_resonance(&self, osc1: &NeuralOscillator, osc2: &NeuralOscillator) -> f64 {
        let freq1 = osc1.frequency();
        let freq2 = osc2.frequency();
        let phase1 = osc1.phase();
        let phase2 = osc2.phase();

        // Frequency resonance (Arnold tongue)
        let freq_ratio = freq1 / freq2;
        let mut freq_resonance = 0.0;

        // Check for harmonic resonance (1:1, 1:2, 2:3, etc.)
        for p in 1..=5 {
            for q in 1..=5 {
                let ratio = p as f64 / q as f64;
                let diff = (freq_ratio - ratio).abs();
                if diff < self.frequency_resolution {
                    freq_resonance = freq_resonance.max(1.0 - diff / self.frequency_resolution);
                }
            }
        }

        // Phase coherence
        let phase_diff = (phase1 - phase2).abs();
        let phase_coherence = (phase_diff.cos() + 1.0) / 2.0;

        // Combined resonance measure
        freq_resonance * phase_coherence
    }

    /// Detect resonance clusters
    pub fn find_clusters(&self, resonance_map: &Array2<f64>, threshold: f64) -> Vec<Vec<usize>> {
        let n = resonance_map.nrows();
        let mut visited = vec![false; n];
        let mut clusters = Vec::new();

        for i in 0..n {
            if !visited[i] {
                let mut cluster = Vec::new();
                self.dfs_cluster(i, &mut visited, &mut cluster, resonance_map, threshold);

                if cluster.len() > 1 {
                    clusters.push(cluster);
                }
            }
        }

        clusters
    }

    /// Depth-first search for cluster detection
    fn dfs_cluster(
        &self,
        node: usize,
        visited: &mut [bool],
        cluster: &mut Vec<usize>,
        resonance_map: &Array2<f64>,
        threshold: f64,
    ) {
        visited[node] = true;
        cluster.push(node);

        for j in 0..resonance_map.ncols() {
            if !visited[j] && resonance_map[[node, j]] > threshold {
                self.dfs_cluster(j, visited, cluster, resonance_map, threshold);
            }
        }
    }

    /// Calculate global resonance metrics
    pub fn global_metrics(&self, resonance_map: &Array2<f64>) -> ResonanceMetrics {
        let total_resonance = resonance_map.sum();
        let n = resonance_map.nrows() as f64;
        let mean_resonance = total_resonance / (n * n);

        // Calculate variance
        let variance = resonance_map
            .iter()
            .map(|&x| (x - mean_resonance).powi(2))
            .sum::<f64>()
            / (n * n);

        // Find strongest resonance pairs
        let mut max_resonance = 0.0;
        let mut strongest_pair = (0, 0);

        for i in 0..resonance_map.nrows() {
            for j in i + 1..resonance_map.ncols() {
                if resonance_map[[i, j]] > max_resonance {
                    max_resonance = resonance_map[[i, j]];
                    strongest_pair = (i, j);
                }
            }
        }

        ResonanceMetrics {
            mean_resonance,
            variance,
            max_resonance,
            strongest_pair,
        }
    }
}

/// Global resonance metrics
#[derive(Debug, Clone)]
pub struct ResonanceMetrics {
    pub mean_resonance: f64,
    pub variance: f64,
    pub max_resonance: f64,
    pub strongest_pair: (usize, usize),
}
