//! Preset configurations for common use cases

use crate::encoding::{
    EncodingConfig, RateParams, TemporalParams, PopulationParams, 
    PhaseParams, NoiseParams, QualityControl, RateScaling, NoiseType
};
use crate::types::EncodingMethod;

/// Collection of preset configurations
pub struct EncodingPresets;

impl EncodingPresets {
    /// High-frequency trading optimized configuration
    pub fn high_frequency() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 5000,
            window_ms: 100.0, // 100ms window for HFT
            method: EncodingMethod::Rate,
            rate_params: RateParams {
                max_rate: 200.0, // High spike rates
                min_rate: 5.0,
                scaling: RateScaling::Linear,
            },
            temporal_params: TemporalParams {
                max_delay_ms: 10.0, // Short delays for speed
                min_delay_ms: 0.0,
                resolution_ms: 0.01, // 10μs resolution
            },
            noise_params: NoiseParams {
                level: 0.005, // Low noise for precision
                noise_type: NoiseType::Gaussian,
                seed: None,
            },
            quality_control: QualityControl {
                validate_spikes: true,
                max_spikes: Some(50000),
                min_information: Some(0.5),
            },
            ..Default::default()
        }
    }

    /// Low-latency configuration for real-time processing
    pub fn low_latency() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 1000, // Fewer neurons for speed
            window_ms: 50.0,    // Very short window
            method: EncodingMethod::Latency, // Latency encoding is fastest
            temporal_params: TemporalParams {
                max_delay_ms: 5.0,
                min_delay_ms: 0.0,
                resolution_ms: 0.001, // 1μs resolution
            },
            noise_params: NoiseParams {
                level: 0.001, // Minimal noise
                noise_type: NoiseType::Uniform,
                seed: Some(42), // Fixed seed for reproducibility
            },
            quality_control: QualityControl {
                validate_spikes: false, // Skip validation for speed
                max_spikes: Some(5000),
                min_information: None,
            },
            ..Default::default()
        }
    }

    /// High-precision configuration for detailed analysis
    pub fn high_precision() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 20000, // Many neurons for precision
            window_ms: 5000.0,   // Long window for detailed patterns
            method: EncodingMethod::Population,
            population_params: PopulationParams {
                neurons_per_feature: 50, // Many neurons per feature
                tuning_width: 0.1,       // Narrow tuning curves
                overlap: 0.3,            // Reduced overlap
            },
            rate_params: RateParams {
                max_rate: 500.0, // Very high rates
                min_rate: 0.1,
                scaling: RateScaling::Logarithmic,
            },
            noise_params: NoiseParams {
                level: 0.0001, // Very low noise
                noise_type: NoiseType::Gaussian,
                seed: None,
            },
            quality_control: QualityControl {
                validate_spikes: true,
                max_spikes: Some(100000),
                min_information: Some(0.8),
            },
            ..Default::default()
        }
    }

    /// Memory-efficient configuration for resource-constrained environments
    pub fn memory_efficient() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 500,  // Minimal neurons
            window_ms: 500.0,   // Moderate window
            method: EncodingMethod::Rate,
            rate_params: RateParams {
                max_rate: 50.0,  // Lower rates
                min_rate: 1.0,
                scaling: RateScaling::Linear,
            },
            quality_control: QualityControl {
                validate_spikes: false, // Skip validation
                max_spikes: Some(2000), // Limit spikes
                min_information: None,
            },
            ..Default::default()
        }
    }

    /// Research configuration with extensive parameters
    pub fn research() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 10000,
            window_ms: 2000.0,
            method: EncodingMethod::Phase,
            phase_params: PhaseParams {
                base_frequency: 40.0,   // Gamma frequency
                phase_range: 4.0 * std::f64::consts::PI, // Extended range
                cycles_per_window: 2.0, // Multiple cycles
            },
            noise_params: NoiseParams {
                level: 0.02,  // Moderate noise for realism
                noise_type: NoiseType::Poisson,
                seed: None,
            },
            quality_control: QualityControl {
                validate_spikes: true,
                max_spikes: Some(80000),
                min_information: Some(0.3),
            },
            ..Default::default()
        }
    }

    /// Robust configuration that works well in most scenarios
    pub fn balanced() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 2000,
            window_ms: 1000.0,
            method: EncodingMethod::Population,
            population_params: PopulationParams {
                neurons_per_feature: 20,
                tuning_width: 0.2,
                overlap: 0.5,
            },
            rate_params: RateParams {
                max_rate: 100.0,
                min_rate: 2.0,
                scaling: RateScaling::Sigmoid,
            },
            noise_params: NoiseParams {
                level: 0.01,
                noise_type: NoiseType::Gaussian,
                seed: None,
            },
            quality_control: QualityControl {
                validate_spikes: true,
                max_spikes: Some(20000),
                min_information: Some(0.4),
            },
            ..Default::default()
        }
    }

    /// Configuration optimized for burst detection
    pub fn burst_detection() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 1500,
            window_ms: 1500.0,
            method: EncodingMethod::Burst,
            temporal_params: TemporalParams {
                max_delay_ms: 100.0,
                min_delay_ms: 5.0,
                resolution_ms: 0.1,
            },
            noise_params: NoiseParams {
                level: 0.015,
                noise_type: NoiseType::Poisson,
                seed: None,
            },
            ..Default::default()
        }
    }

    /// Get all available presets with names
    pub fn all_presets() -> Vec<(&'static str, EncodingConfig)> {
        vec![
            ("high_frequency", Self::high_frequency()),
            ("low_latency", Self::low_latency()),
            ("high_precision", Self::high_precision()),
            ("memory_efficient", Self::memory_efficient()),
            ("research", Self::research()),
            ("balanced", Self::balanced()),
            ("burst_detection", Self::burst_detection()),
        ]
    }

    /// Get preset by name
    pub fn get_preset(name: &str) -> Option<EncodingConfig> {
        match name {
            "high_frequency" => Some(Self::high_frequency()),
            "low_latency" => Some(Self::low_latency()),
            "high_precision" => Some(Self::high_precision()),
            "memory_efficient" => Some(Self::memory_efficient()),
            "research" => Some(Self::research()),
            "balanced" => Some(Self::balanced()),
            "burst_detection" => Some(Self::burst_detection()),
            _ => None,
        }
    }
}

/// Financial market specific presets
pub struct FinancialPresets;

impl FinancialPresets {
    /// Default configuration for financial markets
    pub fn default_config() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 3000,
            window_ms: 1000.0,
            method: EncodingMethod::Population,
            population_params: PopulationParams {
                neurons_per_feature: 15,
                tuning_width: 0.15,
                overlap: 0.4,
            },
            rate_params: RateParams {
                max_rate: 150.0,
                min_rate: 1.0,
                scaling: RateScaling::Logarithmic, // Good for price data
            },
            temporal_params: TemporalParams {
                max_delay_ms: 50.0,
                min_delay_ms: 0.5,
                resolution_ms: 0.1,
            },
            noise_params: NoiseParams {
                level: 0.008, // Realistic market noise
                noise_type: NoiseType::Gaussian,
                seed: None,
            },
            quality_control: QualityControl {
                validate_spikes: true,
                max_spikes: Some(30000),
                min_information: Some(0.3),
            },
            ..Default::default()
        }
    }

    /// Configuration optimized for cryptocurrency markets
    pub fn crypto_config() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 4000,
            window_ms: 500.0, // Crypto moves fast
            method: EncodingMethod::Rate,
            rate_params: RateParams {
                max_rate: 300.0, // High volatility needs high rates
                min_rate: 2.0,
                scaling: RateScaling::Exponential,
            },
            noise_params: NoiseParams {
                level: 0.02, // Higher noise for crypto volatility
                noise_type: NoiseType::Poisson,
                seed: None,
            },
            quality_control: QualityControl {
                validate_spikes: true,
                max_spikes: Some(40000),
                min_information: Some(0.25),
            },
            ..Default::default()
        }
    }

    /// Configuration for forex markets
    pub fn forex_config() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 2500,
            window_ms: 2000.0, // Forex trends develop slowly
            method: EncodingMethod::Phase,
            phase_params: PhaseParams {
                base_frequency: 25.0, // Lower frequency for forex
                phase_range: 2.0 * std::f64::consts::PI,
                cycles_per_window: 1.5,
            },
            rate_params: RateParams {
                max_rate: 80.0, // Lower volatility
                min_rate: 0.5,
                scaling: RateScaling::Sigmoid,
            },
            noise_params: NoiseParams {
                level: 0.005, // Lower noise for major pairs
                noise_type: NoiseType::Gaussian,
                seed: None,
            },
            ..Default::default()
        }
    }

    /// Configuration for stock markets
    pub fn stocks_config() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 3500,
            window_ms: 1500.0,
            method: EncodingMethod::Population,
            population_params: PopulationParams {
                neurons_per_feature: 25,
                tuning_width: 0.18,
                overlap: 0.45,
            },
            rate_params: RateParams {
                max_rate: 120.0,
                min_rate: 1.0,
                scaling: RateScaling::Linear,
            },
            noise_params: NoiseParams {
                level: 0.01,
                noise_type: NoiseType::Gaussian,
                seed: None,
            },
            ..Default::default()
        }
    }

    /// Configuration for commodity markets
    pub fn commodities_config() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 2000,
            window_ms: 3000.0, // Commodities move slowly
            method: EncodingMethod::Temporal,
            temporal_params: TemporalParams {
                max_delay_ms: 200.0, // Longer delays for slow markets
                min_delay_ms: 5.0,
                resolution_ms: 1.0,
            },
            rate_params: RateParams {
                max_rate: 60.0, // Lower rates for slower markets
                min_rate: 0.2,
                scaling: RateScaling::Logarithmic,
            },
            noise_params: NoiseParams {
                level: 0.012,
                noise_type: NoiseType::Uniform,
                seed: None,
            },
            ..Default::default()
        }
    }

    /// Configuration for options/derivatives markets
    pub fn derivatives_config() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 5000, // Complex derivatives need more neurons
            window_ms: 800.0,
            method: EncodingMethod::Population,
            population_params: PopulationParams {
                neurons_per_feature: 30, // High resolution for Greeks
                tuning_width: 0.12,
                overlap: 0.35,
            },
            rate_params: RateParams {
                max_rate: 250.0,
                min_rate: 1.5,
                scaling: RateScaling::Exponential,
            },
            noise_params: NoiseParams {
                level: 0.015,
                noise_type: NoiseType::Gaussian,
                seed: None,
            },
            quality_control: QualityControl {
                validate_spikes: true,
                max_spikes: Some(50000),
                min_information: Some(0.4),
            },
            ..Default::default()
        }
    }

    /// Configuration for high-frequency trading
    pub fn hft_config() -> EncodingConfig {
        EncodingConfig {
            neuron_count: 8000,
            window_ms: 50.0, // Ultra-short window
            method: EncodingMethod::Latency, // Fastest encoding
            temporal_params: TemporalParams {
                max_delay_ms: 2.0, // Microsecond precision
                min_delay_ms: 0.0,
                resolution_ms: 0.001,
            },
            rate_params: RateParams {
                max_rate: 1000.0, // Very high rates
                min_rate: 10.0,
                scaling: RateScaling::Linear,
            },
            noise_params: NoiseParams {
                level: 0.002, // Minimal noise for precision
                noise_type: NoiseType::Uniform,
                seed: Some(123), // Reproducible for testing
            },
            quality_control: QualityControl {
                validate_spikes: false, // Skip for speed
                max_spikes: Some(80000),
                min_information: None,
            },
            ..Default::default()
        }
    }

    /// Get all financial presets
    pub fn all_financial_presets() -> Vec<(&'static str, EncodingConfig)> {
        vec![
            ("default", Self::default_config()),
            ("crypto", Self::crypto_config()),
            ("forex", Self::forex_config()),
            ("stocks", Self::stocks_config()),
            ("commodities", Self::commodities_config()),
            ("derivatives", Self::derivatives_config()),
            ("hft", Self::hft_config()),
        ]
    }

    /// Get financial preset by name
    pub fn get_preset(name: &str) -> Option<EncodingConfig> {
        match name {
            "default" => Some(Self::default_config()),
            "crypto" => Some(Self::crypto_config()),
            "forex" => Some(Self::forex_config()),
            "stocks" => Some(Self::stocks_config()),
            "commodities" => Some(Self::commodities_config()),
            "derivatives" => Some(Self::derivatives_config()),
            "hft" => Some(Self::hft_config()),
            _ => None,
        }
    }
}

/// Preset builder for creating custom configurations
pub struct PresetBuilder {
    config: EncodingConfig,
}

impl PresetBuilder {
    /// Start with a base preset
    pub fn from_preset(name: &str) -> Option<Self> {
        EncodingPresets::get_preset(name)
            .or_else(|| FinancialPresets::get_preset(name))
            .map(|config| Self { config })
    }

    /// Start with default configuration
    pub fn new() -> Self {
        Self {
            config: EncodingConfig::default(),
        }
    }

    /// Set neuron count
    pub fn neuron_count(mut self, count: usize) -> Self {
        self.config.neuron_count = count;
        self
    }

    /// Set time window
    pub fn window_ms(mut self, window: f64) -> Self {
        self.config.window_ms = window;
        self
    }

    /// Set encoding method
    pub fn method(mut self, method: EncodingMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set maximum spike rate
    pub fn max_rate(mut self, rate: f64) -> Self {
        self.config.rate_params.max_rate = rate;
        self
    }

    /// Set noise level
    pub fn noise_level(mut self, level: f64) -> Self {
        self.config.noise_params.level = level;
        self
    }

    /// Enable/disable spike validation
    pub fn validate_spikes(mut self, validate: bool) -> Self {
        self.config.quality_control.validate_spikes = validate;
        self
    }

    /// Build the final configuration
    pub fn build(self) -> EncodingConfig {
        self.config
    }
}

impl Default for PresetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets_work() {
        for (name, config) in EncodingPresets::all_presets() {
            // Test that each preset can be created
            assert!(config.neuron_count > 0, "Preset {} has invalid neuron count", name);
            assert!(config.window_ms > 0.0, "Preset {} has invalid window", name);
        }
    }

    #[test]
    fn test_financial_presets() {
        for (name, config) in FinancialPresets::all_financial_presets() {
            assert!(config.neuron_count > 0, "Financial preset {} has invalid neuron count", name);
            assert!(config.window_ms > 0.0, "Financial preset {} has invalid window", name);
        }
    }

    #[test]
    fn test_preset_retrieval() {
        assert!(EncodingPresets::get_preset("balanced").is_some());
        assert!(EncodingPresets::get_preset("nonexistent").is_none());
        
        assert!(FinancialPresets::get_preset("crypto").is_some());
        assert!(FinancialPresets::get_preset("nonexistent").is_none());
    }

    #[test]
    fn test_preset_builder() {
        let config = PresetBuilder::new()
            .neuron_count(5000)
            .window_ms(500.0)
            .method(EncodingMethod::Rate)
            .max_rate(200.0)
            .noise_level(0.01)
            .validate_spikes(false)
            .build();

        assert_eq!(config.neuron_count, 5000);
        assert_eq!(config.window_ms, 500.0);
        assert_eq!(config.method, EncodingMethod::Rate);
        assert_eq!(config.rate_params.max_rate, 200.0);
        assert_eq!(config.noise_params.level, 0.01);
        assert_eq!(config.quality_control.validate_spikes, false);
    }

    #[test]
    fn test_builder_from_preset() {
        let config = PresetBuilder::from_preset("balanced")
            .unwrap()
            .neuron_count(10000)
            .max_rate(500.0)
            .build();

        assert_eq!(config.neuron_count, 10000);
        assert_eq!(config.rate_params.max_rate, 500.0);
        // Other parameters should be from the balanced preset
        assert_eq!(config.method, EncodingMethod::Population);
    }

    #[test]
    fn test_hft_config_characteristics() {
        let config = FinancialPresets::hft_config();
        
        // HFT should have short window and high rates
        assert!(config.window_ms <= 100.0);
        assert!(config.rate_params.max_rate >= 500.0);
        assert_eq!(config.method, EncodingMethod::Latency);
        assert_eq!(config.quality_control.validate_spikes, false);
    }

    #[test]
    fn test_commodities_config_characteristics() {
        let config = FinancialPresets::commodities_config();
        
        // Commodities should have longer windows and lower rates
        assert!(config.window_ms >= 2000.0);
        assert!(config.rate_params.max_rate <= 100.0);
        assert_eq!(config.method, EncodingMethod::Temporal);
    }
}