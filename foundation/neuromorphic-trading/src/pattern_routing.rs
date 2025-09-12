//! PHASE 2C.2: Advanced Pattern Data Routing System
//! Revolutionary intelligent routing for neuromorphic pattern distribution
//! Implements priority-based, content-aware routing with adaptive load balancing

use crate::drpp::{DynamicResonancePatternProcessor, DrppState, Pattern, PatternType};
use crate::neuromorphic::{ResonancePattern, ResonancePatternType};
use crate::adp::{AdaptiveDecisionProcessor, Decision, Action};
use crate::multi_timeframe::{MultiTimeframeResult, TimeHorizon};
use crate::phase_coherence::{CoherencePattern, MarketRegime};
use crate::spike_encoding::Spike;
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque, BinaryHeap};
use anyhow::{Result, anyhow};
use tokio::time::{Duration, Instant};
use tokio::sync::mpsc;
use rand::prelude::*;

/// Advanced pattern routing engine with intelligent load balancing
pub struct PatternRoutingEngine {
    /// Routing table for pattern destinations
    routing_table: Arc<RwLock<RoutingTable>>,
    /// Pattern priority queue for high-priority patterns
    priority_queue: Arc<RwLock<BinaryHeap<PriorityPattern>>>,
    /// Load balancing manager
    load_balancer: Arc<RwLock<LoadBalancer>>,
    /// Pattern delivery statistics
    delivery_stats: Arc<RwLock<DeliveryStatistics>>,
    /// Routing strategy configuration
    strategy: RoutingStrategy,
    /// Content-based routing filters
    content_filters: HashMap<PatternType, ContentFilter>,
    /// Geographic routing for multi-node systems
    geographic_routing: Arc<RwLock<GeographicRouter>>,
    /// Adaptive routing learning system
    adaptive_learner: Arc<RwLock<AdaptiveRouter>>,
}

/// Routing table mapping pattern types to destinations
#[derive(Debug, Clone)]
pub struct RoutingTable {
    /// Pattern type to destination mapping
    routes: HashMap<PatternType, Vec<RouteDestination>>,
    /// Default route for unmatched patterns
    default_route: Option<RouteDestination>,
    /// Route health status
    route_health: HashMap<RouteDestination, RouteHealth>,
    /// Last update timestamp
    last_updated: Instant,
}

/// Route destination with load balancing information
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RouteDestination {
    /// DRPP processor node
    DrppProcessor(u32),
    /// ADP decision processor
    AdpProcessor(u32), 
    /// Multi-timeframe analyzer
    MultiTimeframe(TimeHorizon),
    /// External system integration
    External(String),
    /// Local processing queue
    Local(String),
    /// Broadcast to multiple destinations
    Broadcast(Vec<Box<RouteDestination>>),
}

/// Route health status for intelligent failover
#[derive(Debug, Clone)]
pub struct RouteHealth {
    /// Current load (0.0 to 1.0)
    pub load: f64,
    /// Response latency (microseconds)
    pub latency_us: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Queue depth
    pub queue_depth: usize,
    /// Last health check timestamp
    pub last_check: Instant,
    /// Is route available
    pub available: bool,
}

/// Priority pattern wrapper for queue ordering
#[derive(Debug, Clone)]
pub struct PriorityPattern {
    /// Pattern data
    pub pattern: PatternData,
    /// Priority score (higher = more urgent)
    pub priority: u32,
    /// Creation timestamp
    pub timestamp: Instant,
    /// Expiration time
    pub expires_at: Option<Instant>,
    /// Retry count
    pub retry_count: u32,
    /// Source identifier
    pub source_id: String,
}

/// Unified pattern data container
#[derive(Debug, Clone)]
pub enum PatternData {
    /// DRPP pattern
    Drpp(Pattern),
    /// Resonance pattern
    Resonance(ResonancePattern),
    /// Coherence pattern
    Coherence(CoherencePattern),
    /// Multi-timeframe result
    MultiFrame(MultiTimeframeResult),
    /// Spike pattern
    Spikes(Vec<Spike>),
    /// Custom pattern
    Custom(CustomPattern),
}

/// Custom pattern for extensibility
#[derive(Debug, Clone)]
pub struct CustomPattern {
    pub pattern_type: String,
    pub data: Vec<f64>,
    pub metadata: HashMap<String, String>,
    pub confidence: f64,
}

/// Content-based routing filter
#[derive(Debug, Clone)]
pub struct ContentFilter {
    /// Minimum pattern strength threshold
    pub min_strength: f64,
    /// Maximum pattern age (milliseconds)
    pub max_age_ms: u64,
    /// Required metadata fields
    pub required_fields: Vec<String>,
    /// Pattern importance weighting
    pub importance_weight: f64,
    /// Market regime restrictions
    pub regime_filter: Option<Vec<MarketRegime>>,
}

/// Load balancing manager
#[derive(Debug)]
pub struct LoadBalancer {
    /// Current load per destination
    destination_loads: HashMap<RouteDestination, f64>,
    /// Load balancing algorithm
    algorithm: LoadBalancingAlgorithm,
    /// Target load threshold (0.0 to 1.0)
    target_load: f64,
    /// Circuit breaker states
    circuit_breakers: HashMap<RouteDestination, CircuitBreakerState>,
}

/// Load balancing algorithms
#[derive(Debug, Clone)]
pub enum LoadBalancingAlgorithm {
    /// Round-robin routing
    RoundRobin,
    /// Least loaded destination
    LeastLoaded,
    /// Weighted round-robin
    WeightedRoundRobin(HashMap<RouteDestination, f64>),
    /// Consistent hashing
    ConsistentHash,
    /// AI-based predictive routing
    AiPredictive,
}

/// Circuit breaker state for fault tolerance
#[derive(Debug, Clone)]
pub enum CircuitBreakerState {
    Closed,  // Normal operation
    Open,    // Circuit breaker triggered
    HalfOpen, // Testing if service recovered
}

/// Geographic routing for distributed systems
#[derive(Debug)]
pub struct GeographicRouter {
    /// Node locations and capabilities
    node_topology: HashMap<String, NodeInfo>,
    /// Distance matrix between nodes
    distance_matrix: HashMap<(String, String), f64>,
    /// Network partition detection
    partitions: Vec<Vec<String>>,
}

/// Node information for geographic routing
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: String,
    pub location: (f64, f64), // lat, lon
    pub capabilities: Vec<String>,
    pub processing_power: f64,
    pub network_latency_ms: f64,
}

/// Adaptive routing learning system
#[derive(Debug)]
pub struct AdaptiveRouter {
    /// Routing success history
    success_history: VecDeque<RoutingDecision>,
    /// Pattern-destination affinity matrix
    affinity_matrix: HashMap<(PatternType, RouteDestination), f64>,
    /// Learning rate for adaptation
    learning_rate: f64,
    /// Exploration vs exploitation balance
    epsilon: f64,
}

/// Routing decision record for learning
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub pattern_type: PatternType,
    pub destination: RouteDestination,
    pub latency_us: f64,
    pub success: bool,
    pub timestamp: Instant,
    pub load_at_decision: f64,
}

/// Routing strategy configuration
#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Priority-based routing (high priority first)
    Priority,
    /// Content-aware routing (based on pattern characteristics)
    ContentAware,
    /// Load-aware routing (consider destination load)
    LoadAware,
    /// Latency-optimized routing (minimize latency)
    LatencyOptimized,
    /// Throughput-optimized routing (maximize throughput)
    ThroughputOptimized,
    /// Hybrid strategy combining multiple approaches
    Hybrid(Vec<RoutingStrategy>),
}

/// Delivery statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct DeliveryStatistics {
    pub total_patterns_routed: u64,
    pub successful_deliveries: u64,
    pub failed_deliveries: u64,
    pub average_latency_us: f64,
    pub patterns_per_second: f64,
    pub queue_overflow_count: u64,
    pub circuit_breaker_triggers: u64,
    pub last_reset: Instant,
}

impl PatternRoutingEngine {
    /// Create new pattern routing engine
    pub fn new(strategy: RoutingStrategy) -> Self {
        let routing_table = Arc::new(RwLock::new(RoutingTable {
            routes: HashMap::new(),
            default_route: Some(RouteDestination::Local("default".to_string())),
            route_health: HashMap::new(),
            last_updated: Instant::now(),
        }));

        let priority_queue = Arc::new(RwLock::new(BinaryHeap::new()));
        
        let load_balancer = Arc::new(RwLock::new(LoadBalancer {
            destination_loads: HashMap::new(),
            algorithm: LoadBalancingAlgorithm::LeastLoaded,
            target_load: 0.8,
            circuit_breakers: HashMap::new(),
        }));

        let geographic_routing = Arc::new(RwLock::new(GeographicRouter {
            node_topology: HashMap::new(),
            distance_matrix: HashMap::new(),
            partitions: Vec::new(),
        }));

        let adaptive_learner = Arc::new(RwLock::new(AdaptiveRouter {
            success_history: VecDeque::with_capacity(10000),
            affinity_matrix: HashMap::new(),
            learning_rate: 0.01,
            epsilon: 0.1,
        }));

        // Initialize content filters
        let mut content_filters = HashMap::new();
        
        content_filters.insert(PatternType::Emergent, ContentFilter {
            min_strength: 0.8,
            max_age_ms: 100,
            required_fields: vec!["timestamp".to_string(), "strength".to_string()],
            importance_weight: 2.0,
            regime_filter: None,
        });

        content_filters.insert(PatternType::Synchronous, ContentFilter {
            min_strength: 0.6,
            max_age_ms: 500,
            required_fields: vec!["coherence".to_string()],
            importance_weight: 1.5,
            regime_filter: Some(vec![MarketRegime::Trending, MarketRegime::RegimeShift]),
        });

        content_filters.insert(PatternType::Chaotic, ContentFilter {
            min_strength: 0.4,
            max_age_ms: 50,
            required_fields: vec!["entropy".to_string()],
            importance_weight: 1.8,
            regime_filter: Some(vec![MarketRegime::Chaotic, MarketRegime::Transitional]),
        });

        Self {
            routing_table,
            priority_queue,
            load_balancer,
            delivery_stats: Arc::new(RwLock::new(DeliveryStatistics::default())),
            strategy,
            content_filters,
            geographic_routing,
            adaptive_learner,
        }
    }

    /// Route pattern to optimal destination
    pub async fn route_pattern(&self, pattern_data: PatternData, priority: u32) -> Result<RouteDestination> {
        let start_time = Instant::now();
        
        // Create priority pattern
        let priority_pattern = PriorityPattern {
            pattern: pattern_data.clone(),
            priority,
            timestamp: start_time,
            expires_at: Some(start_time + Duration::from_millis(1000)), // 1 second TTL
            retry_count: 0,
            source_id: "routing_engine".to_string(),
        };

        // Apply content filtering
        if !self.passes_content_filter(&pattern_data).await? {
            return Err(anyhow!("Pattern rejected by content filter"));
        }

        // Determine optimal destination based on strategy
        let destination = match &self.strategy {
            RoutingStrategy::Priority => {
                self.route_by_priority(&priority_pattern).await?
            },
            RoutingStrategy::ContentAware => {
                self.route_by_content(&pattern_data).await?
            },
            RoutingStrategy::LoadAware => {
                self.route_by_load(&pattern_data).await?
            },
            RoutingStrategy::LatencyOptimized => {
                self.route_by_latency(&pattern_data).await?
            },
            RoutingStrategy::ThroughputOptimized => {
                self.route_by_throughput(&pattern_data).await?
            },
            RoutingStrategy::Hybrid(strategies) => {
                self.route_by_hybrid(strategies, &pattern_data, priority).await?
            },
        };

        // Update routing statistics
        self.update_routing_stats(&destination, start_time.elapsed()).await?;

        // Learn from routing decision
        self.learn_from_routing(&pattern_data, &destination, start_time).await?;

        Ok(destination)
    }

    /// Apply content filtering to pattern
    async fn passes_content_filter(&self, pattern_data: &PatternData) -> Result<bool> {
        let pattern_type = match pattern_data {
            PatternData::Drpp(p) => p.pattern_type,
            PatternData::Resonance(p) => {
                // Map resonance types to DRPP pattern types
                match p.pattern_type {
                    ResonancePatternType::GlobalSync => PatternType::Synchronous,
                    ResonancePatternType::ChimeraState => PatternType::Chaotic,
                    ResonancePatternType::EmergentResonance => PatternType::Emergent,
                    _ => PatternType::Standing,
                }
            },
            PatternData::Coherence(_) => PatternType::Synchronous,
            PatternData::MultiFrame(_) => PatternType::Emergent,
            PatternData::Spikes(_) => PatternType::Standing,
            PatternData::Custom(_) => PatternType::Standing,
        };

        let filter = match self.content_filters.get(&pattern_type) {
            Some(f) => f,
            None => return Ok(true), // No filter means accept
        };

        // Check pattern strength
        let pattern_strength = self.get_pattern_strength(pattern_data);
        if pattern_strength < filter.min_strength {
            return Ok(false);
        }

        // Check pattern age
        let pattern_age = self.get_pattern_age(pattern_data);
        if pattern_age > filter.max_age_ms {
            return Ok(false);
        }

        Ok(true)
    }

    /// Route by priority (high priority patterns get best routes)
    async fn route_by_priority(&self, priority_pattern: &PriorityPattern) -> Result<RouteDestination> {
        let routing_table = self.routing_table.read();
        
        let pattern_type = self.get_pattern_type(&priority_pattern.pattern);
        
        if let Some(routes) = routing_table.routes.get(&pattern_type) {
            // For high priority patterns, select the best available route
            if priority_pattern.priority > 200 {
                for route in routes {
                    if let Some(health) = routing_table.route_health.get(route) {
                        if health.available && health.load < 0.7 {
                            return Ok(route.clone());
                        }
                    }
                }
            }
            
            // For normal priority, use first available route
            if let Some(route) = routes.first() {
                return Ok(route.clone());
            }
        }

        // Fallback to default route
        routing_table.default_route.clone()
            .ok_or_else(|| anyhow!("No route available"))
    }

    /// Route by content characteristics
    async fn route_by_content(&self, pattern_data: &PatternData) -> Result<RouteDestination> {
        let pattern_type = self.get_pattern_type(pattern_data);
        
        // Content-aware routing decisions
        let destination = match pattern_type {
            PatternType::Emergent => {
                // Emergent patterns need immediate ADP processing
                RouteDestination::AdpProcessor(0)
            },
            PatternType::Synchronous => {
                // Synchronous patterns benefit from DRPP analysis
                RouteDestination::DrppProcessor(0)
            },
            PatternType::Traveling => {
                // Traveling patterns need multi-timeframe analysis
                RouteDestination::MultiTimeframe(TimeHorizon::Medium)
            },
            PatternType::Chaotic => {
                // Chaotic patterns need specialized processing
                RouteDestination::External("chaos_analyzer".to_string())
            },
            PatternType::Standing => {
                // Standing patterns can use local processing
                RouteDestination::Local("standing_processor".to_string())
            },
        };

        Ok(destination)
    }

    /// Route by load balancing
    async fn route_by_load(&self, pattern_data: &PatternData) -> Result<RouteDestination> {
        let load_balancer = self.load_balancer.read();
        let pattern_type = self.get_pattern_type(pattern_data);
        
        // Get available destinations for pattern type
        let routing_table = self.routing_table.read();
        let routes = routing_table.routes.get(&pattern_type)
            .ok_or_else(|| anyhow!("No routes for pattern type"))?;

        // Find least loaded destination
        let mut best_route = None;
        let mut lowest_load = f64::INFINITY;

        for route in routes {
            if let Some(load) = load_balancer.destination_loads.get(route) {
                if *load < lowest_load {
                    lowest_load = *load;
                    best_route = Some(route.clone());
                }
            }
        }

        best_route.ok_or_else(|| anyhow!("No available route found"))
    }

    /// Route by latency optimization
    async fn route_by_latency(&self, pattern_data: &PatternData) -> Result<RouteDestination> {
        let routing_table = self.routing_table.read();
        let pattern_type = self.get_pattern_type(pattern_data);
        
        let routes = routing_table.routes.get(&pattern_type)
            .ok_or_else(|| anyhow!("No routes for pattern type"))?;

        // Find route with lowest latency
        let mut best_route = None;
        let mut lowest_latency = f64::INFINITY;

        for route in routes {
            if let Some(health) = routing_table.route_health.get(route) {
                if health.available && health.latency_us < lowest_latency {
                    lowest_latency = health.latency_us;
                    best_route = Some(route.clone());
                }
            }
        }

        best_route.ok_or_else(|| anyhow!("No low-latency route available"))
    }

    /// Route by throughput optimization
    async fn route_by_throughput(&self, pattern_data: &PatternData) -> Result<RouteDestination> {
        let routing_table = self.routing_table.read();
        let pattern_type = self.get_pattern_type(pattern_data);
        
        let routes = routing_table.routes.get(&pattern_type)
            .ok_or_else(|| anyhow!("No routes for pattern type"))?;

        // Find route with highest throughput capacity (lowest queue depth + load)
        let mut best_route = None;
        let mut highest_capacity = 0.0;

        for route in routes {
            if let Some(health) = routing_table.route_health.get(route) {
                if health.available {
                    let capacity = (1.0 - health.load) * health.success_rate / (health.queue_depth as f64 + 1.0);
                    if capacity > highest_capacity {
                        highest_capacity = capacity;
                        best_route = Some(route.clone());
                    }
                }
            }
        }

        best_route.ok_or_else(|| anyhow!("No high-throughput route available"))
    }

    /// Route using hybrid strategy
    async fn route_by_hybrid(
        &self,
        strategies: &[RoutingStrategy],
        pattern_data: &PatternData,
        priority: u32,
    ) -> Result<RouteDestination> {
        let mut candidate_routes = Vec::new();
        
        // Collect routes from each strategy
        for strategy in strategies {
            let route = match strategy {
                RoutingStrategy::ContentAware => self.route_by_content(pattern_data).await,
                RoutingStrategy::LoadAware => self.route_by_load(pattern_data).await,
                RoutingStrategy::LatencyOptimized => self.route_by_latency(pattern_data).await,
                RoutingStrategy::ThroughputOptimized => self.route_by_throughput(pattern_data).await,
                _ => continue, // Skip complex strategies in hybrid
            };
            
            if let Ok(r) = route {
                candidate_routes.push(r);
            }
        }

        if candidate_routes.is_empty() {
            return Err(anyhow!("No routes found from hybrid strategies"));
        }

        // Select best route based on combined scoring
        let routing_table = self.routing_table.read();
        let mut best_route = candidate_routes[0].clone();
        let mut best_score = 0.0;

        for route in candidate_routes {
            if let Some(health) = routing_table.route_health.get(&route) {
                if health.available {
                    // Combined score: latency, load, success rate
                    let latency_score = 1.0 / (health.latency_us + 1.0);
                    let load_score = 1.0 - health.load;
                    let success_score = health.success_rate;
                    
                    let combined_score = latency_score * 0.3 + load_score * 0.4 + success_score * 0.3;
                    
                    if combined_score > best_score {
                        best_score = combined_score;
                        best_route = route;
                    }
                }
            }
        }

        Ok(best_route)
    }

    /// Learn from routing decisions to improve future routing
    async fn learn_from_routing(
        &self,
        pattern_data: &PatternData,
        destination: &RouteDestination,
        start_time: Instant,
    ) -> Result<()> {
        let mut adaptive_learner = self.adaptive_learner.write();
        let pattern_type = self.get_pattern_type(pattern_data);
        
        // This would normally measure actual success/latency, simplified for now
        let simulated_latency = 100.0 + rand::random::<f64>() * 200.0; // 100-300 μs
        let simulated_success = rand::random::<f64>() > 0.05; // 95% success rate

        let decision = RoutingDecision {
            pattern_type,
            destination: destination.clone(),
            latency_us: simulated_latency,
            success: simulated_success,
            timestamp: start_time,
            load_at_decision: 0.5, // Simplified
        };

        // Update affinity matrix
        let key = (pattern_type, destination.clone());
        let current_affinity = adaptive_learner.affinity_matrix.get(&key).copied().unwrap_or(0.5);
        
        let reward = if simulated_success { 1.0 } else { 0.0 } / (simulated_latency / 100.0);
        let new_affinity = current_affinity + adaptive_learner.learning_rate * (reward - current_affinity);
        
        adaptive_learner.affinity_matrix.insert(key, new_affinity);

        // Add to history
        adaptive_learner.success_history.push_back(decision);
        if adaptive_learner.success_history.len() > 10000 {
            adaptive_learner.success_history.pop_front();
        }

        Ok(())
    }

    /// Update routing statistics
    async fn update_routing_stats(&self, destination: &RouteDestination, latency: Duration) -> Result<()> {
        let mut stats = self.delivery_stats.write();
        
        stats.total_patterns_routed += 1;
        stats.successful_deliveries += 1; // Simplified - assume success
        
        let latency_us = latency.as_micros() as f64;
        stats.average_latency_us = (stats.average_latency_us * (stats.total_patterns_routed - 1) as f64 + latency_us) 
            / stats.total_patterns_routed as f64;

        // Update PPS calculation
        let elapsed_since_reset = stats.last_reset.elapsed().as_secs_f64();
        if elapsed_since_reset > 0.0 {
            stats.patterns_per_second = stats.total_patterns_routed as f64 / elapsed_since_reset;
        }

        Ok(())
    }

    /// Get pattern type from pattern data
    fn get_pattern_type(&self, pattern_data: &PatternData) -> PatternType {
        match pattern_data {
            PatternData::Drpp(p) => p.pattern_type,
            PatternData::Resonance(p) => {
                match p.pattern_type {
                    ResonancePatternType::GlobalSync => PatternType::Synchronous,
                    ResonancePatternType::ChimeraState => PatternType::Chaotic,
                    ResonancePatternType::EmergentResonance => PatternType::Emergent,
                    _ => PatternType::Standing,
                }
            },
            PatternData::Coherence(_) => PatternType::Synchronous,
            PatternData::MultiFrame(_) => PatternType::Emergent,
            PatternData::Spikes(_) => PatternType::Standing,
            PatternData::Custom(_) => PatternType::Standing,
        }
    }

    /// Get pattern strength
    fn get_pattern_strength(&self, pattern_data: &PatternData) -> f64 {
        match pattern_data {
            PatternData::Drpp(p) => p.strength,
            PatternData::Resonance(p) => p.coherence_score,
            PatternData::Coherence(p) => p.coherence_score,
            PatternData::MultiFrame(r) => r.global_sync_state.network_coherence,
            PatternData::Spikes(s) => s.iter().map(|spike| spike.strength as f64).sum::<f64>() / s.len().max(1) as f64,
            PatternData::Custom(p) => p.confidence,
        }
    }

    /// Get pattern age in milliseconds
    fn get_pattern_age(&self, pattern_data: &PatternData) -> u64 {
        // Simplified - would extract actual timestamp from pattern
        50 // Assume 50ms age for now
    }

    /// Add new route to routing table
    pub async fn add_route(&self, pattern_type: PatternType, destination: RouteDestination) -> Result<()> {
        let mut routing_table = self.routing_table.write();
        
        routing_table.routes
            .entry(pattern_type)
            .or_insert_with(Vec::new)
            .push(destination.clone());

        // Initialize health status
        routing_table.route_health.insert(destination, RouteHealth {
            load: 0.0,
            latency_us: 100.0,
            success_rate: 1.0,
            queue_depth: 0,
            last_check: Instant::now(),
            available: true,
        });

        routing_table.last_updated = Instant::now();
        Ok(())
    }

    /// Get routing statistics
    pub fn get_routing_stats(&self) -> DeliveryStatistics {
        self.delivery_stats.read().clone()
    }

    /// Get current routing table
    pub fn get_routing_table(&self) -> RoutingTable {
        self.routing_table.read().clone()
    }
}

impl Ord for PriorityPattern {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for PriorityPattern {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PriorityPattern {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PriorityPattern {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drpp::Pattern;
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_pattern_routing_engine_creation() {
        let engine = PatternRoutingEngine::new(RoutingStrategy::Priority);
        let stats = engine.get_routing_stats();
        
        assert_eq!(stats.total_patterns_routed, 0);
        assert_eq!(stats.successful_deliveries, 0);
    }

    #[tokio::test]
    async fn test_add_route() {
        let engine = PatternRoutingEngine::new(RoutingStrategy::ContentAware);
        
        let result = engine.add_route(
            PatternType::Emergent,
            RouteDestination::AdpProcessor(0),
        ).await;
        
        assert!(result.is_ok());
        
        let routing_table = engine.get_routing_table();
        assert!(routing_table.routes.contains_key(&PatternType::Emergent));
    }

    #[tokio::test]
    async fn test_content_based_routing() {
        let mut engine = PatternRoutingEngine::new(RoutingStrategy::ContentAware);
        
        // Add routes
        engine.add_route(PatternType::Emergent, RouteDestination::AdpProcessor(0)).await.unwrap();
        engine.add_route(PatternType::Synchronous, RouteDestination::DrppProcessor(0)).await.unwrap();
        
        // Test emergent pattern routing
        let pattern = PatternData::Drpp(Pattern {
            id: 1,
            pattern_type: PatternType::Emergent,
            strength: 0.9,
            timestamp: SystemTime::now(),
            oscillators: vec![],
            phase_coherence: 0.8,
            frequency_content: vec![],
        });
        
        let destination = engine.route_pattern(pattern, 100).await.unwrap();
        assert!(matches!(destination, RouteDestination::AdpProcessor(0)));
    }

    #[test]
    fn test_priority_pattern_ordering() {
        let mut queue = BinaryHeap::new();
        
        let low_priority = PriorityPattern {
            pattern: PatternData::Custom(CustomPattern {
                pattern_type: "test".to_string(),
                data: vec![1.0],
                metadata: HashMap::new(),
                confidence: 0.5,
            }),
            priority: 100,
            timestamp: Instant::now(),
            expires_at: None,
            retry_count: 0,
            source_id: "test".to_string(),
        };
        
        let high_priority = PriorityPattern {
            pattern: PatternData::Custom(CustomPattern {
                pattern_type: "test".to_string(),
                data: vec![1.0],
                metadata: HashMap::new(),
                confidence: 0.8,
            }),
            priority: 255,
            timestamp: Instant::now(),
            expires_at: None,
            retry_count: 0,
            source_id: "test".to_string(),
        };
        
        queue.push(low_priority);
        queue.push(high_priority);
        
        let first = queue.pop().unwrap();
        assert_eq!(first.priority, 255);
    }
}