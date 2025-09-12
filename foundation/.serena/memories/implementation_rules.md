# ARES ChronoFabric Implementation Rules

## ABSOLUTE IMPLEMENTATION REQUIREMENTS

### Rule 1: Zero Tolerance for Placeholders
- **NEVER** use `todo!()`, `unimplemented!()`, or placeholder implementations
- **NEVER** use mock data, fake data, or dummy data in production code
- **NEVER** use stub implementations or commented-out functionality
- **ALWAYS** implement complete, working algorithms from first principles

### Rule 2: Mathematical Completeness
- **ALWAYS** implement algorithms with full mathematical rigor
- **ALWAYS** provide complete eigenvalue decompositions using LAPACK/BLAS
- **ALWAYS** implement full quantum state evolution with proper Hamiltonian dynamics
- **ALWAYS** provide complete statistical analysis with proper error bounds
- **NEVER** use simplified approximations without mathematical justification

### Rule 3: Hardware Integration Authenticity  
- **ALWAYS** use actual hardware APIs (CUDA Driver API, Vulkan, etc.)
- **ALWAYS** implement real device detection and capability querying
- **ALWAYS** provide actual performance monitoring and thermal management
- **NEVER** use hardcoded values for hardware properties

### Rule 4: Network Protocol Completeness
- **ALWAYS** implement complete network protocol stacks
- **ALWAYS** provide real encryption and authentication
- **ALWAYS** implement actual connection pooling and load balancing
- **NEVER** skip error handling or timeout mechanisms

### Rule 5: Data Processing Authenticity
- **ALWAYS** use real external APIs with proper authentication
- **ALWAYS** implement complete data validation and sanitization
- **ALWAYS** provide actual streaming and buffering mechanisms
- **NEVER** use simulated or generated data in place of real feeds

### Rule 6: Temporal Processing Precision
- **ALWAYS** implement nanosecond-precision timing using hardware counters
- **ALWAYS** provide actual causality checking and temporal ordering
- **ALWAYS** implement real-time scheduling with mathematical guarantees
- **NEVER** use standard library sleep or timeout functions for critical timing

### Rule 7: Testing and Validation
- **ALWAYS** provide mathematical proofs of algorithm correctness
- **ALWAYS** implement property-based testing for all algorithms
- **ALWAYS** provide performance benchmarks with statistical significance
- **NEVER** accept "it works" without rigorous validation

### Rule 8: Error Handling Completeness
- **ALWAYS** implement comprehensive error recovery mechanisms  
- **ALWAYS** provide detailed error diagnostics with context
- **ALWAYS** implement circuit breakers and fallback strategies
- **NEVER** use generic error types without specific handling

## Implementation Verification Checklist
Before any code is considered complete:
- [ ] No placeholder text or TODO comments
- [ ] All algorithms mathematically complete and verified
- [ ] Hardware integration uses actual vendor APIs
- [ ] Network protocols fully implemented with security
- [ ] Data sources are real external systems
- [ ] Temporal precision uses hardware timing
- [ ] Comprehensive test coverage with proofs
- [ ] Complete error handling and recovery