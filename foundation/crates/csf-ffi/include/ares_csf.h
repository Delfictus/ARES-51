#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "stdint.h"
#include "stdbool.h"

/**
 * FFI representation of a phase packet
 */
typedef struct CSF_Packet {
  uint64_t packet_id_high;
  uint64_t packet_id_low;
  uint8_t packet_type;
  uint8_t priority;
  uint16_t flags;
  uint64_t timestamp;
  uint16_t source_node;
  uint16_t destination_node;
  uint64_t causality_hash;
  uint8_t *data;
  uintptr_t data_len;
  char *metadata;
} CSF_Packet;

/**
 * FFI representation of an MLIR module
 */
typedef struct CSF_MLIRModule {
  uint64_t module_id;
  char *name;
  char *mlir_code;
  bool is_compiled;
} CSF_MLIRModule;

/**
 * FFI representation of a tensor
 */
typedef struct CSF_Tensor {
  uint8_t *data;
  uintptr_t data_len;
  uint8_t dtype;
  int64_t *shape;
  uintptr_t shape_len;
  int64_t *strides;
  uintptr_t strides_len;
  uint8_t layout;
} CSF_Tensor;

/**
 * Kernel configuration
 */
typedef struct CSF_KernelConfig {
  uint64_t tick_interval_ns;
  uint32_t max_tasks;
  uint32_t thread_pool_size;
  bool enable_real_time;
} CSF_KernelConfig;

/**
 * FFI representation of C-LOGIC state
 */
typedef struct CSF_CLogicState {
  double drpp_coherence;
  double adp_load;
  uint32_t egc_decisions_pending;
  double ems_valence;
  double ems_arousal;
  uint64_t timestamp;
} CSF_CLogicState;

/**
 * FFI representation of a quantum operation
 */
typedef struct CSF_QuantumOp {
  uint8_t op_type;
  uint32_t qubit1;
  uint32_t qubit2;
  double param1;
  double param2;
  double param3;
} CSF_QuantumOp;

/**
 * FFI representation of a quantum circuit
 */
typedef struct CSF_QuantumCircuit {
  uint32_t num_qubits;
  struct CSF_QuantumOp *operations;
  uintptr_t num_operations;
  uint32_t *measurements;
  uintptr_t num_measurements;
} CSF_QuantumCircuit;

/**
 * Profiling data
 */
typedef struct CSF_ProfilingData {
  uint64_t total_packets;
  uint64_t total_tasks;
  uint64_t avg_latency_ns;
  double throughput_pps;
  double cpu_usage_percent;
  double memory_usage_mb;
} CSF_ProfilingData;

/**
 * Initialize the CSF runtime
 *
 * This must be called before any other CSF functions.
 * Returns 0 on success, negative error code on failure.
 */
int32_t csf_init(void);

/**
 * Shutdown the CSF runtime
 *
 * This should be called when done using CSF.
 */
int32_t csf_shutdown(void);

/**
 * Get the CSF version string
 */
const char *csf_version(void);

/**
 * Free a string returned by CSF
 */
void csf_free_string(char *s);

/**
 * Create a new phase packet
 */
struct CSF_Packet *csf_packet_create(uint8_t packet_type,
                                     uint8_t priority,
                                     const uint8_t *data,
                                     uintptr_t data_len);

/**
 * Destroy a phase packet
 */
void csf_packet_destroy(struct CSF_Packet *packet);

/**
 * Send a packet through the bus
 */
int32_t csf_bus_send(const char *channel_name, const struct CSF_Packet *packet);

/**
 * Create a new MLIR module
 */
struct CSF_MLIRModule *csf_mlir_module_create(const char *name, const char *mlir_code);

/**
 * Compile and load an MLIR module
 */
int32_t csf_mlir_module_load(struct CSF_MLIRModule *module);

/**
 * Execute an MLIR module
 */
int32_t csf_mlir_module_execute(uint64_t module_id,
                                const struct CSF_Tensor *inputs,
                                uintptr_t num_inputs,
                                struct CSF_Tensor *outputs,
                                uintptr_t num_outputs);

/**
 * Create a new temporal kernel with comprehensive safety validation
 *
 * # Safety
 *
 * This function is memory-safe and validates all inputs before processing.
 * Returns null on any validation failure or allocation error.
 *
 * # Arguments
 *
 * * `config` - Optional configuration pointer. If null, uses safe defaults.
 *
 * # Returns
 *
 * * Valid kernel pointer on success
 * * Null pointer on failure (check logs for details)
 */
void *csf_kernel_create(const struct CSF_KernelConfig *config);

/**
 * Safely destroy a temporal kernel with proper cleanup
 *
 * # Safety
 *
 * This function safely handles null pointers and ensures proper cleanup
 * of all kernel resources without memory leaks or double-free errors.
 *
 * # Arguments
 *
 * * `kernel` - Kernel pointer from csf_kernel_create(). Safe to pass null.
 */
void csf_kernel_destroy(void *kernel);

/**
 * Schedule a task on the kernel with comprehensive input validation
 *
 * # Safety
 *
 * This function validates all inputs and handles errors gracefully.
 * All pointer operations are bounds-checked and memory-safe.
 *
 * # Arguments
 *
 * * `kernel` - Valid kernel pointer from csf_kernel_create()
 * * `task_name` - Null-terminated C string with task name (max 4096 bytes)
 * * `priority` - Task priority (0-255)
 * * `deadline_ns` - Relative deadline in nanoseconds
 *
 * # Returns
 *
 * * Task ID (>0) on success
 * * 0 on failure
 */
uint64_t csf_kernel_schedule_task(void *kernel,
                                  const char *task_name,
                                  uint8_t priority,
                                  uint64_t deadline_ns);

/**
 * Get C-LOGIC system state with safe memory operations
 *
 * # Safety
 *
 * This function validates the output buffer and writes safe default values.
 * All floating-point operations are checked for validity.
 *
 * # Arguments
 *
 * * `state` - Output buffer for C-LOGIC state (must be valid CSF_CLogicState*)
 *
 * # Returns
 *
 * * CSF_ErrorCode::Success on success
 * * Error code on failure
 */
int32_t csf_clogic_get_state(struct CSF_CLogicState *state);

/**
 * Create a quantum circuit with input validation
 *
 * # Safety
 *
 * This function validates the qubit count and creates a properly initialized
 * circuit structure with safe default values.
 *
 * # Arguments
 *
 * * `num_qubits` - Number of qubits (must be > 0 and <= MAX_QUBITS)
 *
 * # Returns
 *
 * * Valid circuit pointer on success
 * * Null pointer on failure
 */
struct CSF_QuantumCircuit *csf_quantum_circuit_create(uint32_t num_qubits);

/**
 * Add a quantum gate to circuit with memory-safe allocation
 *
 * # Safety
 *
 * This function uses safe Rust Vec allocation instead of raw libc operations
 * to prevent memory corruption, double-free, and allocation failures.
 *
 * # Arguments
 *
 * * `circuit` - Valid circuit pointer from csf_quantum_circuit_create()
 * * `op` - Quantum operation to add
 *
 * # Returns
 *
 * * 0 on success
 * * CSF_ErrorCode on failure
 */
int32_t csf_quantum_circuit_add_gate(struct CSF_QuantumCircuit *circuit, struct CSF_QuantumOp op);

/**
 * Execute a quantum circuit with bounds-checked array operations
 *
 * # Safety
 *
 * This function validates all array bounds and prevents buffer overflows.
 * All memory operations are checked for safety before execution.
 *
 * # Arguments
 *
 * * `circuit` - Valid circuit pointer
 * * `shots` - Number of quantum measurement shots (must be > 0 and <= 1M)
 * * `results` - Output buffer for measurement results (must have size >= 2^num_qubits * sizeof(u32))
 * * `probabilities` - Output buffer for state probabilities (must have size >= 2^num_qubits * sizeof(f64))
 *
 * # Returns
 *
 * * CSF_ErrorCode::Success on success
 * * Error code on failure
 */
int32_t csf_quantum_circuit_execute(const struct CSF_QuantumCircuit *circuit,
                                    uint32_t shots,
                                    uint32_t *results,
                                    double *probabilities);

/**
 * Safely destroy a quantum circuit with proper memory cleanup
 *
 * # Safety
 *
 * This function properly handles all cleanup in the correct order to prevent
 * memory leaks, double-free errors, and use-after-free vulnerabilities.
 *
 * # Arguments
 *
 * * `circuit` - Circuit pointer from csf_quantum_circuit_create(). Safe to pass null.
 */
void csf_quantum_circuit_destroy(struct CSF_QuantumCircuit *circuit);

/**
 * Get profiling data with safe output buffer handling
 *
 * # Safety
 *
 * This function validates the output buffer and writes realistic profiling
 * data with proper bounds checking.
 *
 * # Arguments
 *
 * * `data` - Output buffer for profiling data (must be valid CSF_ProfilingData*)
 *
 * # Returns
 *
 * * CSF_ErrorCode::Success on success
 * * Error code on failure
 */
int32_t csf_get_profiling_data(struct CSF_ProfilingData *data);

/**
 * Get the last error message
 */
const char *csf_get_last_error(void);
