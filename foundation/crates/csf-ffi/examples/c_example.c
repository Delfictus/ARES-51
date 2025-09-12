/**
 * Example C program using ARES CSF
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../include/ares_csf.h"

void print_error_and_exit(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    const char* last_error = csf_get_last_error();
    if (last_error) {
        fprintf(stderr, "Last error: %s\n", last_error);
        csf_free_string((char*)last_error);
    }
    exit(1);
}

int main() {
    printf("ARES CSF C Example\n");
    printf("Version: %s\n", csf_version());
    
    // Initialize runtime
    if (csf_init() != 0) {
        print_error_and_exit("Failed to initialize CSF runtime");
    }
    
    printf("CSF runtime initialized\n");
    
    // Create a phase packet
    uint8_t data[] = {0x01, 0x02, 0x03, 0x04};
    CSF_Packet* packet = csf_packet_create(
        1,  // Data packet
        128, // Medium priority
        data,
        sizeof(data)
    );
    
    if (!packet) {
        print_error_and_exit("Failed to create packet");
    }
    
    printf("Created packet with ID: %llu%llu\n", 
           packet->packet_id_high, packet->packet_id_low);
    
    // Send packet through bus
    int result = csf_bus_send("example.channel", packet);
    if (result != 0) {
        csf_packet_destroy(packet);
        print_error_and_exit("Failed to send packet");
    }
    
    printf("Packet sent successfully\n");
    
    // Create and compile an MLIR module
    const char* mlir_code = 
        "func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {\n"
        "  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>\n"
        "  return %0 : tensor<4xf32>\n"
        "}";
    
    CSF_MLIRModule* module = csf_mlir_module_create("add_module", mlir_code);
    if (!module) {
        csf_packet_destroy(packet);
        print_error_and_exit("Failed to create MLIR module");
    }
    
    printf("Created MLIR module: %s\n", module->name);
    
    // Load the module
    if (csf_mlir_module_load(module) != 0) {
        csf_packet_destroy(packet);
        print_error_and_exit("Failed to load MLIR module");
    }
    
    printf("MLIR module loaded with ID: %llu\n", module->module_id);
    
    // Create input tensors
    float input_data1[] = {1.0, 2.0, 3.0, 4.0};
    float input_data2[] = {5.0, 6.0, 7.0, 8.0};
    int64_t shape[] = {4};
    
    CSF_Tensor inputs[2] = {
        {
            .data = (uint8_t*)input_data1,
            .data_len = sizeof(input_data1),
            .dtype = 1, // F32
            .shape = shape,
            .shape_len = 1,
            .strides = shape,
            .strides_len = 1,
            .layout = 0 // CPU
        },
        {
            .data = (uint8_t*)input_data2,
            .data_len = sizeof(input_data2),
            .dtype = 1, // F32
            .shape = shape,
            .shape_len = 1,
            .strides = shape,
            .strides_len = 1,
            .layout = 0 // CPU
        }
    };
    
    CSF_Tensor output;
    
    // Execute the module
    if (csf_mlir_module_execute(module->module_id, inputs, 2, &output, 1) != 0) {
        csf_packet_destroy(packet);
        print_error_and_exit("Failed to execute MLIR module");
    }
    
    printf("MLIR module executed successfully\n");
    
    // Print results
    float* output_data = (float*)output.data;
    printf("Result: [");
    for (int i = 0; i < 4; i++) {
        printf("%.1f%s", output_data[i], i < 3 ? ", " : "");
    }
    printf("]\n");
    
    // Create a quantum circuit
    CSF_QuantumCircuit* circuit = csf_quantum_circuit_create(2);
    if (!circuit) {
        csf_packet_destroy(packet);
        print_error_and_exit("Failed to create quantum circuit");
    }
    
    // Add gates
    CSF_QuantumOp h_gate = {
        .op_type = 0, // H gate
        .qubit1 = 0,
        .qubit2 = 0,
        .param1 = 0.0,
        .param2 = 0.0,
        .param3 = 0.0
    };
    csf_quantum_circuit_add_gate(circuit, h_gate);
    
    CSF_QuantumOp cnot_gate = {
        .op_type = 6, // CNOT
        .qubit1 = 0,
        .qubit2 = 1,
        .param1 = 0.0,
        .param2 = 0.0,
        .param3 = 0.0
    };
    csf_quantum_circuit_add_gate(circuit, cnot_gate);
    
    printf("Created quantum circuit with %d qubits and %zu operations\n",
           circuit->num_qubits, circuit->num_operations);
    
    // Execute quantum circuit
    uint32_t results[4];
    double probabilities[4];
    
    if (csf_quantum_circuit_execute(circuit, 1000, results, probabilities) != 0) {
        csf_packet_destroy(packet);
        csf_quantum_circuit_destroy(circuit);
        print_error_and_exit("Failed to execute quantum circuit");
    }
    
    printf("Quantum circuit execution results:\n");
    for (int i = 0; i < 4; i++) {
        printf("  State %d: %.3f probability\n", results[i], probabilities[i]);
    }
    
    // Get C-LOGIC state
    CSF_CLogicState clogic_state;
    if (csf_clogic_get_state(&clogic_state) == 0) {
        printf("\nC-LOGIC State:\n");
        printf("  DRPP Coherence: %.3f\n", clogic_state.drpp_coherence);
        printf("  ADP Load: %.3f\n", clogic_state.adp_load);
        printf("  EGC Decisions: %u\n", clogic_state.egc_decisions_pending);
        printf("  EMS Valence: %.3f\n", clogic_state.ems_valence);
        printf("  EMS Arousal: %.3f\n", clogic_state.ems_arousal);
    }
    
    // Get profiling data
    CSF_ProfilingData profiling;
    if (csf_get_profiling_data(&profiling) == 0) {
        printf("\nProfiling Data:\n");
        printf("  Total Packets: %llu\n", profiling.total_packets);
        printf("  Total Tasks: %llu\n", profiling.total_tasks);
        printf("  Avg Latency: %llu ns\n", profiling.avg_latency_ns);
        printf("  Throughput: %.2f packets/sec\n", profiling.throughput_pps);
        printf("  CPU Usage: %.1f%%\n", profiling.cpu_usage_percent);
        printf("  Memory Usage: %.1f MB\n", profiling.memory_usage_mb);
    }
    
    // Cleanup
    csf_packet_destroy(packet);
    csf_quantum_circuit_destroy(circuit);
    
    // Shutdown runtime
    if (csf_shutdown() != 0) {
        print_error_and_exit("Failed to shutdown CSF runtime");
    }
    
    printf("\nCSF runtime shutdown successfully\n");
    
    return 0;
}