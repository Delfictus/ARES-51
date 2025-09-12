#!/usr/bin/env python3
"""
Example Python program using ARES CSF
"""

import ares_csf
import numpy as np
import asyncio
import json

def main():
    print("ARES CSF Python Example")
    print(f"Version: {ares_csf.get_version()}")
    
    # Initialize runtime
    ares_csf.init_runtime()
    print("CSF runtime initialized")
    
    # Create runtime instance
    runtime = ares_csf.PyCSFRuntime()
    
    # Create and send a phase packet
    packet = ares_csf.PyPhasePacket(
        packet_type=1,  # Data packet
        priority=128,    # Medium priority
        data=bytes([0x01, 0x02, 0x03, 0x04])
    )
    
    # Add metadata
    packet.metadata["source"] = "python_example"
    packet.metadata["timestamp"] = 12345
    
    print(f"Created packet with type={packet.packet_type}, priority={packet.priority}")
    
    # Send packet
    runtime.send_packet("example.channel", packet)
    print("Packet sent successfully")
    
    # Create and compile an MLIR module
    mlir_code = """
    func.func @matmul(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
        %C = linalg.matmul ins(%A, %B : tensor<2x3xf32>, tensor<3x4xf32>)
                          outs(%C : tensor<2x4xf32>) -> tensor<2x4xf32>
        return %C : tensor<2x4xf32>
    }
    """
    
    module = ares_csf.PyMLIRModule("matmul_module", mlir_code)
    print(f"Created MLIR module: {module.name}")
    
    # Load the module
    module_id = runtime.load_mlir_module(module)
    print(f"MLIR module loaded with ID: {module_id}")
    
    # Create input tensors
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    B = np.array([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]], dtype=np.float32)
    
    tensor_A = ares_csf.PyTensor(A.flatten().tolist(), list(A.shape))
    tensor_B = ares_csf.PyTensor(B.flatten().tolist(), list(B.shape))
    
    # Execute the module
    outputs = runtime.execute_mlir_module(module_id, [tensor_A, tensor_B])
    print("MLIR module executed successfully")
    
    # Get result as numpy array
    result = outputs[0].numpy()
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result}")
    
    # Expected result
    expected = np.matmul(A, B)
    print(f"Expected:\n{expected}")
    
    # Create a quantum circuit
    circuit = ares_csf.PyQuantumCircuit(3)
    
    # Create GHZ state
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    
    print("Created 3-qubit GHZ state circuit")
    
    # Execute circuit
    results = circuit.execute(1000)
    print(f"Quantum circuit results: {results}")
    
    # Get C-LOGIC state
    clogic = ares_csf.PyCLogicSystem()
    state = clogic.get_state()
    print("\nC-LOGIC State:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Demonstrate async packet receiving
    async def receive_packets():
        # Try to receive a packet (with timeout)
        received = runtime.receive_packet("example.channel", timeout_ms=100)
        if received:
            print(f"\nReceived packet: type={received.packet_type}, data={received.data}")
        else:
            print("\nNo packet received (timeout)")
    
    # Run async operation
    asyncio.run(receive_packets())
    
    # Create a more complex quantum circuit
    print("\nCreating quantum algorithm circuit...")
    qft_circuit = ares_csf.PyQuantumCircuit(4)
    
    # Simple QFT-like circuit
    for i in range(4):
        qft_circuit.h(i)
        for j in range(i+1, 4):
            angle = np.pi / (2 ** (j - i))
            qft_circuit.rx(j, angle)
    
    qft_results = qft_circuit.execute(2000)
    print(f"QFT-like circuit results: {qft_results['shots']} shots")
    
    # Demonstrate tensor operations
    print("\nTensor operations example:")
    
    # Create random tensors
    x = ares_csf.PyTensor(
        np.random.randn(10, 20).flatten().tolist(),
        [10, 20]
    )
    
    y = ares_csf.PyTensor(
        np.random.randn(20, 30).flatten().tolist(),
        [20, 30]
    )
    
    print(f"Tensor X shape: {x.shape}")
    print(f"Tensor Y shape: {y.shape}")
    
    # Create an MLIR module for the operation
    tensor_mlir = """
    func.func @tensor_op(%X: tensor<10x20xf32>, %Y: tensor<20x30xf32>) -> tensor<10x30xf32> {
        %Z = linalg.matmul ins(%X, %Y : tensor<10x20xf32>, tensor<20x30xf32>)
                          outs(%Z : tensor<10x30xf32>) -> tensor<10x30xf32>
        return %Z : tensor<10x30xf32>
    }
    """
    
    tensor_module = ares_csf.PyMLIRModule("tensor_op", tensor_mlir)
    tensor_module_id = runtime.load_mlir_module(tensor_module)
    
    # Execute
    z_results = runtime.execute_mlir_module(tensor_module_id, [x, y])
    z = z_results[0]
    print(f"Result tensor shape: {z.shape}")
    
    # Performance test
    print("\nPerformance test:")
    import time
    
    num_iterations = 100
    start_time = time.time()
    
    for i in range(num_iterations):
        packet = ares_csf.PyPhasePacket(1, 100, bytes(1024))
        runtime.send_packet(f"perf.test.{i}", packet)
    
    elapsed = time.time() - start_time
    throughput = num_iterations / elapsed
    
    print(f"Sent {num_iterations} packets in {elapsed:.3f} seconds")
    print(f"Throughput: {throughput:.1f} packets/second")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()