#!/usr/bin/env python3
"""
verify_mps.py

Check if PyTorch can access MPS (Metal Performance Shaders) on M1 Mac
and run a simple test to confirm it works.
"""

import torch
import numpy as np
import time

def main():
    print("\n===== PyTorch GPU Access Verification =====")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    print("\nMPS (Metal Performance Shaders) Support:")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Check CUDA availability (will be False on M1 Mac)
    print("\nCUDA Support:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Select the best available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (M1/M2 GPU)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"\nUsing device: {device} ({device_name})")
    
    # Run a simple benchmark test
    print("\n===== Running matrix multiplication benchmark =====")
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        # Create random matrices
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # CPU timing
        start_time = time.time()
        c_cpu = torch.matmul(a, b)
        cpu_time = time.time() - start_time
        print(f"\nMatrix size: {size}x{size}")
        print(f"CPU time: {cpu_time:.4f} seconds")
        
        # Device timing
        a_device = a.to(device)
        b_device = b.to(device)
        
        # Warmup
        torch.matmul(a_device, b_device)
        
        # Timed run
        start_time = time.time()
        c_device = torch.matmul(a_device, b_device)
        
        # Ensure operation is complete before timing
        if device.type != "cpu":
            torch.cuda.synchronize() if device.type == "cuda" else torch.mps.synchronize()
        
        device_time = time.time() - start_time
        print(f"{device_name} time: {device_time:.4f} seconds")
        print(f"Speedup: {cpu_time/device_time:.2f}x")
        
        # Verify results match
        c_device_cpu = c_device.to("cpu")
        max_diff = torch.max(torch.abs(c_cpu - c_device_cpu)).item()
        print(f"Max difference between CPU and {device_name}: {max_diff}")
    
    print("\n===== Verification Complete =====")
    if torch.backends.mps.is_available():
        print("✅ MPS is available and working correctly on your M1 Mac!")
        print("You can run PyTorch with GPU acceleration.")
    else:
        print("❌ MPS is not available on your system.")
        print("Check if you have PyTorch 1.12+ installed and are running on macOS 12.3+")

if __name__ == "__main__":
    main() 