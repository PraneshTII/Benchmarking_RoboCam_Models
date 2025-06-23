#!/usr/bin/env python3
"""
NPU Performance Diagnostics - Find out why inference is so slow
 Perfoms inference on cpu vs NPU. 
- By default uses yolov4 tflite model
- Check for warmup time.
Usage: 
   python3 npu_performance_diagnostics.py
"""
import sys;import site;import functools;sys.argv[0] = '/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/bin/tf_lite_test.py';functools.reduce(lambda k, p: site.addsitedir(p, k), ['/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/s31xv27jkg2k2qsgs62spd5cgb3h9ira-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/8l32kk3qvcda079lv8506r71h6hl86pn-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/qn2z3wnqi8knn84chryz0iww5kkshaz7-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/rrjjkwkcx1fy43md458yd7avcr0hy0q2-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());
import os
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

def test_cpu_vs_npu_performance(model_path):
    """Compare CPU vs NPU performance to see if NPU is really working"""
    
    print("NPU Performance Diagnostics")
    print("=" * 50)
    
    # Test 1: CPU-only inference
    print("\n1. Testing CPU-only inference...")
    try:
        cpu_interpreter = tflite.Interpreter(model_path=model_path)
        cpu_interpreter.allocate_tensors()
        
        input_details = cpu_interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        
        # Create test input
        if input_details[0]['dtype'] == np.int8:
            test_input = np.random.randint(-128, 127, input_shape, dtype=np.int8)
        else:
            test_input = np.random.random(input_shape).astype(np.float32)
        
        # Warm up
        cpu_interpreter.set_tensor(input_details[0]['index'], test_input)
        cpu_interpreter.invoke()
        
        # Time multiple inferences
        num_runs = 5
        start_time = time.time()
        for _ in range(num_runs):
            cpu_interpreter.set_tensor(input_details[0]['index'], test_input)
            cpu_interpreter.invoke()
        cpu_total_time = (time.time() - start_time) * 1000  # ms
        cpu_avg_time = cpu_total_time / num_runs
        
        print(f"  CPU average inference time: {cpu_avg_time:.1f} ms")
        
    except Exception as e:
        print(f"  CPU test failed: {e}")
        cpu_avg_time = None
    
    # Test 2: NPU inference
    print("\n2. Testing NPU inference...")
    try:
        npu_interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tflite.load_delegate('/nix/store/nxds5qr80pafcvcx9yvsvn7wg14qk7qg-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
            ]
        )
        npu_interpreter.allocate_tensors()
        
        # Warm up
        npu_interpreter.set_tensor(input_details[0]['index'], test_input)
        npu_interpreter.invoke()
        
        # Time multiple inferences
        start_time = time.time()
        for _ in range(num_runs):
            npu_interpreter.set_tensor(input_details[0]['index'], test_input)
            npu_interpreter.invoke()
        npu_total_time = (time.time() - start_time) * 1000  # ms
        npu_avg_time = npu_total_time / num_runs
        
        print(f"  NPU average inference time: {npu_avg_time:.1f} ms")
        
    except Exception as e:
        print(f"  NPU test failed: {e}")
        npu_avg_time = None
    
    # Test 3: First inference timing (cold start)
    print("\n3. Testing cold start performance...")
    try:
        cold_interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tflite.load_delegate('/nix/store/nxds5qr80pafcvcx9yvsvn7wg14qk7qg-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
            ]
        )
        cold_interpreter.allocate_tensors()
        
        # Time first inference (cold start)
        start_time = time.time()
        cold_interpreter.set_tensor(input_details[0]['index'], test_input)
        cold_interpreter.invoke()
        cold_start_time = (time.time() - start_time) * 1000
        
        print(f"  Cold start inference time: {cold_start_time:.1f} ms")
        
    except Exception as e:
        print(f"  Cold start test failed: {e}")
        cold_start_time = None
    
    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS:")
    
    if cpu_avg_time and npu_avg_time:
        if npu_avg_time < cpu_avg_time * 0.8:
            print(f"✓ NPU is working! {cpu_avg_time/npu_avg_time:.1f}x speedup")
        elif npu_avg_time > cpu_avg_time * 2:
            print(f"✗ NPU is slower than CPU! Something is wrong.")
        else:
            print(f"? NPU and CPU performance are similar. NPU may not be used.")
    
    if cold_start_time and cold_start_time > 1000:
        print(f"✗ Cold start is very slow ({cold_start_time:.1f}ms) - model compilation issue")
    
    if npu_avg_time and npu_avg_time > 100:
        print(f"✗ NPU inference is too slow ({npu_avg_time:.1f}ms) - should be <50ms for YOLOv4-tiny")
    
    return cpu_avg_time, npu_avg_time, cold_start_time

def test_delegate_info(model_path):
    """Check what operations are running on NPU vs CPU"""
    
    print("\n4. Checking delegate operation mapping...")
    
    try:
        # Load with delegate and verbose logging
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tflite.load_delegate('/nix/store/nxds5qr80pafcvcx9yvsvn7wg14qk7qg-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
            ]
        )
        interpreter.allocate_tensors()
        
        # Get execution plan
        print("  Model loaded with VX delegate")
        
        # Check if we can get more info about delegate usage
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input tensors: {len(input_details)}")
        print(f"  Output tensors: {len(output_details)}")
        
        # Run a test inference to see if there are fallback warnings
        input_shape = input_details[0]['shape']
        if input_details[0]['dtype'] == np.int8:
            test_input = np.random.randint(-128, 127, input_shape, dtype=np.int8)
        else:
            test_input = np.random.random(input_shape).astype(np.float32)
        
        print("  Running test inference (watch for delegate warnings)...")
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        print("  Test inference completed")
        
    except Exception as e:
        print(f"  Delegate info check failed: {e}")

def test_memory_usage():
    """Check memory usage during inference"""
    
    print("\n5. Checking system resources...")
    
    try:
        # Check available memory
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        for line in meminfo.split('\n'):
            if 'MemAvailable' in line or 'MemFree' in line:
                print(f"  {line}")
        
        # Check CPU usage
        with open('/proc/loadavg', 'r') as f:
            loadavg = f.read().strip()
            print(f"  Load average: {loadavg}")
        
    except Exception as e:
        print(f"  Resource check failed: {e}")

def test_different_input_sizes(model_path):
    """Test if smaller input improves performance"""
    
    print("\n6. Testing performance with different input preprocessing...")
    
    try:
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tflite.load_delegate('/nix/store/nxds5qr80pafcvcx9yvsvn7wg14qk7qg-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
            ]
        )
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        
        # Test with pre-allocated input
        if input_details[0]['dtype'] == np.int8:
            test_input = np.random.randint(-128, 127, input_shape, dtype=np.int8)
        else:
            test_input = np.random.random(input_shape).astype(np.float32)
        
        # Test inference with minimal operations
        times = []
        for i in range(10):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000
            times.append(inference_time)
            
            if i < 3:
                print(f"  Inference {i+1}: {inference_time:.1f} ms")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"  Average: {avg_time:.1f} ms")
        print(f"  Range: {min_time:.1f} - {max_time:.1f} ms")
        
        if max_time > min_time * 2:
            print(f"  ⚠ High variance suggests inconsistent performance")
        
    except Exception as e:
        print(f"  Performance test failed: {e}")

def main():
    model_path = "/home/scmd/yolo-v4-tiny.tflite"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Run all diagnostic tests
    cpu_time, npu_time, cold_time = test_cpu_vs_npu_performance(model_path)
    test_delegate_info(model_path)
    test_memory_usage()
    test_different_input_sizes(model_path)
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    
    if npu_time and npu_time > 1000:
        print("1. NPU performance is unacceptably slow")
        print("   - Try running without NPU: --no_npu flag")
        print("   - Check if model has unsupported operations")
        print("   - Verify VX delegate installation")
    
    if cold_time and cold_time > 5000:
        print("2. Cold start is extremely slow")
        print("   - Model may have compilation issues")
        print("   - Try a different model or quantization")
    
    print("3. Expected performance for YOLOv4-tiny on IMX8MP NPU:")
    print("   - Should be 10-50ms per inference")
    print("   - 20+ FPS should be achievable")
    
    if cpu_time and cpu_time < 200:
        print("4. CPU performance is reasonable")
        print("   - Consider using CPU-only mode if NPU is problematic")

if __name__ == "__main__":
    main()
