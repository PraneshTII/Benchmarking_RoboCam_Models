#!/usr/bin/env python3
"""
TensorFlow Lite Model Benchmarking Script
Benchmarks model execution time on CPU vs NPU (VX Delegate)
"""
import sys
import site
import functools

# Nix store paths for dependencies
functools.reduce(lambda k, p: site.addsitedir(p, k), 
['/nix/store/7wdal3kjsp1hn6ig64m2y6baxg4f929h-tflite-opencv-test-app-cpu-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/1bvpxg3kvzjhrp7n9q114xcjxmyx2ik8-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/d5hmgjyy2wy17k79z1j1gjs0fv1wh5ki-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/n60lsdw7cm43lv10bdrcdcxqv7dpnn0b-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/s58a7qp81ms8an7xdc7h008d8kc13kqp-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo())

import os
import argparse
import cv2
import numpy as np
import time
import statistics
import json
from threading import Thread
import tflite_runtime.interpreter as tflite
import traceback
from datetime import datetime

# VX Delegate path
VX_DELEGATE_PATH = '/nix/store/96bsy96b042wsqgzazpdhcdkqhai9k7n-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so'

# List of model files to benchmark
model_files = [
    "tiny_wav2letter_int8.tflite"
]

class ModelBenchmark:
    def __init__(self, model_path, warmup_runs=5, benchmark_runs=100):
        self.model_path = model_path
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
        
    def create_random_tensor(self, shape, dtype):
        """Create random tensor based on shape and dtype"""
        if dtype == np.float32:
            return np.random.uniform(-1.0, 1.0, shape).astype(dtype)
        elif dtype == np.float16:
            return np.random.uniform(-1.0, 1.0, shape).astype(dtype)
        elif dtype == np.int32:
            return np.random.randint(0, 2, shape).astype(dtype)
        elif dtype == np.int8:
            return np.random.randint(-128, 127, shape).astype(dtype)
        elif dtype == np.uint8:
            return np.random.randint(0, 255, shape).astype(dtype)
        else:
            print(f"    WARNING: Unsupported dtype {dtype}, using float32")
            return np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
    
    def resolve_dynamic_shape(self, shape):
        """Resolve dynamic shapes by replacing -1 with reasonable values"""
        if -1 not in shape:
            return shape
            
        new_shape = []
        for dim in shape:
            if dim == -1:
                if len(shape) == 2:  # Likely sequence length
                    new_shape.append(100)
                elif len(shape) == 3:  # Likely batch, time, features
                    new_shape.append(50)
                else:
                    new_shape.append(10)
            else:
                new_shape.append(dim)
        return tuple(new_shape)
    
    def prepare_inputs(self, input_details):
        """Prepare random inputs for the model"""
        inputs = []
        for detail in input_details:
            shape = self.resolve_dynamic_shape(detail['shape'])
            dtype = detail['dtype']
            tensor = self.create_random_tensor(shape, dtype)
            inputs.append(tensor)
        return inputs
    
    def benchmark_interpreter(self, interpreter, input_details, output_details, inputs, mode_name):
        """Benchmark a specific interpreter configuration"""
        print(f"\n{'='*50}")
        print(f"BENCHMARKING: {mode_name}")
        print(f"{'='*50}")
        
        # Warmup runs
        print(f"Performing {self.warmup_runs} warmup runs...")
        for i in range(self.warmup_runs):
            try:
                # Set inputs
                for detail, tensor in zip(input_details, inputs):
                    interpreter.set_tensor(detail['index'], tensor)
                
                # Run inference
                interpreter.invoke()
                
                if (i + 1) % max(1, self.warmup_runs // 5) == 0:
                    print(f"  Warmup {i+1}/{self.warmup_runs}")
                    
            except Exception as e:
                print(f"Error during warmup {i+1}: {e}")
                return None
        
        # Benchmark runs
        print(f"\nPerforming {self.benchmark_runs} benchmark runs...")
        execution_times = []
        
        for i in range(self.benchmark_runs):
            try:
                # Set inputs
                for detail, tensor in zip(input_details, inputs):
                    interpreter.set_tensor(detail['index'], tensor)
                
                # Time the inference
                start_time = time.perf_counter()
                interpreter.invoke()
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                execution_times.append(execution_time)
                
                if (i + 1) % max(1, self.benchmark_runs // 10) == 0:
                    print(f"  Run {i+1}/{self.benchmark_runs} - {execution_time:.2f}ms")
                    
            except Exception as e:
                print(f"Error during benchmark run {i+1}: {e}")
                continue
        
        if not execution_times:
            print(f"No successful runs for {mode_name}")
            return None
        
        # Calculate statistics
        stats = {
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'mean_time': statistics.mean(execution_times),
            'median_time': statistics.median(execution_times),
            'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'total_runs': len(execution_times),
            'successful_runs': len(execution_times),
            'raw_times': execution_times
        }
        
        print(f"\n{mode_name} RESULTS:")
        print(f"  Successful runs: {stats['successful_runs']}/{self.benchmark_runs}")
        print(f"  Mean time:       {stats['mean_time']:.2f} ms")
        print(f"  Median time:     {stats['median_time']:.2f} ms")
        print(f"  Min time:        {stats['min_time']:.2f} ms")
        print(f"  Max time:        {stats['max_time']:.2f} ms")
        print(f"  Std deviation:   {stats['std_dev']:.2f} ms")
        print(f"  Throughput:      {1000/stats['mean_time']:.2f} inferences/sec")
        
        return stats
    
    def benchmark_model(self):
        """Benchmark the model on both CPU and NPU"""
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            return None
        
        print(f"\n{'='*80}")
        print(f"BENCHMARKING MODEL: {os.path.basename(self.model_path)}")
        print(f"Model path: {self.model_path}")
        print(f"File size: {os.path.getsize(self.model_path) / (1024*1024):.1f} MB")
        print(f"Warmup runs: {self.warmup_runs}")
        print(f"Benchmark runs: {self.benchmark_runs}")
        print(f"{'='*80}")
        
        # First, load CPU interpreter to get model details and prepare inputs
        try:
            print("\nLoading model for input inspection...")
            cpu_interpreter = tflite.Interpreter(model_path=self.model_path)
            cpu_interpreter.allocate_tensors()
            
            input_details = cpu_interpreter.get_input_details()
            output_details = cpu_interpreter.get_output_details()
            
            print(f"Model has {len(input_details)} inputs and {len(output_details)} outputs")
            
            # Display input/output details
            for i, detail in enumerate(input_details):
                print(f"  Input {i}: {detail['name']} - {detail['shape']} - {detail['dtype']}")
            for i, detail in enumerate(output_details):
                print(f"  Output {i}: {detail['name']} - {detail['shape']} - {detail['dtype']}")
            
            # Prepare inputs (same inputs will be used for both CPU and NPU)
            inputs = self.prepare_inputs(input_details)
            print(f"\nPrepared {len(inputs)} input tensors")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return None
        
        results = {
            'model_path': self.model_path,
            'model_name': os.path.basename(self.model_path),
            'file_size_mb': os.path.getsize(self.model_path) / (1024*1024),
            'timestamp': datetime.now().isoformat(),
            'warmup_runs': self.warmup_runs,
            'benchmark_runs': self.benchmark_runs,
            'input_details': input_details,
            'output_details': output_details
        }
        
        # Benchmark CPU execution
        try:
            print(f"\n{'='*50}")
            print("SETTING UP CPU INTERPRETER")
            print(f"{'='*50}")
            
            cpu_interpreter = tflite.Interpreter(model_path=self.model_path)
            cpu_interpreter.allocate_tensors()
            
            cpu_stats = self.benchmark_interpreter(
                cpu_interpreter, input_details, output_details, inputs, "CPU EXECUTION"
            )
            
            if cpu_stats:
                results['cpu'] = cpu_stats
            
        except Exception as e:
            print(f"Error benchmarking CPU: {e}")
            traceback.print_exc()
            results['cpu'] = {'error': str(e)}
        
        # Benchmark NPU execution (with VX delegate)
        try:
            print(f"\n{'='*50}")
            print("SETTING UP NPU INTERPRETER (VX DELEGATE)")
            print(f"{'='*50}")
            
            if not os.path.exists(VX_DELEGATE_PATH):
                print(f"VX Delegate not found at: {VX_DELEGATE_PATH}")
                results['npu'] = {'error': 'VX Delegate not found'}
            else:
                npu_interpreter = tflite.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[
                        tflite.load_delegate(VX_DELEGATE_PATH,{
                         'allowed_builtin_code': '1',  # Enable caching
                          'device_assignment': '0'    # Use primary NPU
                    })]
                )
                npu_interpreter.allocate_tensors()
                
                npu_stats = self.benchmark_interpreter(
                    npu_interpreter, input_details, output_details, inputs, "NPU EXECUTION (VX DELEGATE)"
                )
                
                if npu_stats:
                    results['npu'] = npu_stats
            
        except Exception as e:
            print(f"Error benchmarking NPU: {e}")
            traceback.print_exc()
            results['npu'] = {'error': str(e)}
        
        # Calculate speedup if both succeeded
        if 'cpu' in results and 'npu' in results and 'mean_time' in results['cpu'] and 'mean_time' in results['npu']:
            speedup = results['cpu']['mean_time'] / results['npu']['mean_time']
            results['speedup'] = speedup
            
            print(f"\n{'='*50}")
            print("PERFORMANCE COMPARISON")
            print(f"{'='*50}")
            print(f"CPU mean time:    {results['cpu']['mean_time']:.2f} ms")
            print(f"NPU mean time:    {results['npu']['mean_time']:.2f} ms")
            print(f"NPU Speedup:      {speedup:.2f}x")
            print(f"CPU Throughput:   {1000/results['cpu']['mean_time']:.2f} inferences/sec")
            print(f"NPU Throughput:   {1000/results['npu']['mean_time']:.2f} inferences/sec")
        
        self.results = results
        return results
    
    def save_results(self, output_file=None):
        """Save benchmark results to JSON file"""
        if not self.results:
            print("No results to save")
            return
        
        if output_file is None:
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_{model_name}_{timestamp}.json"
        
        # Remove raw_times to reduce file size, keep only statistics
        results_to_save = self.results.copy()
        for mode in ['cpu', 'npu']:
            if mode in results_to_save and 'raw_times' in results_to_save[mode]:
                del results_to_save[mode]['raw_times']
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

def benchmark_all_models(warmup_runs=5, benchmark_runs=100, save_results=True):
    """Benchmark all models in the model_files list"""
    all_results = []
    
    print("TensorFlow Lite Model Benchmarking Suite")
    print("=" * 80)
    print(f"Warmup runs per model: {warmup_runs}")
    print(f"Benchmark runs per model: {benchmark_runs}")
    print(f"Models to benchmark: {len(model_files)}")
    print("=" * 80)
    
    for i, model_file in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] Starting benchmark for: {model_file}")
        
        benchmark = ModelBenchmark(model_file, warmup_runs, benchmark_runs)
        results = benchmark.benchmark_model()
        
        if results:
            all_results.append(results)
            if save_results:
                benchmark.save_results()
        
        print(f"\n{'='*80}")
    
    # Print summary
    print(f"\nBENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully benchmarked: {len(all_results)}/{len(model_files)} models")
    
    for result in all_results:
        model_name = result['model_name']
        print(f"\n{model_name}:")
        
        if 'cpu' in result and 'mean_time' in result['cpu']:
            print(f"  CPU:  {result['cpu']['mean_time']:.2f} ms")
        else:
            print(f"  CPU:  Failed")
            
        if 'npu' in result and 'mean_time' in result['npu']:
            print(f"  NPU:  {result['npu']['mean_time']:.2f} ms")
        else:
            print(f"  NPU:  Failed")
            
        if 'speedup' in result:
            print(f"  Speedup: {result['speedup']:.2f}x")
    
    return all_results

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark TensorFlow Lite models on CPU vs NPU')
    parser.add_argument('--model', type=str, help='Specific model file to benchmark')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup runs (default: 5)')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs (default: 100)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results to file')
    
    args = parser.parse_args()
    
    if args.model:
        # Benchmark single model
        benchmark = ModelBenchmark(args.model, args.warmup, args.runs)
        results = benchmark.benchmark_model()
        if results and not args.no_save:
            benchmark.save_results(args.output)
    else:
        # Benchmark all models
        benchmark_all_models(args.warmup, args.runs, not args.no_save)

if __name__ == "__main__":
    main()
