#!/usr/bin/env python3
"""
Realistic Wav2Letter Benchmarking Script
Simulates actual sliding window inference pattern used in production
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
import numpy as np
import time
import statistics
import json
import tflite_runtime.interpreter as tflite
import traceback
from datetime import datetime

# VX Delegate path
VX_DELEGATE_PATH = '/nix/store/96bsy96b042wsqgzazpdhcdkqhai9k7n-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so'

class RealisticWav2LetterBenchmark:
    def __init__(self, model_path, warmup_runs=5):
        self.model_path = model_path
        self.warmup_runs = warmup_runs
        self.window_size = 296
        self.context_size = 98  # 24 + 2 * (7 * 3 + 16)
        self.inner_size = self.window_size - 2 * self.context_size  # 100
        self.results = {}
        
    def calculate_windows_needed(self, audio_length_seconds):
        """Calculate number of inference calls needed for given audio length"""
        # Assume 16kHz audio with hop_length=160 -> ~100 frames per second
        frames_per_second = 100
        total_frames = int(audio_length_seconds * frames_per_second)
        
        if total_frames <= self.window_size:
            return 1
        
        # First window covers window_size - context_size frames
        remaining_frames = total_frames - (self.window_size - self.context_size)
        
        # Each subsequent window covers inner_size frames
        additional_windows = max(0, (remaining_frames + self.inner_size - 1) // self.inner_size)
        
        return 1 + additional_windows
    
    def create_random_input(self):
        """Create random input tensor matching wav2letter format"""
        # Input shape: [1, 296, 39] - batch, time, features (13 MFCC + 13 delta + 13 delta2)
        shape = (1, self.window_size, 39)
        return np.random.randint(-128, 127, shape).astype(np.int8)
    
    def benchmark_sliding_window_inference(self, interpreter, mode_name, audio_durations=[1, 5, 10, 30, 60]):
        """Benchmark sliding window inference for different audio lengths"""
        print(f"\n{'='*70}")
        print(f"REALISTIC SLIDING WINDOW BENCHMARK: {mode_name}")
        print(f"{'='*70}")
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input tensor (will be reused for all windows)
        input_tensor = self.create_random_input()
        
        # Warmup
        print(f"Performing {self.warmup_runs} warmup inferences...")
        for i in range(self.warmup_runs):
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            if (i + 1) % max(1, self.warmup_runs // 2) == 0:
                print(f"  Warmup {i+1}/{self.warmup_runs}")
        
        results = {}
        
        for duration in audio_durations:
            windows_needed = self.calculate_windows_needed(duration)
            print(f"\n--- Testing {duration}s audio ({windows_needed} inference calls) ---")
            
            # Time multiple complete transcriptions
            transcription_times = []
            single_inference_times = []
            
            for run in range(5):  # 5 complete transcriptions per duration
                transcription_start = time.perf_counter()
                
                # Simulate sliding window inference
                for window in range(windows_needed):
                    # Set input (in real usage, this would be different audio data each time)
                    interpreter.set_tensor(input_details[0]['index'], input_tensor)
                    
                    # Time individual inference
                    inference_start = time.perf_counter()
                    interpreter.invoke()
                    inference_end = time.perf_counter()
                    
                    single_inference_times.append((inference_end - inference_start) * 1000)
                    
                    # Get output (simulate real usage)
                    _ = interpreter.get_tensor(output_details[0]['index'])
                
                transcription_end = time.perf_counter()
                total_time = (transcription_end - transcription_start) * 1000
                transcription_times.append(total_time)
                
                print(f"  Run {run+1}: {total_time:.1f}ms total, {total_time/windows_needed:.1f}ms avg per window")
            
            # Calculate statistics
            stats = {
                'audio_duration_seconds': duration,
                'windows_needed': windows_needed,
                'total_transcription_times_ms': transcription_times,
                'mean_transcription_time_ms': statistics.mean(transcription_times),
                'std_transcription_time_ms': statistics.stdev(transcription_times) if len(transcription_times) > 1 else 0,
                'mean_single_inference_ms': statistics.mean(single_inference_times),
                'realtime_factor': statistics.mean(transcription_times) / (duration * 1000),
                'throughput_hours_per_hour': (duration * 3600) / statistics.mean(transcription_times)
            }
            
            results[f'{duration}s'] = stats
            
            print(f"  Mean total time: {stats['mean_transcription_time_ms']:.1f}ms")
            print(f"  Mean per window: {stats['mean_single_inference_ms']:.1f}ms")
            print(f"  Real-time factor: {stats['realtime_factor']:.2f}x")
            if stats['realtime_factor'] < 1.0:
                print(f"  ✓ Faster than real-time!")
            else:
                print(f"  ⚠ Slower than real-time")
            print(f"  Throughput: {stats['throughput_hours_per_hour']:.1f} hours of audio per hour")
        
        return results
    
    def benchmark_batched_inference(self, interpreter, mode_name, batch_sizes=[1, 2, 4, 8]):
        """Test batched inference capability (if model supports it)"""
        print(f"\n{'='*60}")
        print(f"BATCHED INFERENCE BENCHMARK: {mode_name}")
        print(f"{'='*60}")
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        original_shape = input_details[0]['shape']
        print(f"Original input shape: {original_shape}")
        
        # Check if batch dimension is dynamic
        if original_shape[0] == -1 or original_shape[0] == 1:
            print("Model supports dynamic/single batch size")
            
            results = {}
            
            for batch_size in batch_sizes:
                print(f"\n--- Testing batch size {batch_size} ---")
                
                try:
                    # Create batched input
                    batched_shape = [batch_size] + list(original_shape[1:])
                    batched_input = np.random.randint(-128, 127, batched_shape).astype(np.int8)
                    
                    # Resize interpreter input tensor if needed
                    if original_shape[0] == -1:
                        interpreter.resize_tensor_input(input_details[0]['index'], batched_shape)
                        interpreter.allocate_tensors()
                    elif batch_size != 1:
                        print(f"  Skipping batch size {batch_size} - model only supports batch size 1")
                        continue
                    
                    # Warmup
                    for _ in range(3):
                        interpreter.set_tensor(input_details[0]['index'], batched_input)
                        interpreter.invoke()
                    
                    # Benchmark
                    batch_times = []
                    for run in range(10):
                        interpreter.set_tensor(input_details[0]['index'], batched_input)
                        
                        start_time = time.perf_counter()
                        interpreter.invoke()
                        end_time = time.perf_counter()
                        
                        batch_time = (end_time - start_time) * 1000
                        batch_times.append(batch_time)
                    
                    mean_batch_time = statistics.mean(batch_times)
                    per_sample_time = mean_batch_time / batch_size
                    
                    results[f'batch_{batch_size}'] = {
                        'batch_size': batch_size,
                        'mean_batch_time_ms': mean_batch_time,
                        'per_sample_time_ms': per_sample_time,
                        'samples_per_second': 1000 / per_sample_time,
                        'efficiency_vs_single': (results.get('batch_1', {}).get('per_sample_time_ms', per_sample_time)) / per_sample_time if 'batch_1' in results else 1.0
                    }
                    
                    print(f"  Batch time: {mean_batch_time:.1f}ms")
                    print(f"  Per sample: {per_sample_time:.1f}ms")
                    print(f"  Throughput: {1000/per_sample_time:.1f} samples/sec")
                    if 'batch_1' in results:
                        efficiency = results[f'batch_{batch_size}']['efficiency_vs_single']
                        print(f"  Efficiency vs single: {efficiency:.2f}x")
                
                except Exception as e:
                    print(f"  Error with batch size {batch_size}: {e}")
                    continue
            
            return results
        else:
            print(f"Model has fixed batch size: {original_shape[0]}")
            return {'note': 'Fixed batch size model'}
    
    def benchmark_model(self):
        """Run comprehensive benchmarking including realistic usage patterns"""
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            return None
        
        print(f"\n{'='*80}")
        print(f"REALISTIC WAV2LETTER BENCHMARKING: {os.path.basename(self.model_path)}")
        print(f"Model path: {self.model_path}")
        print(f"File size: {os.path.getsize(self.model_path) / (1024*1024):.1f} MB")
        print(f"{'='*80}")
        
        results = {
            'model_path': self.model_path,
            'model_name': os.path.basename(self.model_path),
            'file_size_mb': os.path.getsize(self.model_path) / (1024*1024),
            'timestamp': datetime.now().isoformat(),
            'window_size': self.window_size,
            'context_size': self.context_size,
            'inner_size': self.inner_size
        }
        
        # Calculate expected windows for different audio lengths
        durations = [1, 5, 10, 30, 60]
        windows_info = {}
        for duration in durations:
            windows_needed = self.calculate_windows_needed(duration)
            windows_info[f'{duration}s'] = {
                'duration': duration,
                'windows_needed': windows_needed,
                'windows_per_second': windows_needed / duration
            }
        
        results['windows_analysis'] = windows_info
        
        print(f"\nSLIDING WINDOW ANALYSIS:")
        for duration, info in windows_info.items():
            print(f"  {duration}: {info['windows_needed']} windows ({info['windows_per_second']:.1f} windows/sec)")
        
        # Benchmark CPU
        try:
            print(f"\n{'='*60}")
            print("SETTING UP CPU INTERPRETER")
            print(f"{'='*60}")
            
            cpu_interpreter = tflite.Interpreter(model_path=self.model_path)
            cpu_interpreter.allocate_tensors()
            
            cpu_sliding_results = self.benchmark_sliding_window_inference(cpu_interpreter, "CPU")
            cpu_batch_results = self.benchmark_batched_inference(cpu_interpreter, "CPU")
            
            results['cpu'] = {
                'sliding_window': cpu_sliding_results,
                'batched': cpu_batch_results
            }
            
        except Exception as e:
            print(f"CPU benchmark error: {e}")
            traceback.print_exc()
            results['cpu'] = {'error': str(e)}
        
        # Benchmark NPU
        try:
            print(f"\n{'='*60}")
            print("SETTING UP NPU INTERPRETER (VX DELEGATE)")
            print(f"{'='*60}")
            
            if os.path.exists(VX_DELEGATE_PATH):
                npu_interpreter = tflite.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[tflite.load_delegate(VX_DELEGATE_PATH)]
                )
                npu_interpreter.allocate_tensors()
                
                npu_sliding_results = self.benchmark_sliding_window_inference(npu_interpreter, "NPU")
                npu_batch_results = self.benchmark_batched_inference(npu_interpreter, "NPU")
                
                results['npu'] = {
                    'sliding_window': npu_sliding_results,
                    'batched': npu_batch_results
                }
            else:
                results['npu'] = {'error': 'VX Delegate not found'}
                
        except Exception as e:
            print(f"NPU benchmark error: {e}")
            traceback.print_exc()
            results['npu'] = {'error': str(e)}
        
        # Comprehensive comparison
        if ('cpu' in results and 'npu' in results and 
            'sliding_window' in results['cpu'] and 'sliding_window' in results['npu']):
            
            print(f"\n{'='*80}")
            print("REALISTIC PERFORMANCE COMPARISON")
            print(f"{'='*80}")
            
            comparison = {}
            
            for duration in durations:
                duration_key = f'{duration}s'
                if (duration_key in results['cpu']['sliding_window'] and 
                    duration_key in results['npu']['sliding_window']):
                    
                    cpu_time = results['cpu']['sliding_window'][duration_key]['mean_transcription_time_ms']
                    npu_time = results['npu']['sliding_window'][duration_key]['mean_transcription_time_ms']
                    speedup = cpu_time / npu_time
                    
                    cpu_rtf = results['cpu']['sliding_window'][duration_key]['realtime_factor']
                    npu_rtf = results['npu']['sliding_window'][duration_key]['realtime_factor']
                    
                    comparison[duration_key] = {
                        'cpu_time_ms': cpu_time,
                        'npu_time_ms': npu_time,
                        'npu_speedup': speedup,
                        'cpu_realtime_factor': cpu_rtf,
                        'npu_realtime_factor': npu_rtf
                    }
                    
                    print(f"\n{duration}s audio transcription:")
                    print(f"  CPU: {cpu_time:.0f}ms (RTF: {cpu_rtf:.2f})")
                    print(f"  NPU: {npu_time:.0f}ms (RTF: {npu_rtf:.2f})")
                    print(f"  NPU speedup: {speedup:.2f}x")
                    
                    if npu_rtf < 1.0:
                        print(f"  ✓ NPU achieves real-time performance")
                    if cpu_rtf < 1.0:
                        print(f"  ✓ CPU achieves real-time performance")
            
            results['comparison'] = comparison
            
            # Overall recommendation
            avg_speedup = statistics.mean([comp['npu_speedup'] for comp in comparison.values()])
            results['overall_speedup'] = avg_speedup
            
            print(f"\nOVERALL ANALYSIS:")
            print(f"  Average NPU speedup: {avg_speedup:.2f}x")
            
            if avg_speedup >= 1.5:
                recommendation = "✓ Strong recommendation for NPU deployment"
            elif avg_speedup >= 1.2:
                recommendation = "✓ Moderate recommendation for NPU deployment"
            elif avg_speedup >= 0.8:
                recommendation = "~ Consider NPU for power efficiency, performance similar"
            else:
                recommendation = "⚠ Recommend CPU deployment"
            
            results['recommendation'] = recommendation
            print(f"  {recommendation}")
        
        self.results = results
        return results
    
    def save_results(self, output_file=None):
        """Save comprehensive results to JSON file"""
        if not self.results:
            print("No results to save")
            return
        
        if output_file is None:
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"realistic_benchmark_{model_name}_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Realistic Wav2Letter model benchmarking')
    parser.add_argument('--model', type=str, required=True, help='Model file to benchmark')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup runs (default: 5)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    benchmark = RealisticWav2LetterBenchmark(args.model, args.warmup)
    results = benchmark.benchmark_model()
    
    if results and args.output:
        benchmark.save_results(args.output)
    elif results:
        benchmark.save_results()

if __name__ == "__main__":
    main()
