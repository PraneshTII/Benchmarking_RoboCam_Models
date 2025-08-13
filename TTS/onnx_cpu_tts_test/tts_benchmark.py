




#!/usr/bin/env python3

import sys;
import site;
import functools;
import wave
functools.reduce(lambda k, p: site.addsitedir(p, k), 
['/nix/store/7wdal3kjsp1hn6ig64m2y6baxg4f929h-tflite-opencv-test-app-cpu-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/1bvpxg3kvzjhrp7n9q114xcjxmyx2ik8-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/d5hmgjyy2wy17k79z1j1gjs0fv1wh5ki-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/n60lsdw7cm43lv10bdrcdcxqv7dpnn0b-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/s58a7qp81ms8an7xdc7h008d8kc13kqp-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/n4wvcq4sllahxr7ppqj4mxsyh1y1jqwp-python3.12-scipy-1.14.1-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import tflite_runtime.interpreter as tflite
from scipy.fft import dct
from scipy.io import wavfile
from scipy.signal import get_window
from scipy.fftpack import dct
from scipy.special import logsumexp
import sys
sys.path.insert(0,'/home/scmd/audio_test_python_setup')
import tensorflow as tf
import librosa
from jiwer import wer
from collections import namedtuple
from collections import defaultdict, namedtuple
from pygtrie import Trie
import heapq
import difflib
import re
import time
import psutil
import threading
import statistics
import wave
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Callable, Any

@dataclass
class BenchmarkResult:
    """Store benchmark results for a single test"""
    model_name: str
    text_length: int
    audio_duration: float
    
    # Latency & Speed Metrics
    inference_latency: float  # Time to generate audio
    real_time_factor: float  # RTF = processing_time / audio_duration
    throughput_chars_per_sec: float  # Characters processed per second
    throughput_words_per_sec: float  # Words processed per second
    
    # Computational Efficiency Metrics
    peak_cpu_percent: float
    avg_cpu_percent: float
    peak_memory_mb: float
    avg_memory_mb: float
    memory_increase_mb: float  # Memory increase from baseline

class ResourceMonitor:
    """Monitor CPU and memory usage during TTS generation"""
    
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.baseline_memory = None
        
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        
        # Get baseline memory before starting
        process = psutil.Process()
        self.baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        
        if not self.cpu_samples or not self.memory_samples:
            return {
                'peak_cpu': 0.0, 'avg_cpu': 0.0,
                'peak_memory': 0.0, 'avg_memory': 0.0,
                'memory_increase': 0.0
            }
            
        return {
            'peak_cpu': max(self.cpu_samples),
            'avg_cpu': statistics.mean(self.cpu_samples),
            'peak_memory': max(self.memory_samples),
            'avg_memory': statistics.mean(self.memory_samples),
            'memory_increase': max(self.memory_samples) - self.baseline_memory
        }
    
    def _monitor_loop(self):
        """Internal monitoring loop"""
        process = psutil.Process()
        while self.monitoring:
            try:
                self.cpu_samples.append(process.cpu_percent())
                memory_mb = process.memory_info().rss / (1024 * 1024)  # Convert to MB
                self.memory_samples.append(memory_mb)
                time.sleep(self.interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

class TTSBenchmark:
    """Main benchmarking framework for TTS models"""
    
    def __init__(self):
        self.results = []
        
    def get_audio_duration(self, audio_data, sample_rate: int) -> float:
        """Calculate audio duration in seconds"""
        if isinstance(audio_data, np.ndarray):
            return len(audio_data) / sample_rate
        return 0.0
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def benchmark_model(
        self, 
        model_name: str,
        tts_function: Callable[[str], Any],
        test_texts: List[str],
        sample_rate: int = 22050,
        num_runs: int = 3,
        warmup_runs: int = 1
    ) -> List[BenchmarkResult]:
        """
        Benchmark a TTS model with given test texts
        
        Args:
            model_name: Name of the TTS model
            tts_function: Function that takes text and returns audio data
            test_texts: List of texts to test
            sample_rate: Audio sample rate
            num_runs: Number of runs per text for averaging
            warmup_runs: Number of warmup runs (not counted in results)
        """
        
        print(f"\n🔥 Benchmarking {model_name}")
        print("=" * 50)
        
        model_results = []
        
        for text_idx, text in enumerate(test_texts):
            print(f"\nText {text_idx + 1}/{len(test_texts)}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Warmup runs
            print(f"  Warming up... ({warmup_runs} runs)")
            for _ in range(warmup_runs):
                try:
                    _ = tts_function(text)
                except Exception as e:
                    print(f"  ⚠️  Warmup failed: {e}")
                    continue
            
            # Actual benchmark runs
            run_results = []
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...", end=" ")
                
                try:
                    # Start monitoring
                    monitor = ResourceMonitor()
                    monitor.start_monitoring()
                    
                    # Measure inference time
                    start_time = time.perf_counter()
                    audio_data = tts_function(text)
                    end_time = time.perf_counter()
                    
                    # Stop monitoring
                    resource_metrics = monitor.stop_monitoring()
                    
                    # Calculate metrics
                    inference_latency = end_time - start_time
                    audio_duration = self.get_audio_duration(audio_data, sample_rate)
                    real_time_factor = inference_latency / audio_duration if audio_duration > 0 else float('inf')
                    
                    char_count = len(text)
                    word_count = self.count_words(text)
                    throughput_chars = char_count / inference_latency
                    throughput_words = word_count / inference_latency
                    
                    # Create result
                    result = BenchmarkResult(
                        model_name=model_name,
                        text_length=char_count,
                        audio_duration=audio_duration,
                        inference_latency=inference_latency,
                        real_time_factor=real_time_factor,
                        throughput_chars_per_sec=throughput_chars,
                        throughput_words_per_sec=throughput_words,
                        peak_cpu_percent=resource_metrics['peak_cpu'],
                        avg_cpu_percent=resource_metrics['avg_cpu'],
                        peak_memory_mb=resource_metrics['peak_memory'],
                        avg_memory_mb=resource_metrics['avg_memory'],
                        memory_increase_mb=resource_metrics['memory_increase']
                    )
                    
                    run_results.append(result)
                    print(f"✅ RTF: {real_time_factor:.3f}, Latency: {inference_latency:.3f}s")
                    
                except Exception as e:
                    print(f"❌ Failed: {e}")
                    continue
            
            if run_results:
                # Average results across runs
                avg_result = self._average_results(run_results)
                model_results.append(avg_result)
                
        self.results.extend(model_results)
        return model_results
    
    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Average multiple benchmark results"""
        if not results:
            raise ValueError("No results to average")
            
        # Use first result as template
        template = results[0]
        
        return BenchmarkResult(
            model_name=template.model_name,
            text_length=template.text_length,
            audio_duration=statistics.mean([r.audio_duration for r in results]),
            inference_latency=statistics.mean([r.inference_latency for r in results]),
            real_time_factor=statistics.mean([r.real_time_factor for r in results]),
            throughput_chars_per_sec=statistics.mean([r.throughput_chars_per_sec for r in results]),
            throughput_words_per_sec=statistics.mean([r.throughput_words_per_sec for r in results]),
            peak_cpu_percent=statistics.mean([r.peak_cpu_percent for r in results]),
            avg_cpu_percent=statistics.mean([r.avg_cpu_percent for r in results]),
            peak_memory_mb=statistics.mean([r.peak_memory_mb for r in results]),
            avg_memory_mb=statistics.mean([r.avg_memory_mb for r in results]),
            memory_increase_mb=statistics.mean([r.memory_increase_mb for r in results])
        )
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No results to display")
            return
            
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)
        
        # Group by model
        models = {}
        for result in self.results:
            if result.model_name not in models:
                models[result.model_name] = []
            models[result.model_name].append(result)
        
        for model_name, model_results in models.items():
            print(f"\n🤖 {model_name}")
            print("-" * 60)
            
            # Calculate averages across all texts
            avg_rtf = statistics.mean([r.real_time_factor for r in model_results])
            avg_latency = statistics.mean([r.inference_latency for r in model_results])
            avg_throughput_chars = statistics.mean([r.throughput_chars_per_sec for r in model_results])
            avg_throughput_words = statistics.mean([r.throughput_words_per_sec for r in model_results])
            avg_peak_cpu = statistics.mean([r.peak_cpu_percent for r in model_results])
            avg_avg_cpu = statistics.mean([r.avg_cpu_percent for r in model_results])
            avg_peak_memory = statistics.mean([r.peak_memory_mb for r in model_results])
            avg_memory_increase = statistics.mean([r.memory_increase_mb for r in model_results])
            
            print(f"⚡ SPEED METRICS:")
            print(f"   Real-Time Factor (RTF):     {avg_rtf:.3f} ({'✅ Real-time capable' if avg_rtf < 1.0 else 'Slower than real-time'})")
            print(f"   Average Inference Latency:  {avg_latency:.3f} seconds")
            print(f"   Throughput (chars/sec):     {avg_throughput_chars:.1f}")
            print(f"   Throughput (words/sec):     {avg_throughput_words:.1f}")
            
            print(f"\n💻 EFFICIENCY METRICS:")
            print(f"   Peak CPU Usage:             {avg_peak_cpu:.1f}%")
            print(f"   Average CPU Usage:          {avg_avg_cpu:.1f}%")
            print(f"   Peak Memory Usage:          {avg_peak_memory:.1f} MB")
            print(f"   Memory Increase:            {avg_memory_increase:.1f} MB")
        
        # Comparison if multiple models
        if len(models) > 1:
            print(f"\n🏆 COMPARISON")
            print("-" * 60)
            
            model_stats = {}
            for model_name, model_results in models.items():
                model_stats[model_name] = {
                    'rtf': statistics.mean([r.real_time_factor for r in model_results]),
                    'latency': statistics.mean([r.inference_latency for r in model_results]),
                    'cpu': statistics.mean([r.avg_cpu_percent for r in model_results]),
                    'memory': statistics.mean([r.peak_memory_mb for r in model_results])
                }
            
            # Find winners
            fastest_rtf = min(model_stats.items(), key=lambda x: x[1]['rtf'])
            lowest_latency = min(model_stats.items(), key=lambda x: x[1]['latency'])
            lowest_cpu = min(model_stats.items(), key=lambda x: x[1]['cpu'])
            lowest_memory = min(model_stats.items(), key=lambda x: x[1]['memory'])
            
            print(f"🥇 Fastest (RTF):           {fastest_rtf[0]} ({fastest_rtf[1]['rtf']:.3f})")
            print(f"🥇 Lowest Latency:          {lowest_latency[0]} ({lowest_latency[1]['latency']:.3f}s)")
            print(f"🥇 Most CPU Efficient:      {lowest_cpu[0]} ({lowest_cpu[1]['cpu']:.1f}%)")
            print(f"🥇 Most Memory Efficient:   {lowest_memory[0]} ({lowest_memory[1]['memory']:.1f} MB)")

# Example usage and wrapper functions for your TTS models
def create_kitten_tts_function():
    """Create a function wrapper for KittenTTS"""
    from kittentts import KittenTTS
    
    model = KittenTTS("KittenML/kitten-tts-nano-0.1")
    
    def kitten_generate(text: str):
        return model.generate(text, voice='expr-voice-2-m')
    
    return kitten_generate

def create_piper_tts_function(model_path: str):
    """Create a function wrapper for Piper TTS"""
    import wave
    import tempfile
    import soundfile as sf
    from piper import PiperVoice
    
    voice = PiperVoice.load(model_path)
    
    def piper_generate(text: str):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            with wave.open(tmp_file.name, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            
            # Read back the audio data
            audio_data, sample_rate = sf.read(tmp_file.name)
            return audio_data
    
    return piper_generate

# Example test texts of varying lengths
TEST_TEXTS = [
    # Short (5-15 words)
    "Hello world, this is a test.",
    "Welcome to the world of speech synthesis!",
    
    # Medium (20-40 words) 
    "This high quality text-to-speech model works without a GPU and can generate natural sounding speech from any text input you provide to it.",
    "Speech synthesis technology has advanced significantly in recent years, enabling more natural and expressive artificial voices for various applications.",
    
    # Long (50+ words)
    "The development of text-to-speech systems has undergone remarkable progress over the past decade, with neural networks and deep learning techniques revolutionizing the quality and naturalness of synthesized speech. Modern TTS models can now produce speech that is nearly indistinguishable from human recordings, opening up new possibilities for accessibility, entertainment, and communication applications across diverse industries and use cases.",
    
    # Complex content
    "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet and is commonly used for testing purposes in typography and speech synthesis systems.",
]

if __name__ == "__main__":
    # Example usage
    print("🚀 TTS Performance Benchmarking Framework")
    print("This framework measures latency, speed, and computational efficiency")
    print("\nTo use this framework:")
    print("1. Create wrapper functions for your TTS models")
    print("2. Initialize the benchmark")
    print("3. Run benchmark_model() for each model")
    print("4. Call print_summary() to see results")
    
    print(f"\nExample test texts included: {len(TEST_TEXTS)} texts")
    print("Ready to benchmark your KittenTTS and Piper models!")
