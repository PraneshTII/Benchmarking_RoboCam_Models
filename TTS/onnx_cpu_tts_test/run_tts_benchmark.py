#!/usr/bin/env python3
"""
TTS Benchmarking Script
Run this to benchmark KittenTTS vs Piper TTS models

Make sure you have the following installed:
pip install kittentts soundfile psutil
pip install piper-tts  # or however you installed Piper
"""



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

import sys
import os
from pathlib import Path


# Import our benchmarking framework
# (Assuming the benchmark code is in a file called tts_benchmark.py)
from tts_benchmark import TTSBenchmark, TEST_TEXTS, create_kitten_tts_function, create_piper_tts_function

def main():
    """Main benchmarking execution"""
    
    print("🚀 Starting TTS Performance Benchmark")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = TTSBenchmark()
    
    # Configuration
    PIPER_MODEL_PATH = "en_US-lessac-medium.onnx"  
    NUM_RUNS = 3  # Number of runs per text for averaging
    WARMUP_RUNS = 1  # Warmup runs to ensure fair timing
    
    # Test texts 
    test_texts = [
        "Hello world, this is a quick test.",
        "Welcome to the world of speech synthesis technology!",
        "This high quality text-to-speech model works efficiently and can generate natural sounding speech from text input.",
        "The development of modern TTS systems has revolutionized accessibility and communication, enabling natural voice synthesis for countless applications."
    ]
    
    print(f"📝 Testing with {len(test_texts)} texts")
    print(f"🔄 {NUM_RUNS} runs per text (+ {WARMUP_RUNS} warmup)")
    print()
    
    # Benchmark KittenTTS
    try:
        print("🐱 Setting up KittenTTS...")
        kitten_function = create_kitten_tts_function()
        
        kitten_results = benchmark.benchmark_model(
            model_name="KittenTTS",
            tts_function=kitten_function,
            test_texts=test_texts,
            sample_rate=24000,  # KittenTTS sample rate
            num_runs=NUM_RUNS,
            warmup_runs=WARMUP_RUNS
        )
        
    except Exception as e:
        print(f"❌ KittenTTS benchmark failed: {e}")
        print("   Make sure KittenTTS is properly installed")
    
    # Benchmark Piper TTS
    try:
        print("🎵 Setting up Piper TTS...")
        
        # Check if model file exists
        if not os.path.exists(PIPER_MODEL_PATH):
            print(f"❌ Piper model not found at: {PIPER_MODEL_PATH}")
            print("   Please update PIPER_MODEL_PATH in the script")
        else:
            piper_function = create_piper_tts_function(PIPER_MODEL_PATH)
            
            piper_results = benchmark.benchmark_model(
                model_name="Piper TTS",
                tts_function=piper_function,
                test_texts=test_texts,
                sample_rate=22050,  # Piper sample rate
                num_runs=NUM_RUNS,
                warmup_runs=WARMUP_RUNS
            )
            
    except Exception as e:
        print(f"❌ Piper TTS benchmark failed: {e}")
        print("   Make sure Piper TTS is properly installed and model path is correct")
    
    # Print comprehensive results
    benchmark.print_summary()
    
    # Save detailed results to file (optional)
    save_detailed_results(benchmark.results)

def save_detailed_results(results, filename="tts_benchmark_results.csv"):
    """Save detailed results to CSV file for further analysis"""
    
    if not results:
        print("No results to save")
        return
        
    try:
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'model_name', 'text_length', 'audio_duration',
                'inference_latency', 'real_time_factor', 
                'throughput_chars_per_sec', 'throughput_words_per_sec',
                'peak_cpu_percent', 'avg_cpu_percent',
                'peak_memory_mb', 'avg_memory_mb', 'memory_increase_mb'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'model_name': result.model_name,
                    'text_length': result.text_length,
                    'audio_duration': result.audio_duration,
                    'inference_latency': result.inference_latency,
                    'real_time_factor': result.real_time_factor,
                    'throughput_chars_per_sec': result.throughput_chars_per_sec,
                    'throughput_words_per_sec': result.throughput_words_per_sec,
                    'peak_cpu_percent': result.peak_cpu_percent,
                    'avg_cpu_percent': result.avg_cpu_percent,
                    'peak_memory_mb': result.peak_memory_mb,
                    'avg_memory_mb': result.avg_memory_mb,
                    'memory_increase_mb': result.memory_increase_mb
                })
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
    except Exception as e:
        print(f"⚠️  Could not save results to CSV: {e}")

def run_custom_test():
    """Run a custom test with your own texts"""
    
    print("\n🧪 Custom Test Mode")
    print("Enter texts to test (press Enter twice to finish):")
    
    custom_texts = []
    while True:
        text = input("> ")
        if not text:
            if custom_texts:
                break
            else:
                print("Please enter at least one text")
                continue
        custom_texts.append(text)
    
    print(f"\n📝 Testing {len(custom_texts)} custom texts...")
    
    benchmark = TTSBenchmark()
    
    # Test both models with custom texts
    try:
        kitten_function = create_kitten_tts_function()
        benchmark.benchmark_model("KittenTTS", kitten_function, custom_texts)
    except Exception as e:
        print(f"KittenTTS failed: {e}")
    
    try:
        piper_function = create_piper_tts_function("/home/scmd/en_US-lessac-medium.onnx")
        benchmark.benchmark_model("Piper TTS", piper_function, custom_texts)
    except Exception as e:
        print(f"Piper TTS failed: {e}")
    
    benchmark.print_summary()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        run_custom_test()
    else:
        main()

    print("\n✨ Benchmark complete!")
    print("\nRun with --custom flag to test your own texts")
