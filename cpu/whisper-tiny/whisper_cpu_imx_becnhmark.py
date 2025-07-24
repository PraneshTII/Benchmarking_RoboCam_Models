#!/usr/bin/env python3
"""
Whisper TFLite Professional Benchmark Suite
Comprehensive performance analysis for production deployment
"""
import sys
import site
import functools


# Add site packages (keep your existing path setup)
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
import threading
from pathlib import Path
import struct
import wave
import json
import statistics
from datetime import datetime
import tflite_runtime.interpreter as tflite



class WhisperBenchmark:
    def __init__(self, model_path, vocab_mel_path, audio_file):
        self.model_path = model_path
        self.vocab_mel_path = vocab_mel_path
        self.audio_file = audio_file
        self.results = {
            'system_info': {},
            'model_info': {},
            'benchmark_results': [],
            'summary': {}
        }
        
        # Performance tracking
        self.monitoring = False
        
        # Initialize components
        self.setup_model()
        self.setup_preprocessing()
        self.collect_system_info()
        
    def collect_system_info(self):
        """Collect basic system information for benchmark context"""
        self.results['system_info'] = {
            'python_version': sys.version,
            'platform': os.uname().sysname + ' ' + os.uname().release,
            'architecture': os.uname().machine,
            'timestamp': datetime.now().isoformat()
        }
        
    def setup_model(self):
        """Initialize TFLite model"""
        print("Loading TFLite model...")
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.results['model_info'] = {
            'model_path': self.model_path,
            'input_shape': self.input_details[0]['shape'].tolist(),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_dtype': str(self.output_details[0]['dtype'])
        }
        
    def setup_preprocessing(self):
        """Load vocabulary and mel filters"""
        print("Loading preprocessing components...")
        self.mel_filters, self.token_vocab = self.load_vocab_mel(self.vocab_mel_path)
        self.id_to_token = {v: k for k, v in self.token_vocab.items()}
        print(f"Loaded {len(self.token_vocab)} tokens, mel filters: {self.mel_filters.shape}")
        
    def load_vocab_mel(self, filepath):
        """Load vocabulary and mel filters from binary file"""
        with open(filepath, 'rb') as f:
            magic = struct.unpack('i', f.read(4))[0]
            assert magic == 0x74666C74, "Invalid magic number"
            
            mel_rows = struct.unpack('i', f.read(4))[0]
            mel_cols = struct.unpack('i', f.read(4))[0]
            mel_filters = np.zeros((mel_rows, mel_cols), dtype=np.float32)
            
            for i in range(mel_rows):
                for j in range(mel_cols):
                    mel_filters[i][j] = struct.unpack('f', f.read(4))[0]
            
            vocab_size = struct.unpack('i', f.read(4))[0]
            tokens = {}
            
            for _ in range(vocab_size):
                token_len = struct.unpack('i', f.read(4))[0]
                token_bytes = f.read(token_len)
                tokens[token_bytes] = len(tokens)
            
            return mel_filters, tokens
    
    def load_wav(self, wav_path):
        """Load WAV file"""
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            if sample_width == 1:
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
            elif sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            elif sample_width == 4:
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            
            return audio, sample_rate
    
    def stft(self, audio, n_fft=400, hop_length=160):
        """Compute Short-Time Fourier Transform"""
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1)))
        pad_length = n_fft // 2
        audio_padded = np.pad(audio, pad_length, mode='reflect')
        n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
        stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio_padded[start:start + n_fft] * window
            fft_frame = np.fft.fft(frame)
            stft_matrix[:, i] = fft_frame[:n_fft // 2 + 1]
        
        return stft_matrix
    
    def preprocess_audio(self, audio_path):
        """Complete audio preprocessing pipeline"""
        # Load audio
        audio, sr = self.load_wav(audio_path)
        
        if sr != 16000:
            print(f"Warning: Expected 16kHz, got {sr}Hz")
        
        # Pad/truncate to 30 seconds
        target_length = 16000 * 30
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Compute mel spectrogram
        stft_result = self.stft(audio, n_fft=400, hop_length=160)
        magnitude = np.abs(stft_result)
        mel_spec = np.dot(self.mel_filters, magnitude)
        log_mel = np.log10(np.maximum(mel_spec, 1e-10))
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
        
        # Ensure exactly 3000 frames
        if log_mel.shape[1] > 3000:
            log_mel = log_mel[:, :3000]
        else:
            log_mel = np.pad(log_mel, ((0, 0), (0, 3000 - log_mel.shape[1])))
        
        return log_mel.reshape(1, 80, 3000).astype(np.float32)
    
    def decode_tokens(self, token_ids):
        """Decode token IDs to text"""
        text_parts = []
        for token_id in token_ids[0]:
            if token_id == 0:
                break
            if token_id in self.id_to_token:
                token_bytes = self.id_to_token[token_id]
                try:
                    text_parts.append(token_bytes.decode('utf-8', errors='ignore'))
                except:
                    continue
        return ''.join(text_parts)
    
    def run_single_inference(self, mel_input):
        """Run single inference with detailed timing"""
        timings = {}
        
        # Model loading time (if cold start)
        start_time = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], mel_input)
        timings['tensor_setup'] = time.perf_counter() - start_time
        
        # Inference time
        start_time = time.perf_counter()
        self.interpreter.invoke()
        timings['inference'] = time.perf_counter() - start_time
        
        # Output retrieval time
        start_time = time.perf_counter()
        output_tokens = self.interpreter.get_tensor(self.output_details[0]['index'])
        timings['output_retrieval'] = time.perf_counter() - start_time
        
        # Post-processing time
        start_time = time.perf_counter()
        if output_tokens.dtype == np.float32:
            predicted_ids = np.argmax(output_tokens, axis=-1)
        else:
            predicted_ids = output_tokens
        transcription = self.decode_tokens(predicted_ids)
        timings['postprocessing'] = time.perf_counter() - start_time
        
        timings['total'] = sum(timings.values())
        
        return transcription, timings, output_tokens.shape
    
    def benchmark_inference(self, iterations=10, warmup_runs=3):
        """Comprehensive inference benchmark"""
        print(f"\n{'='*60}")
        print("WHISPER TFLITE PROFESSIONAL BENCHMARK")
        print(f"{'='*60}")
        
        # Preprocess audio once
        print("Preprocessing audio...")
        start_time = time.perf_counter()
        mel_input = self.preprocess_audio(self.audio_file)
        preprocessing_time = time.perf_counter() - start_time
        print(f"Preprocessing completed: {preprocessing_time:.4f}s")
        
        # Warmup runs
        print(f"\nRunning {warmup_runs} warmup iterations...")
        for i in range(warmup_runs):
            _, _, _ = self.run_single_inference(mel_input)
            print(f"Warmup {i+1}/{warmup_runs} completed")
        
        # Benchmark runs
        print(f"\nRunning {iterations} benchmark iterations...")
        all_timings = []
        transcriptions = []
        
        for i in range(iterations):
            # Run inference
            transcription, timings, output_shape = self.run_single_inference(mel_input)
            
            # Record results
            result = {
                'iteration': i + 1,
                'timings': timings,
                'transcription': transcription,
                'output_shape': output_shape
            }
            
            all_timings.append(timings)
            transcriptions.append(transcription)
            self.results['benchmark_results'].append(result)
            
            print(f"Iteration {i+1}/{iterations}: {timings['total']:.4f}s")
        
        # Calculate statistics
        total_times = [t['total'] for t in all_timings]
        inference_times = [t['inference'] for t in all_timings]
        
        self.results['summary'] = {
            'preprocessing_time': preprocessing_time,
            'iterations': iterations,
            'warmup_runs': warmup_runs,
            'total_time': {
                'mean': statistics.mean(total_times),
                'median': statistics.median(total_times),
                'std': statistics.stdev(total_times) if len(total_times) > 1 else 0,
                'min': min(total_times),
                'max': max(total_times)
            },
            'inference_time': {
                'mean': statistics.mean(inference_times),
                'median': statistics.median(inference_times),
                'std': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                'min': min(inference_times),
                'max': max(inference_times)
            },
            'throughput': {
                'inferences_per_second': 1.0 / statistics.mean(total_times),
                'audio_realtime_factor': 30.0 / statistics.mean(total_times)  # 30s audio
            },
            'sample_transcription': transcriptions[0] if transcriptions else ""
        }
    
    def print_results(self):
        """Print formatted benchmark results"""
        s = self.results['summary']
        
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Mean Inference Time:     {s['inference_time']['mean']*1000:.2f} ms")
        print(f"   Mean Total Time:         {s['total_time']['mean']*1000:.2f} ms")
        print(f"   Std Deviation:           {s['total_time']['std']*1000:.2f} ms")
        print(f"   Min/Max Time:            {s['total_time']['min']*1000:.2f} / {s['total_time']['max']*1000:.2f} ms")
        
        print(f"\n⚡ THROUGHPUT:")
        print(f"   Inferences/Second:       {s['throughput']['inferences_per_second']:.2f}")
        print(f"   Real-time Factor:        {s['throughput']['audio_realtime_factor']:.2f}x")
        
        
        print(f"\n🔧 SYSTEM INFO:")
        si = self.results['system_info']
        print(f"   Platform:                {si['platform']}")
        print(f"   Architecture:            {si['architecture']}")
        
        print(f"\n📝 SAMPLE OUTPUT:")
        print(f"   \"{s['sample_transcription'][:100]}{'...' if len(s['sample_transcription']) > 100 else ''}\"")
        
        
    
    def save_results(self, output_file=None):
        """Save detailed results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"whisper_benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {output_file}")

def main():
    # Configuration (update paths as needed)
    model_path = "/home/scmd/whisper-tiny-en.tflite"
    vocab_mel_path = "/home/scmd/tflt-vocab-mel.bin"
    audio_file = "/home/scmd/input.wav"
    
    # Create benchmark instance
    benchmark = WhisperBenchmark(model_path, vocab_mel_path, audio_file)
    
    # Run benchmark (adjust iterations as needed)
    benchmark.benchmark_inference(iterations=20, warmup_runs=3)
    
    # Display results
    benchmark.print_results()
    
    # Save detailed results
    benchmark.save_results()

if __name__ == "__main__":
    main()
