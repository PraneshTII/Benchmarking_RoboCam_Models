#!/usr/bin/env python3
"""
TensorFlow Lite Model Inspector and Inference Script
Inspects model inputs/outputs and runs inference with random tensors
"""
import sys;
import site;
import functools;


functools.reduce(lambda k, p: site.addsitedir(p, k), 
['/nix/store/7wdal3kjsp1hn6ig64m2y6baxg4f929h-tflite-opencv-test-app-cpu-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/1bvpxg3kvzjhrp7n9q114xcjxmyx2ik8-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/d5hmgjyy2wy17k79z1j1gjs0fv1wh5ki-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/n60lsdw7cm43lv10bdrcdcxqv7dpnn0b-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages',
'/nix/store/s58a7qp81ms8an7xdc7h008d8kc13kqp-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import tflite_runtime.interpreter as tflite
import traceback

# List of model files
model_files = [
    "alexa_v0.1.tflite",
    "tiny_wav2letter_int8.tflite",
    "wav2letter_int8.tflite",
    "wav2letter_pruned_int8.tflite"
]

def inspect_model(model_path):
    """Inspect TFLite model inputs and outputs"""
    print(f"\n{'='*60}")
    print(f"INSPECTING: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load interpreter
        interpreter = tflite.Interpreter(model_path=model_path,
                    experimental_delegates=[
                        tflite.load_delegate('/nix/store/96bsy96b042wsqgzazpdhcdkqhai9k7n-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
                    ])
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nModel: {os.path.basename(model_path)}")
        print(f"File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        print(f"\nINPUT DETAILS ({len(input_details)} inputs):")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            print(f"    Index: {detail['index']}")
            if 'quantization_parameters' in detail:
                quant = detail['quantization_parameters']
                if quant['scales'].size > 0:
                    print(f"    Quantization: scale={quant['scales']}, zero_point={quant['zero_points']}")
        
        print(f"\nOUTPUT DETAILS ({len(output_details)} outputs):")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    Name: {detail['name']}")
            print(f"    Shape: {detail['shape']}")
            print(f"    Type: {detail['dtype']}")
            print(f"    Index: {detail['index']}")
            if 'quantization_parameters' in detail:
                quant = detail['quantization_parameters']
                if quant['scales'].size > 0:
                    print(f"    Quantization: scale={quant['scales']}, zero_point={quant['zero_points']}")
        
        return interpreter, input_details, output_details
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return None, None, None

def create_random_tensor(shape, dtype):
    """Create random tensor based on shape and dtype"""
    if dtype == np.float32:
        # Random float values between -1 and 1
        return np.random.uniform(-1.0, 1.0, shape).astype(dtype)
    elif dtype == np.float16:
        # Random float16 values
        return np.random.uniform(-1.0, 1.0, shape).astype(dtype)
    elif dtype == np.int32:
        # Random integers (often for attention masks, use 0s and 1s)
        return np.random.randint(0, 2, shape).astype(dtype)
    elif dtype == np.int8:
        # Quantized values, typically range around 0
        return np.random.randint(-128, 127, shape).astype(dtype)
    elif dtype == np.uint8:
        # Unsigned quantized values
        return np.random.randint(0, 255, shape).astype(dtype)
    else:
        print(f"    WARNING: Unsupported dtype {dtype}, using float32")
        return np.random.uniform(-1.0, 1.0, shape).astype(np.float32)

def run_inference(interpreter, input_details, output_details, model_name):
    """Run inference with random inputs"""
    print(f"\n{'='*40}")
    print(f"RUNNING INFERENCE: {model_name}")
    print(f"{'='*40}")
    
    try:
        # Create random inputs
        random_inputs = []
        for i, detail in enumerate(input_details):
            shape = detail['shape']
            dtype = detail['dtype']
            
            print(f"\nGenerating random input {i}:")
            print(f"  Shape: {shape}, Type: {dtype}")
            
            # Handle dynamic shapes (replace -1 with reasonable values)
            if -1 in shape:
                new_shape = []
                for dim in shape:
                    if dim == -1:
                        # Common reasonable values for TTS models
                        if len(shape) == 2:  # Likely sequence length
                            new_shape.append(100)  # 100 time steps
                        elif len(shape) == 3:  # Likely batch, time, features
                            new_shape.append(50)   # 50 time steps
                        else:
                            new_shape.append(10)   # Default
                    else:
                        new_shape.append(dim)
                shape = tuple(new_shape)
                print(f"  Resolved dynamic shape to: {shape}")
            
            random_tensor = create_random_tensor(shape, dtype)
            random_inputs.append(random_tensor)
            print(f"  Generated tensor shape: {random_tensor.shape}")
            print(f"  Value range: [{random_tensor.min():.4f}, {random_tensor.max():.4f}]")
        
        # Set input tensors
        for i, (detail, tensor) in enumerate(zip(input_details, random_inputs)):
            interpreter.set_tensor(detail['index'], tensor)
            print(f"  Set input {i} successfully")
        
        # Run inference
        print(f"\nRunning inference...")
        interpreter.invoke()
        print(f"Inference completed successfully!")
        
        # Get outputs
        print(f"\nOUTPUT RESULTS:")
        outputs = []
        for i, detail in enumerate(output_details):
            output = interpreter.get_tensor(detail['index'])
            outputs.append(output)
            print(f"  Output {i}:")
            print(f"    Shape: {output.shape}")
            print(f"    Type: {output.dtype}")
            print(f"    Value range: [{output.min():.4f}, {output.max():.4f}]")
            if output.size < 20:  # Show small outputs
                print(f"    Values: {output.flatten()}")
        
        return outputs
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        traceback.print_exc()
        return None

def main():
    """Main function to process all models"""
    print("TensorFlow Lite Model Inspector and Inference Runner")
    print("=" * 60)
    
    successful_models = []
    failed_models = []
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"\nSKIPPING: {model_file} (file not found)")
            failed_models.append(model_file)
            continue
        
        # Inspect model
        interpreter, input_details, output_details = inspect_model(model_file)
        
        if interpreter is None:
            failed_models.append(model_file)
            continue
        
        # Run inference
        outputs = run_inference(interpreter, input_details, output_details, model_file)
        
        if outputs is not None:
            successful_models.append(model_file)
        else:
            failed_models.append(model_file)
        
        print(f"\n{'-'*60}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful_models)} models")
    for model in successful_models:
        print(f"  ✓ {model}")
    
    print(f"\nFailed models: {len(failed_models)} models")
    for model in failed_models:
        print(f"  ✗ {model}")

# Individual model inspection functions (for line-by-line execution)
def inspect_fastspeech():
    """Inspect FastSpeech model specifically"""
    return inspect_model("fastspeech_quant.tflite")

def inspect_melgan():
    """Inspect MelGAN model specifically"""
    return inspect_model("melgan_float16.tflite")

def inspect_pwg_dr():
    """Inspect Parallel WaveGAN (DR) model specifically"""
    return inspect_model("parallel_wavegan_dr.tflite")

def inspect_pwg_f16():
    """Inspect Parallel WaveGAN (Float16) model specifically"""
    return inspect_model("parallel_wavegan_float16.tflite")

def inspect_tts():
    """Inspect TTS model specifically"""
    return inspect_model("tts_model.tflite")

def inspect_vocoder():
    """Inspect Vocoder model specifically"""
    return inspect_model("vocoder_model.tflite")


if __name__ == "__main__":
    main()
