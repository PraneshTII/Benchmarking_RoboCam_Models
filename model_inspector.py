#!/usr/bin/env python3
"""
TFLite Model Inspector - Examine model metadata and structure

-checks input and output tensor shape and tests with dummpy input tensor.
- usage: 
  python3 model_inspector.py --model yolo-v4-tiny.tflite
- 
"""
import sys;import site;import functools;sys.argv[0] = '/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/bin/tf_lite_test.py';functools.reduce(lambda k, p: site.addsitedir(p, k), ['/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/s31xv27jkg2k2qsgs62spd5cgb3h9ira-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/8l32kk3qvcda079lv8506r71h6hl86pn-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/qn2z3wnqi8knn84chryz0iww5kkshaz7-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/rrjjkwkcx1fy43md458yd7avcr0hy0q2-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());
import argparse
import tflite_runtime.interpreter as tflite
import numpy as np

def inspect_model(model_path):
    print(f"Inspecting model: {model_path}")
    print("=" * 60)
    
    try:
        # Load model
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Model loaded successfully!")
        print(f"Number of input tensors: {len(input_details)}")
        print(f"Number of output tensors: {len(output_details)}")
        
        # Input details
        print(f"\nINPUT TENSORS:")
        for i, input_detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    Name: {input_detail.get('name', 'Unknown')}")
            print(f"    Shape: {input_detail['shape']}")
            print(f"    Dtype: {input_detail['dtype']}")
            
            # Quantization info
            qparams = input_detail.get('quantization_parameters', {})
            if qparams.get('scales') and qparams.get('zero_points'):
                print(f"    Quantization: scale={qparams['scales'][0]:.6f}, zero_point={qparams['zero_points'][0]}")
            else:
                print(f"    Quantization: None (floating point)")
        
        # Output details
        print(f"\nOUTPUT TENSORS:")
        for i, output_detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    Name: {output_detail.get('name', 'Unknown')}")
            print(f"    Shape: {output_detail['shape']}")
            print(f"    Dtype: {output_detail['dtype']}")
            
            # Quantization info
            qparams = output_detail.get('quantization_parameters', {})
            if qparams.get('scales') and qparams.get('zero_points'):
                print(f"    Quantization: scale={qparams['scales'][0]:.6f}, zero_point={qparams['zero_points'][0]}")
            else:
                print(f"    Quantization: None (floating point)")
        
        # Test with dummy input
        print(f"\nTESTING WITH DUMMY INPUT:")
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        # Create test input
        if input_dtype == np.uint8:
            test_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
            print(f"Created random uint8 input: shape={test_input.shape}, range=[{test_input.min()}, {test_input.max()}]")
        elif input_dtype == np.int8:
            test_input = np.random.randint(-128, 128, input_shape, dtype=np.int8)
            print(f"Created random int8 input: shape={test_input.shape}, range=[{test_input.min()}, {test_input.max()}]")
        else:
            test_input = np.random.random(input_shape).astype(input_dtype)
            print(f"Created random float input: shape={test_input.shape}, range=[{test_input.min():.3f}, {test_input.max():.3f}]")
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        # Check outputs
        for i, output_detail in enumerate(output_details):
            output_data = interpreter.get_tensor(output_detail['index'])
            print(f"  Output {i} result:")
            print(f"    Shape: {output_data.shape}")
            print(f"    Dtype: {output_data.dtype}")
            print(f"    Range: [{output_data.min()}, {output_data.max()}]")
            print(f"    Mean: {output_data.mean():.6f}")
            print(f"    Std: {output_data.std():.6f}")
            print(f"    Non-zero values: {np.count_nonzero(output_data)}/{output_data.size}")
            
            # Check for signs of life in the output
            unique_values = len(np.unique(output_data))
            print(f"    Unique values: {unique_values}")
            
            if unique_values == 1:
                print(f"    ⚠️  WARNING: All output values are identical!")
            elif unique_values < 10:
                print(f"    ⚠️  WARNING: Very few unique values - possible quantization issue")
            else:
                print(f"    ✓ Output shows variation - model appears functional")
        
        # Model size info
        try:
            import os
            file_size = os.path.getsize(model_path)
            print(f"\nMODEL FILE INFO:")
            print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        except:
            pass
            
    except Exception as e:
        print(f"ERROR: Failed to load model - {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to .tflite model file', required=True)
    args = parser.parse_args()
    
    success = inspect_model(args.model)
    
    if success:
        print(f"\n" + "=" * 60)
        print("RECOMMENDATIONS:")
        print("1. If all outputs show identical values, the model may be corrupted")
        print("2. If very few unique values, quantization may have damaged the model")
        print("3. If the model shows variation, the issue may be preprocessing")
        print("4. Try comparing with a known working model")
    
if __name__ == '__main__':
    main()
