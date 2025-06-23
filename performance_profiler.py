#!/usr/bin/env python3
"""
Detailed Performance Profiler - Shows exactly where time is spent
including Camera, Processing and NPU. 
Usage ( by default uses yolov4-tiny model) : 
 - python3 performance_profiler.py 
"""
import sys;import site;import functools;sys.argv[0] = '/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/bin/tf_lite_test.py';functools.reduce(lambda k, p: site.addsitedir(p, k), ['/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/s31xv27jkg2k2qsgs62spd5cgb3h9ira-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/8l32kk3qvcda079lv8506r71h6hl86pn-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/qn2z3wnqi8knn84chryz0iww5kkshaz7-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/rrjjkwkcx1fy43md458yd7avcr0hy0q2-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import tflite_runtime.interpreter as tflite

class DetailedPerformanceProfiler:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        print("Loading model for detailed profiling...")
        
        # Load NPU interpreter
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[
                tflite.load_delegate('/nix/store/nxds5qr80pafcvcx9yvsvn7wg14qk7qg-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
            ]
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Warmup
        print("Warming up...")
        input_shape = self.input_details[0]['shape']
        dummy_input = np.zeros(input_shape, dtype=self.input_details[0]['dtype'])
        self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
        self.interpreter.invoke()
        print("Ready for profiling!")
        
        # Performance counters
        self.timings = {
            'camera_capture': [],
            'preprocessing': [],
            'inference': [],
            'postprocessing': [],
            'display': [],
            'total_frame': []
        }

    def profile_camera_only(self, camera_index=3, num_frames=100):
        """Profile just camera capture speed"""
        print(f"\n📹 Profiling camera capture only ({num_frames} frames)...")
        
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        times = []
        for i in range(num_frames):
            start = time.time()
            ret, frame = cap.read()
            capture_time = (time.time() - start) * 1000
            times.append(capture_time)
            
            if i % 20 == 0:
                print(f"  Frame {i}: {capture_time:.1f}ms")
        
        cap.release()
        
        avg_time = sum(times) / len(times)
        max_fps = 1000 / avg_time
        
        print(f"📊 Camera Only Results:")
        print(f"   Average capture: {avg_time:.1f}ms")
        print(f"   Max camera FPS: {max_fps:.1f}")
        
        return avg_time

    def profile_preprocessing_only(self, test_image):
        """Profile just preprocessing speed"""
        print(f"\n⚙️ Profiling preprocessing only...")
        
        times = []
        for i in range(50):
            start = time.time()
            
            # Resize
            resized = cv2.resize(test_image, (416, 416))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Quantization
            normalized = rgb_image.astype(np.float32) / 255.0
            quantized = (normalized / 0.003922) + (-128)
            quantized_clipped = np.clip(quantized, -128, 127).astype(np.int8)
            input_data = np.expand_dims(quantized_clipped, axis=0)
            
            preprocess_time = (time.time() - start) * 1000
            times.append(preprocess_time)
        
        avg_time = sum(times) / len(times)
        
        print(f"📊 Preprocessing Results:")
        print(f"   Average preprocessing: {avg_time:.1f}ms")
        
        return avg_time, input_data

    def profile_inference_only(self, input_data):
        """Profile just inference speed"""
        print(f"\n🧠 Profiling inference only...")
        
        times = []
        for i in range(50):
            start = time.time()
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            inference_time = (time.time() - start) * 1000
            times.append(inference_time)
        
        avg_time = sum(times) / len(times)
        
        print(f"📊 Inference Results:")
        print(f"   Average inference: {avg_time:.1f}ms")
        
        return avg_time

    def profile_postprocessing_only(self):
        """Profile just postprocessing speed"""
        print(f"\n🔄 Profiling postprocessing only...")
        
        # Get sample outputs
        output1 = self.interpreter.get_tensor(self.output_details[0]['index'])
        output2 = self.interpreter.get_tensor(self.output_details[1]['index'])
        
        times = []
        for i in range(50):
            start = time.time()
            
            # Simulate output parsing (simplified)
            detections = []
            for output in [output1, output2]:
                if len(output.shape) == 4:
                    output = output[0]
                
                grid_h, grid_w, channels = output.shape
                if channels >= 85:
                    # Process a few cells (not all for speed test)
                    for i in range(min(5, grid_h)):
                        for j in range(min(5, grid_w)):
                            for a in range(3):
                                # Simulate detection processing
                                confidence = 0.7
                                if confidence > self.confidence_threshold:
                                    detections.append([10, 10, 50, 50, confidence, 0])
            
            # Simulate NMS
            if detections:
                boxes = [[det[0], det[1], det[2]-det[0], det[3]-det[1]] for det in detections]
                scores = [det[4] for det in detections]
                cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)
            
            postprocess_time = (time.time() - start) * 1000
            times.append(postprocess_time)
        
        avg_time = sum(times) / len(times)
        
        print(f"📊 Postprocessing Results:")
        print(f"   Average postprocessing: {avg_time:.1f}ms")
        
        return avg_time

    def profile_display_only(self, test_image):
        """Profile just display speed"""
        print(f"\n🖥️ Profiling display only...")
        
        times = []
        for i in range(30):
            start = time.time()
            
            # Simulate drawing detections
            frame = test_image.copy()
            for j in range(5):  # Simulate 5 detections
                cv2.rectangle(frame, (j*50, j*30), (j*50+100, j*30+80), (0, 255, 0), 2)
                cv2.putText(frame, f'person: 0.85', (j*50, j*30-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Performance overlay
            cv2.putText(frame, f'Inference: 13.5ms', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Profile Test', frame)
            cv2.waitKey(1)
            
            display_time = (time.time() - start) * 1000
            times.append(display_time)
        
        cv2.destroyAllWindows()
        
        avg_time = sum(times) / len(times)
        
        print(f"📊 Display Results:")
        print(f"   Average display: {avg_time:.1f}ms")
        
        return avg_time

    def profile_complete_pipeline(self, camera_index=3):
        """Profile the complete pipeline"""
        print(f"\n🔄 Profiling complete pipeline...")
        
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        frame_count = 0
        while frame_count < 30:  # Profile 30 frames
            total_start = time.time()
            
            # 1. Camera capture
            capture_start = time.time()
            ret, frame = cap.read()
            if not ret:
                continue
            capture_time = (time.time() - capture_start) * 1000
            
            # 2. Preprocessing
            preprocess_start = time.time()
            resized = cv2.resize(frame, (416, 416))
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb_image.astype(np.float32) / 255.0
            quantized = (normalized / 0.003922) + (-128)
            quantized_clipped = np.clip(quantized, -128, 127).astype(np.int8)
            input_data = np.expand_dims(quantized_clipped, axis=0)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # 3. Inference
            inference_start = time.time()
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            inference_time = (time.time() - inference_start) * 1000
            
            # 4. Postprocessing (simplified)
            postprocess_start = time.time()
            detections = []  # Simplified - would normally parse outputs
            postprocess_time = (time.time() - postprocess_start) * 1000
            
            # 5. Display
            display_start = time.time()
            cv2.putText(frame, f'Frame {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Pipeline Profile', frame)
            cv2.waitKey(1)
            display_time = (time.time() - display_start) * 1000
            
            total_time = (time.time() - total_start) * 1000
            
            # Store timings
            self.timings['camera_capture'].append(capture_time)
            self.timings['preprocessing'].append(preprocess_time)
            self.timings['inference'].append(inference_time)
            self.timings['postprocessing'].append(postprocess_time)
            self.timings['display'].append(display_time)
            self.timings['total_frame'].append(total_time)
            
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count}: Total={total_time:.1f}ms "
                      f"(Cap={capture_time:.1f} + Pre={preprocess_time:.1f} + "
                      f"Inf={inference_time:.1f} + Post={postprocess_time:.1f} + "
                      f"Disp={display_time:.1f})")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.generate_report()

    def generate_report(self):
        """Generate detailed performance report"""
        print(f"\n" + "="*60)
        print("🔍 DETAILED PERFORMANCE BREAKDOWN")
        print("="*60)
        
        for component, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                if component == 'total_frame':
                    fps = 1000 / avg_time
                    print(f"📊 {component.upper()}:")
                    print(f"   Average: {avg_time:.1f}ms ({fps:.1f} FPS)")
                else:
                    percentage = (avg_time / sum([sum(t)/len(t) for t in self.timings.values() if t])) * 100
                    print(f"⏱️  {component.upper()}:")
                    print(f"   Average: {avg_time:.1f}ms ({percentage:.1f}% of total)")
                
                print(f"   Range: {min_time:.1f} - {max_time:.1f}ms")
                print()
        
        print("🎯 OPTIMIZATION RECOMMENDATIONS:")
        
        # Calculate component averages
        avg_times = {comp: sum(times)/len(times) for comp, times in self.timings.items() if times}
        
        if avg_times.get('camera_capture', 0) > 20:
            print("   📹 Camera capture is slow - try lower resolution")
        
        if avg_times.get('preprocessing', 0) > 10:
            print("   ⚙️ Preprocessing is slow - optimize image operations")
        
        if avg_times.get('inference', 0) > 15:
            print("   🧠 Inference is slow - NPU may not be fully utilized")
        
        if avg_times.get('display', 0) > 15:
            print("   🖥️ Display is slow - try --no_display for max speed")
        
        total_without_display = sum([avg_times.get(comp, 0) 
                                   for comp in ['camera_capture', 'preprocessing', 'inference', 'postprocessing']])
        max_fps_no_display = 1000 / total_without_display
        
        print(f"\n📈 THEORETICAL MAXIMUM:")
        print(f"   Without display: {max_fps_no_display:.1f} FPS")
        print(f"   With display: {1000/avg_times.get('total_frame', 50):.1f} FPS")

def main():
    model_path = "/home/scmd/yolo-v4-tiny.tflite"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    profiler = DetailedPerformanceProfiler(model_path)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Profile individual components
    camera_time = profiler.profile_camera_only(camera_index=3, num_frames=50)
    preprocess_time, input_data = profiler.profile_preprocessing_only(test_image)
    inference_time = profiler.profile_inference_only(input_data)
    postprocess_time = profiler.profile_postprocessing_only()
    display_time = profiler.profile_display_only(test_image)
    
    print(f"\n" + "="*50)
    print("📋 COMPONENT SUMMARY:")
    print(f"   Camera capture: {camera_time:.1f}ms")
    print(f"   Preprocessing:  {preprocess_time:.1f}ms") 
    print(f"   Inference:      {inference_time:.1f}ms")
    print(f"   Postprocessing: {postprocess_time:.1f}ms")
    print(f"   Display:        {display_time:.1f}ms")
    
    total_theoretical = camera_time + preprocess_time + inference_time + postprocess_time + display_time
    theoretical_fps = 1000 / total_theoretical
    
    print(f"   TOTAL:          {total_theoretical:.1f}ms ({theoretical_fps:.1f} FPS)")
    
    # Profile complete pipeline
    print(f"\nRunning complete pipeline test...")
    profiler.profile_complete_pipeline(camera_index=3)

if __name__ == "__main__":
    main()
