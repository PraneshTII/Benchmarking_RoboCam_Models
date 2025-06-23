#!/usr/bin/env python3
"""
Single-Instance YOLOv4-Tiny detector 
- This program loads the model once and performs repeated inference on the same instance. 
- Pure NPU of Tiny- Yolov4 was found to be ~72-75 FPS.
- End to End FPS is found to be 20 FPS
 
- Example usage: 
  python3 yolov4tiny_detector_v3.py --model /home/scmd/yolo-v4-tiny.tflite --camera 3 --threshold 0.05
  
  (without display)
  python3 yolov4tiny_detector_v3.py --model /home/scmd/yolo-v4-tiny.tflite --camera 3 --threshold 0.05 --no_display
  
"""

import sys;import site;import functools;sys.argv[0] = '/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/bin/tf_lite_test.py';functools.reduce(lambda k, p: site.addsitedir(p, k), ['/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/s31xv27jkg2k2qsgs62spd5cgb3h9ira-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/8l32kk3qvcda079lv8506r71h6hl86pn-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/qn2z3wnqi8knn84chryz0iww5kkshaz7-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/rrjjkwkcx1fy43md458yd7avcr0hy0q2-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());

import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import tflite_runtime.interpreter as tflite
from collections import OrderedDict

class SingleInstanceYOLOv4TinyDetector:
    def __init__(self, model_path, confidence_threshold=0.5, use_npu=True):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        print(f"Loading YOLOv4-Tiny model: {os.path.basename(model_path)}")
        print("⏳ ONE-TIME model compilation ...")
        
        # Create SINGLE interpreter instance that will be reused
        if use_npu:
            try:
                print("Creating NPU interpreter (one-time compilation)...")
                compilation_start = time.time()
                
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[
                        tflite.load_delegate('/nix/store/nxds5qr80pafcvcx9yvsvn7wg14qk7qg-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
                    ]
                )
                self.interpreter.allocate_tensors()
                
                compilation_time = (time.time() - compilation_start) * 1000
                print(f"✓ NPU compiled in {compilation_time:.0f}ms")
                self.using_npu = True
                
            except Exception as e:
                print(f"⚠ NPU failed, using CPU: {e}")
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.using_npu = False
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.using_npu = False
        
        # Get model details once
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Print model info
        print(f"Input: {self.input_details[0]['shape']} {self.input_details[0]['dtype']}")
        print(f"Number of outputs: {len(self.output_details)}")
        
        # Get input size
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 4:
            self.input_size = (input_shape[2], input_shape[1])  # (width, height)
        else:
            self.input_size = (416, 416)
        
        print(f"Input size: {self.input_size}")
        
        # Get quantization parameters
        self.input_scale, self.input_zero_point = self._get_quantization_params(self.input_details[0])
        
        self.output_scales = []
        self.output_zero_points = []
        for output_detail in self.output_details:
            scale, zero_point = self._get_quantization_params(output_detail)
            self.output_scales.append(scale)
            self.output_zero_points.append(zero_point)
        
        print(f"Quantization: scale={self.input_scale:.6f}, zero_point={self.input_zero_point}")
        
        # Pre-allocate input buffer to avoid allocations during inference
        self.input_buffer = np.zeros(self.input_details[0]['shape'], dtype=self.input_details[0]['dtype'])
        
        # Critical: Do the first inference to complete NPU compilation
        print("🔥 Completing NPU compilation with first inference...")
        warmup_start = time.time()
        self._complete_npu_compilation()
        warmup_time = (time.time() - warmup_start) * 1000
        print(f"✓ NPU fully compiled in {warmup_time:.0f}ms")
        
        # YOLOv4-tiny anchors
        self.anchors = [
            [[10, 14], [23, 27], [37, 58]],
            [[81, 82], [135, 169], [344, 319]]
        ]
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Performance tracking
        self.inference_times = []
        
        # Test actual performance now that compilation is complete
        print("🧪 Testing actual inference speed...")
        test_times = []
        for i in range(5):
            start = time.time()
            self._complete_npu_compilation()  # Reuse the test inference
            test_time = (time.time() - start) * 1000
            test_times.append(test_time)
            print(f"   Test {i+1}: {test_time:.1f}ms")
        
        avg_test_time = sum(test_times) / len(test_times)
        expected_fps = 1000 / avg_test_time
        
        print(f"✅ Average inference: {avg_test_time:.1f}ms ({expected_fps:.1f} FPS)")
        print(f"🚀 Model ready! NPU: {'Active' if self.using_npu else 'Inactive'}")

    def _complete_npu_compilation(self):
        """Complete NPU compilation with a real inference"""
        # Use the pre-allocated buffer
        if self.input_details[0]['dtype'] == np.int8:
            self.input_buffer.fill(0)  # Fill with zeros
        else:
            self.input_buffer.fill(0.5)  # Fill with mid-range values
        
        # Run inference using the single interpreter instance
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_buffer)
        self.interpreter.invoke()

    def _get_quantization_params(self, tensor_details):
        """Extract quantization parameters"""
        qparams = tensor_details.get('quantization_parameters', {})
        scales = qparams.get('scales', [1.0])
        zero_points = qparams.get('zero_points', [0])
        return scales[0] if scales else 1.0, zero_points[0] if zero_points else 0

    def preprocess_image_fast(self, image):
        """Ultra-fast preprocessing using pre-allocated buffer"""
        # Resize image
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Handle quantization based on input type
        if self.input_details[0]['dtype'] == np.int8:
            # INT8 quantization - fix casting issue
            normalized = rgb_image.astype(np.float32) / 255.0
            quantized = (normalized / self.input_scale) + self.input_zero_point
            quantized_clipped = np.clip(quantized, -128, 127).astype(np.int8)
            self.input_buffer[0] = quantized_clipped
        elif self.input_details[0]['dtype'] == np.uint8:
            # UINT8 quantization
            normalized = rgb_image.astype(np.float32) / 255.0
            quantized = (normalized / self.input_scale) + self.input_zero_point
            quantized_clipped = np.clip(quantized, 0, 255).astype(np.uint8)
            self.input_buffer[0] = quantized_clipped
        else:
            # FLOAT32
            self.input_buffer[0] = rgb_image.astype(np.float32) / 255.0
        
        return self.input_buffer

    def detect_objects_fast(self, image, target_class='person'):
        """Ultra-fast detection using the single interpreter instance"""
        img_height, img_width = image.shape[:2]
        target_class_id = self.class_names.index(target_class) if target_class in self.class_names else 0
        
        # Fast preprocessing
        input_data = self.preprocess_image_fast(image)
        
        # Fast inference using the SAME interpreter (no recompilation!)
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Track performance
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # Parse outputs
        detections = self._parse_outputs_fast(target_class_id, img_width, img_height)
        
        return detections, inference_time

    def _parse_outputs_fast(self, target_class_id, img_width, img_height):
        """Fast output parsing focused on target class"""
        all_detections = []
        
        for i, output_detail in enumerate(self.output_details):
            raw_output = self.interpreter.get_tensor(output_detail['index'])
            
            # Dequantize if needed
            if raw_output.dtype in [np.int8, np.uint8]:
                scale = self.output_scales[i]
                zero_point = self.output_zero_points[i]
                output = scale * (raw_output.astype(np.float32) - zero_point)
            else:
                output = raw_output.astype(np.float32)
            
            # Parse this layer efficiently
            layer_detections = self._parse_layer_fast(output, i, target_class_id, img_width, img_height)
            all_detections.extend(layer_detections)
        
        # Quick NMS
        return self._apply_nms_fast(all_detections)

    def _parse_layer_fast(self, output, layer_idx, target_class_id, img_width, img_height):
        """Fast layer parsing"""
        detections = []
        
        # Handle batch dimension
        if len(output.shape) == 4:
            output = output[0]
        
        grid_h, grid_w, channels = output.shape
        
        # Determine anchor count and classes
        if channels == 255:  # 3 * (5 + 80)
            num_anchors = 3
            num_classes = 80
        elif channels == 18:   # 3 * (5 + 1) 
            num_anchors = 3
            num_classes = 1
        else:
            return detections
        
        # Reshape for processing
        output = output.reshape(grid_h, grid_w, num_anchors, 5 + num_classes)
        
        # Get anchors for this layer
        if layer_idx < len(self.anchors):
            layer_anchors = self.anchors[layer_idx]
        else:
            layer_anchors = self.anchors[0]
        
        # Vectorized objectness calculation
        objectness = 1 / (1 + np.exp(-np.clip(output[:, :, :, 4], -500, 500)))
        
        # Find promising cells
        promising_mask = objectness > 0.05
        promising_indices = np.where(promising_mask)
        
        # Process promising detections
        for idx in range(len(promising_indices[0])):
            i, j, a = promising_indices[0][idx], promising_indices[1][idx], promising_indices[2][idx]
            
            if a >= len(layer_anchors):
                continue
            
            prediction = output[i, j, a]
            obj_score = objectness[i, j, a]
            
            # Coordinates
            x = 1 / (1 + np.exp(-prediction[0]))
            y = 1 / (1 + np.exp(-prediction[1]))
            
            center_x = (j + x) / grid_w
            center_y = (i + y) / grid_h
            
            # Size
            anchor_w, anchor_h = layer_anchors[a]
            width = (anchor_w * np.exp(np.clip(prediction[2], -10, 10))) / self.input_size[0]
            height = (anchor_h * np.exp(np.clip(prediction[3], -10, 10))) / self.input_size[1]
            
            # Class confidence
            if num_classes > 1 and target_class_id < num_classes:
                class_score = 1 / (1 + np.exp(-np.clip(prediction[5 + target_class_id], -500, 500)))
            else:
                class_score = 1.0
            
            confidence = obj_score * class_score
            
            if confidence > self.confidence_threshold:
                # Convert to pixels
                x1 = max(0, min(int((center_x - width/2) * img_width), img_width-1))
                y1 = max(0, min(int((center_y - height/2) * img_height), img_height-1))
                x2 = max(0, min(int((center_x + width/2) * img_width), img_width-1))
                y2 = max(0, min(int((center_y + height/2) * img_height), img_height-1))
                
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, confidence, target_class_id])
        
        return detections

    def _apply_nms_fast(self, detections):
        """Fast NMS"""
        if len(detections) <= 1:
            return detections
        
        detections = np.array(detections)
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        # Convert to [x, y, w, h] for OpenCV NMS
        nms_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            nms_boxes.append([x1, y1, x2-x1, y2-y1])
        
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores.tolist(),
            self.confidence_threshold,
            0.45
        )
        
        if len(indices) > 0:
            return detections[indices.flatten()].tolist()
        return []

    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return None
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        min_time = min(self.inference_times)
        max_time = max(self.inference_times)
        
        return {
            'avg_inference_ms': avg_time,
            'min_inference_ms': min_time,
            'max_inference_ms': max_time,
            'avg_fps': 1000 / avg_time if avg_time > 0 else 0,
            'samples': len(self.inference_times)
        }

class FastVideoStream:
    def __init__(self, resolution=(640,480), framerate=30, camera_index=0):
        self.stream = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def main():
    parser = argparse.ArgumentParser(description='Single-Instance High-Performance YOLOv4-Tiny')
    parser.add_argument('--model', required=True, help='Path to YOLOv4-tiny .tflite model')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--target_class', default='person', help='Target class to detect')
    parser.add_argument('--no_npu', action='store_true', help='Disable NPU acceleration')
    parser.add_argument('--no_display', action='store_true', help='Disable display for max performance')
    
    args = parser.parse_args()
    
    print("Single-Instance High-Performance YOLOv4-Tiny")
    print("=" * 50)
    
    # Initialize detector (one-time compilation)
    detector = SingleInstanceYOLOv4TinyDetector(args.model, args.threshold, use_npu=not args.no_npu)
    
    # Initialize camera
    videostream = FastVideoStream(camera_index=args.camera).start()
    time.sleep(1)
    
    print(f"\n🎬 Starting high-speed detection...")
    print(f"Target class: {args.target_class}")
    print(f"Confidence threshold: {args.threshold}")
    print("Press 'q' to quit, 's' for performance stats")
    
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    
    try:
        while True:
            frame = videostream.read()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Ultra-fast detection (should be ~13ms now!)
            detections, inference_time = detector.detect_objects_fast(frame, args.target_class)
            
            # Calculate and display FPS every 30 frames
            fps_counter += 1
            if fps_counter >= 30:
                elapsed = time.time() - fps_start
                current_fps = 30 / elapsed
                fps_start = time.time()
                fps_counter = 0
                print(f"🚀 FPS: {current_fps:.1f} | Inference: {inference_time:.1f}ms | Detections: {len(detections)}")
            
            if not args.no_display:
                # Draw detections
                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{args.target_class}: {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Performance overlay
                cv2.putText(frame, f'Inference: {inference_time:.1f}ms', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'NPU: {"ON" if detector.using_npu else "OFF"}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detector.using_npu else (0, 0, 255), 2)
                
                cv2.imshow('High-Speed YOLOv4-Tiny', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = detector.get_performance_stats()
                    if stats:
                        print(f"\n📊 Performance Stats:")
                        print(f"   Average: {stats['avg_inference_ms']:.1f}ms ({stats['avg_fps']:.1f} FPS)")
                        print(f"   Range: {stats['min_inference_ms']:.1f} - {stats['max_inference_ms']:.1f}ms")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Cleanup
        videostream.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Final stats
        final_stats = detector.get_performance_stats()
        if final_stats:
            print(f"\n🏁 Final Results:")
            print(f"   Average inference: {final_stats['avg_inference_ms']:.1f}ms")
            print(f"   Theoretical FPS: {final_stats['avg_fps']:.1f}")
            print(f"   NPU: {'Active' if detector.using_npu else 'Inactive'}")

if __name__ == '__main__':
    main()
