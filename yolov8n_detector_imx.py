#!/usr/bin/env python3
"""
Single-Instance YOLOv8n detector for IMX8MP
- Optimized for YOLOv8n with better INT7 quantization support
- Expected performance: 6-8 FPS on IMX8MP NPU (faster than YOLOv5s)
 
- Example usage: 
  python3 yolov8n_detector_imx.py --model /home/scmd/yolov8n-640-int8.tflite --camera 3 --threshold 0.25
  
"""

#import sys;import site;import functools;sys.argv[0] = '/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/bin/tf_lite_test.py';functools.reduce(lambda k, p: site.addsitedir(p, k), ['/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/s31xv27jkg2k2qsgs62spd5cgb3h9ira-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/8l32kk3qvcda079lv8506r71h6hl86pn-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/qn2z3wnqi8knn84chryz0iyq5kkshaz7-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/rrjjkwkcx1fy43md458yd7avcr0hy0q2-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());

import sys;import site;import functools;sys.argv[0] = '/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/bin/tf_lite_test.py';functools.reduce(lambda k, p: site.addsitedir(p, k), ['/nix/store/l622cd50w22zncfvx2zy4dnaw9312pmd-tflite-opencv-test-app-1.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/s31xv27jkg2k2qsgs62spd5cgb3h9ira-python3.12-pillow-11.0.0-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/8l32kk3qvcda079lv8506r71h6hl86pn-python3.12-opencv-imx-python-4.20-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/qn2z3wnqi8knn84chryz0iww5kkshaz7-python3.12-tflite-imx-python-2.14-aarch64-unknown-linux-gnu/lib/python3.12/site-packages','/nix/store/rrjjkwkcx1fy43md458yd7avcr0hy0q2-python3.12-numpy-1.26.4-aarch64-unknown-linux-gnu/lib/python3.12/site-packages'], site._init_pathinfo());



import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import tflite_runtime.interpreter as tflite

class SingleInstanceYOLOv8nDetector:
    def __init__(self, model_path, confidence_threshold=0.25, use_npu=True):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        print(f"Loading YOLOv8n model: {os.path.basename(model_path)}")
        print("⏳ ONE-TIME model compilation ...")
        
        # Create SINGLE interpreter instance
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
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Input: {self.input_details[0]['shape']} {self.input_details[0]['dtype']}")
        print(f"Number of outputs: {len(self.output_details)}")
        
        for i, output in enumerate(self.output_details):
            print(f"Output {i}: {output['shape']} {output['dtype']}")
        
        # Get input size
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 4:
            self.input_size = (input_shape[2], input_shape[1])  # (width, height)
        else:
            self.input_size = (640, 640)  # Default for YOLOv8n
        
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
        
        # Pre-allocate input buffer
        self.input_buffer = np.zeros(self.input_details[0]['shape'], dtype=self.input_details[0]['dtype'])
        
        # Complete NPU compilation
        print("🔥 Completing NPU compilation with first inference...")
        warmup_start = time.time()
        self._complete_npu_compilation()
        warmup_time = (time.time() - warmup_start) * 1000
        print(f"✓ NPU fully compiled in {warmup_time:.0f}ms")
        
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
        
        # Test performance
        print("🧪 Testing inference speed...")
        test_times = []
        for i in range(5):
            start = time.time()
            self._complete_npu_compilation()
            test_time = (time.time() - start) * 1000
            test_times.append(test_time)
            print(f"   Test {i+1}: {test_time:.1f}ms")
        
        avg_test_time = sum(test_times) / len(test_times)
        expected_fps = 1000 / avg_test_time
        
        print(f"✅ Average inference: {avg_test_time:.1f}ms ({expected_fps:.1f} FPS)")
        print(f"🚀 YOLOv8n ready! NPU: {'Active' if self.using_npu else 'Inactive'}")

    def _complete_npu_compilation(self):
        """Complete NPU compilation with a real inference"""
        if self.input_details[0]['dtype'] == np.int8:
            self.input_buffer.fill(0)
        elif self.input_details[0]['dtype'] == np.uint8:
            self.input_buffer.fill(128)
        else:
            self.input_buffer.fill(0.5)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_buffer)
        self.interpreter.invoke()

    def _get_quantization_params(self, tensor_details):
        """Extract quantization parameters"""
        qparams = tensor_details.get('quantization_parameters', {})
        scales = qparams.get('scales', [1.0])
        zero_points = qparams.get('zero_points', [0])
        return scales[0] if scales else 1.0, zero_points[0] if zero_points else 0

    def preprocess_image_fast(self, image):
        """YOLOv8n preprocessing with letterboxing"""
        img_height, img_width = image.shape[:2]
        target_width, target_height = self.input_size
        
        # Calculate scale and padding
        scale = min(target_width / img_width, target_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image with gray background (114) - YOLOv8 standard
        padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_top = (target_height - new_height) // 2
        pad_left = (target_width - new_width) // 2
        
        # Place resized image in center
        padded[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Handle quantization
        if self.input_details[0]['dtype'] == np.int8:
            # INT8 quantization
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
        
        return self.input_buffer, scale, pad_left, pad_top

    def detect_objects_fast(self, image, target_class='person'):
        """Ultra-fast YOLOv8n detection"""
        img_height, img_width = image.shape[:2]
        target_class_id = self.class_names.index(target_class) if target_class in self.class_names else 0
        
        # Fast preprocessing
        input_data, scale, pad_left, pad_top = self.preprocess_image_fast(image)
        
        # Fast inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Track performance
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # Parse YOLOv8n outputs (same format as YOLOv5)
        detections = self._parse_yolov8_outputs(target_class_id, img_width, img_height, scale, pad_left, pad_top)
        
        return detections, inference_time

    def _parse_yolov8_outputs(self, target_class_id, img_width, img_height, scale, pad_left, pad_top):
        """Parse YOLOv8n outputs - similar to YOLOv5 but often more stable"""
        all_detections = []
        
        for i, output_detail in enumerate(self.output_details):
            raw_output = self.interpreter.get_tensor(output_detail['index'])
            
            # Dequantize if needed
            if raw_output.dtype in [np.int8, np.uint8]:
                scale_quant = self.output_scales[i]
                zero_point = self.output_zero_points[i]
                output = scale_quant * (raw_output.astype(np.float32) - zero_point)
            else:
                output = raw_output.astype(np.float32)
            
            # Handle batch dimension and transpose if needed
            if len(output.shape) == 3:
                output = output[0]  # Remove batch dimension
                
                # Check if we need to transpose
                if output.shape[0] < output.shape[1] and output.shape[0] >= 84:
                    output = output.T  # Transpose to [N, 84/85]
            
            # YOLOv8 output format: [N, 84] where 84 = 4(bbox) + 80(classes) - no objectness!
            if len(output.shape) == 2 and output.shape[1] >= 84:
                detections = self._parse_yolov8_detections(
                    output, target_class_id, img_width, img_height, scale, pad_left, pad_top
                )
                all_detections.extend(detections)
        
        # Apply NMS
        return self._apply_nms_fast(all_detections)

    def _parse_yolov8_detections(self, output, target_class_id, img_width, img_height, scale, pad_left, pad_top):
        """Parse YOLOv8n detection format: [N, 84] where each row is [x, y, w, h, class0, class1, ...]"""
        detections = []
        
        num_detections, num_attrs = output.shape
        
        if num_attrs < 84:
            return detections
        
        # YOLOv8 format: [center_x, center_y, width, height, class0, class1, ...class79]
        # No objectness score - uses class scores directly
        boxes = output[:, :4]  # [center_x, center_y, width, height]
        class_scores = output[:, 4:]  # Class scores (80 classes)
        
        # Get the target class scores
        if target_class_id < class_scores.shape[1]:
            target_scores = class_scores[:, target_class_id]
        else:
            # If target class not in model, use max class score
            target_scores = np.max(class_scores, axis=1)
        
        # YOLOv8 often outputs better-calibrated scores, but still apply sigmoid if needed
        if target_scores.max() > 1.0:
            # Scores look like logits, apply sigmoid
            target_scores = 1 / (1 + np.exp(-np.clip(target_scores, -500, 500)))
        
        # Filter by confidence threshold
        conf_mask = target_scores > self.confidence_threshold
        
        if not np.any(conf_mask):
            return detections
        
        # Extract filtered detections
        filtered_boxes = boxes[conf_mask]
        filtered_conf = target_scores[conf_mask]
        
        # Convert coordinates
        for box, conf in zip(filtered_boxes, filtered_conf):
            center_x, center_y, width, height = box
            
            # YOLOv8 outputs are typically normalized (0-1)
            if center_x <= 1.0 and center_y <= 1.0:
                center_x *= self.input_size[0]
                center_y *= self.input_size[1]
                width *= self.input_size[0]
                height *= self.input_size[1]
            
            # Convert to corner format
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            # Remove padding and scale back to original image size
            x1 = (x1 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            x2 = (x2 - pad_left) / scale
            y2 = (y2 - pad_top) / scale
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # Only add valid boxes
            if x2 > x1 and y2 > y1:
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), target_class_id])
        
        return detections

    def _apply_nms_fast(self, detections):
        """Fast NMS using OpenCV"""
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
            0.45  # NMS threshold
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
    parser = argparse.ArgumentParser(description='Single-Instance High-Performance YOLOv8n')
    parser.add_argument('--model', required=True, help='Path to YOLOv8n .tflite model')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--threshold', type=float, default=0.25, help='Confidence threshold (YOLOv8n default: 0.25)')
    parser.add_argument('--target_class', default='person', help='Target class to detect')
    parser.add_argument('--no_npu', action='store_true', help='Disable NPU acceleration')
    parser.add_argument('--no_display', action='store_true', help='Disable display for max performance')
    
    args = parser.parse_args()
    
    print("Single-Instance High-Performance YOLOv8n")
    print("=" * 50)
    
    # Initialize detector (one-time compilation)
    detector = SingleInstanceYOLOv8nDetector(args.model, args.threshold, use_npu=not args.no_npu)
    
    # Initialize camera
    videostream = FastVideoStream(camera_index=args.camera).start()
    time.sleep(1)
    
    print(f"\n🎬 Starting high-speed YOLOv8n detection...")
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
            
            # Ultra-fast YOLOv8n detection
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
                cv2.putText(frame, f'YOLOv8n: {inference_time:.1f}ms', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'NPU: {"ON" if detector.using_npu else "OFF"}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detector.using_npu else (0, 0, 255), 2)
                
                cv2.imshow('High-Speed YOLOv8n', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = detector.get_performance_stats()
                    if stats:
                        print(f"\n📊 YOLOv8n Performance Stats:")
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
            print(f"\n🏁 YOLOv8n Final Results:")
            print(f"   Average inference: {final_stats['avg_inference_ms']:.1f}ms")
            print(f"   Theoretical FPS: {final_stats['avg_fps']:.1f}")
            print(f"   NPU: {'Active' if detector.using_npu else 'Inactive'}")
            print(f"   Model: YOLOv8n (optimized for efficiency)")

if __name__ == '__main__':
    main()
