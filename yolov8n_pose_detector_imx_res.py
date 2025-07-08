#!/usr/bin/env python3
"""
Multi-Resolution YOLO Pose detector for IMX8MP
- Supports YOLOv8n-pose and YOLOv11n-pose models
- Handles 320x320, 416x416, 640x640 and custom resolutions
- Real-time pose estimation with skeleton visualization
- Expected performance: 12+ FPS on 320x320, 8+ FPS on 416x416, 4+ FPS on 640x640
 
- Example usage: 
  python3 yolo_pose_detector_imx.py --model /home/scmd/yolov8n-pose-320-int8.tflite --camera 3
  python3 yolo_pose_detector_imx.py --model /home/scmd/yolov11n-pose-416-int8.tflite --camera 3
  python3 yolo_pose_detector_imx.py --model /home/scmd/yolov8n-pose.tflite --input-size 320 --camera 3
  
  # Export pose models:
  yolo export model=yolov8n-pose.pt format=tflite int8=True imgsz=320
  yolo export model=yolov11n-pose.pt format=tflite int8=True imgsz=416
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


class MultiResolutionYOLOPoseDetector:
    def __init__(self, model_path, confidence_threshold=0.25, use_npu=True, force_input_size=None):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        print(f"Loading YOLO Pose model: {os.path.basename(model_path)}")
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
        
        # Auto-detect or force input size
        input_shape = self.input_details[0]['shape']
        if force_input_size:
            if isinstance(force_input_size, (list, tuple)) and len(force_input_size) == 2:
                self.input_size = tuple(force_input_size)  # (width, height)
            else:
                self.input_size = (force_input_size, force_input_size)  # square
            print(f"🔧 FORCED input size: {self.input_size}")
        else:
            if len(input_shape) == 4:
                self.input_size = (input_shape[2], input_shape[1])  # (width, height)
            else:
                self.input_size = (640, 640)  # Default fallback
            print(f"🔍 AUTO-DETECTED input size: {self.input_size}")
        
        # Determine model type and expected outputs
        self._analyze_model_outputs()
        
        # Performance predictions
        width, height = self.input_size
        avg_size = (width + height) / 2
        if avg_size <= 320:
            expected_fps = "12-20 FPS"
            performance_tier = "🚀 ULTRA-FAST"
        elif avg_size <= 416:
            expected_fps = "8-15 FPS"
            performance_tier = "⚡ FAST"
        elif avg_size <= 512:
            expected_fps = "6-10 FPS"
            performance_tier = "🏃 MODERATE"
        else:
            expected_fps = "4-6 FPS"
            performance_tier = "🐌 ACCURATE"
        
        print(f"📈 Expected pose performance: {expected_fps} ({performance_tier})")
        
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
        
        # COCO pose keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections for drawing pose
        self.skeleton_connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes, eyes-ears
            # Body
            (5, 6),   # shoulders
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12), # shoulder-hip
            (11, 12), # hips
            # Legs
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16)  # right leg
        ]
        
        # Colors for different body parts
        self.pose_colors = {
            'head': (0, 255, 255),      # Yellow - head keypoints
            'upper_body': (0, 255, 0),  # Green - shoulders, arms
            'lower_body': (255, 0, 0),  # Blue - hips, legs
            'connections': (255, 255, 255)  # White - skeleton lines
        }
        
        # Performance tracking
        self.inference_times = []
        self.detection_history = []  # For anti-flickering
        self.smoothing_frames = 3
        
        # Test performance
        print("🧪 Testing pose inference speed...")
        test_times = []
        for i in range(5):
            start = time.time()
            self._complete_npu_compilation()
            test_time = (time.time() - start) * 1000
            test_times.append(test_time)
            print(f"   Test {i+1}: {test_time:.1f}ms")
        
        avg_test_time = sum(test_times) / len(test_times)
        expected_fps_actual = 1000 / avg_test_time
        
        print(f"✅ Average inference: {avg_test_time:.1f}ms ({expected_fps_actual:.1f} FPS)")
        print(f"🚀 YOLO Pose ready! NPU: {'Active' if self.using_npu else 'Inactive'}")

    def _analyze_model_outputs(self):
        """Analyze model outputs to determine pose format"""
        output_shapes = [detail['shape'] for detail in self.output_details]
        print(f"🔍 Analyzing pose model outputs: {output_shapes}")
        
        # Look for pose output patterns
        # YOLOv8/v11 pose typically outputs [1, 56, N] where 56 = 4(bbox) + 1(conf) + 51(17*3 keypoints)
        # Or [1, N, 56] depending on transpose
        
        pose_output_found = False
        for i, shape in enumerate(output_shapes):
            if len(shape) >= 2:
                # Check for pose pattern: should have 56 channels (4+1+51) for person+pose
                if 56 in shape or 51 in shape:
                    pose_output_found = True
                    self.pose_output_index = i
                    print(f"✓ Pose output detected at index {i}: {shape}")
                    break
        
        if not pose_output_found:
            print("⚠ Warning: Pose output pattern not clearly detected")
            print("   Assuming first output contains pose data")
            self.pose_output_index = 0
        
        self.is_pose_model = True

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

    def preprocess_image_adaptive(self, image):
        """Adaptive preprocessing for any input resolution"""
        img_height, img_width = image.shape[:2]
        target_width, target_height = self.input_size
        
        # Calculate scale and padding for letterboxing
        scale = min(target_width / img_width, target_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Choose interpolation based on scaling direction and input size
        if scale < 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        
        # For very small input sizes, prioritize speed
        if target_width <= 320 or target_height <= 320:
            interpolation = cv2.INTER_LINEAR
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # Create padded image with gray background (114) - YOLO standard
        padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_top = (target_height - new_height) // 2
        pad_left = (target_width - new_width) // 2
        
        # Place resized image in center
        padded[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Handle quantization based on model input type
        if self.input_details[0]['dtype'] == np.int8:
            normalized = rgb_image.astype(np.float32) / 255.0
            quantized = (normalized / self.input_scale) + self.input_zero_point
            quantized_clipped = np.clip(quantized, -128, 127).astype(np.int8)
            self.input_buffer[0] = quantized_clipped
        elif self.input_details[0]['dtype'] == np.uint8:
            normalized = rgb_image.astype(np.float32) / 255.0
            quantized = (normalized / self.input_scale) + self.input_zero_point
            quantized_clipped = np.clip(quantized, 0, 255).astype(np.uint8)
            self.input_buffer[0] = quantized_clipped
        else:
            self.input_buffer[0] = rgb_image.astype(np.float32) / 255.0
        
        return self.input_buffer, scale, pad_left, pad_top

    def detect_poses_multirez(self, image):
        """Multi-resolution YOLO Pose detection"""
        img_height, img_width = image.shape[:2]
        
        # Adaptive preprocessing
        input_data, scale, pad_left, pad_top = self.preprocess_image_adaptive(image)
        
        # Fast inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Track performance
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # Parse YOLO pose outputs
        poses = self._parse_pose_outputs(img_width, img_height, scale, pad_left, pad_top)
        
        # Apply temporal smoothing to reduce flickering
        stable_poses = self._stabilize_poses(poses)
        
        return stable_poses, inference_time

    def _parse_pose_outputs(self, img_width, img_height, scale, pad_left, pad_top):
        """Parse YOLO pose outputs for person detection + keypoints"""
        all_poses = []
        
        # Get the main pose output
        output_detail = self.output_details[self.pose_output_index]
        raw_output = self.interpreter.get_tensor(output_detail['index'])
        
        # Dequantize if needed
        if raw_output.dtype in [np.int8, np.uint8]:
            scale_quant = self.output_scales[self.pose_output_index]
            zero_point = self.output_zero_points[self.pose_output_index]
            output = scale_quant * (raw_output.astype(np.float32) - zero_point)
        else:
            output = raw_output.astype(np.float32)
        
        # Handle batch dimension and transpose if needed
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
            
            # Check if we need to transpose - pose models often need this
            if output.shape[0] < output.shape[1] and output.shape[0] >= 56:
                output = output.T  # Transpose to [N, 56]
        
        # YOLO pose output format: [N, 56] where 56 = 4(bbox) + 1(conf) + 51(17*3 keypoints)
        if len(output.shape) == 2 and output.shape[1] >= 56:
            poses = self._parse_pose_detections(
                output, img_width, img_height, scale, pad_left, pad_top
            )
            all_poses.extend(poses)
        
        # Apply NMS for pose detections
        return self._apply_pose_nms(all_poses)

    def _parse_pose_detections(self, output, img_width, img_height, scale, pad_left, pad_top):
        """Parse individual pose detections"""
        poses = []
        
        num_detections, num_attrs = output.shape
        
        if num_attrs < 56:
            print(f"⚠ Warning: Expected 56 attributes for pose, got {num_attrs}")
            return poses
        
        # YOLO pose format: [center_x, center_y, width, height, confidence, kp1_x, kp1_y, kp1_conf, ...]
        boxes = output[:, :4]  # [center_x, center_y, width, height]
        confidences = output[:, 4]  # Person detection confidence
        keypoints_raw = output[:, 5:]  # 51 values = 17 keypoints * 3 (x, y, confidence)
        
        # Apply confidence threshold
        conf_mask = confidences > self.confidence_threshold
        
        if not np.any(conf_mask):
            return poses
        
        # Extract filtered detections
        filtered_boxes = boxes[conf_mask]
        filtered_conf = confidences[conf_mask]
        filtered_keypoints = keypoints_raw[conf_mask]
        
        # Process each detection
        for box, conf, kp_raw in zip(filtered_boxes, filtered_conf, filtered_keypoints):
            center_x, center_y, width, height = box
            
            # Normalize coordinates if needed
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
            
            # Parse keypoints (17 keypoints * 3 values each)
            keypoints = []
            for i in range(17):
                kp_x = kp_raw[i * 3]
                kp_y = kp_raw[i * 3 + 1]
                kp_conf = kp_raw[i * 3 + 2]
                
                # Normalize keypoint coordinates if needed
                if kp_x <= 1.0 and kp_y <= 1.0:
                    kp_x *= self.input_size[0]
                    kp_y *= self.input_size[1]
                
                # Remove padding and scale back
                kp_x = (kp_x - pad_left) / scale
                kp_y = (kp_y - pad_top) / scale
                
                # Clamp to image boundaries
                kp_x = max(0, min(kp_x, img_width))
                kp_y = max(0, min(kp_y, img_height))
                
                keypoints.append({
                    'x': float(kp_x),
                    'y': float(kp_y),
                    'confidence': float(kp_conf),
                    'visible': kp_conf > 0.5  # Visibility threshold
                })
            
            # Only add valid poses
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:  # Minimum person size
                poses.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'keypoints': keypoints
                })
        
        return poses

    def _apply_pose_nms(self, poses):
        """Apply NMS to pose detections"""
        if len(poses) <= 1:
            return poses
        
        # Extract bounding boxes and confidences
        boxes = []
        scores = []
        for pose in poses:
            x1, y1, x2, y2 = pose['bbox']
            boxes.append([x1, y1, x2-x1, y2-y1])  # Convert to [x, y, w, h]
            scores.append(pose['confidence'])
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_threshold,
            0.45  # NMS threshold
        )
        
        if len(indices) > 0:
            return [poses[i] for i in indices.flatten()]
        return []

    def _stabilize_poses(self, current_poses):
        """Anti-flickering temporal smoothing for poses"""
        self.detection_history.append(current_poses)
        if len(self.detection_history) > self.smoothing_frames:
            self.detection_history.pop(0)
        
        # Simple temporal filtering - only return poses seen in multiple frames
        stable_poses = []
        for pose in current_poses:
            confidence_votes = 1
            for past_poses in self.detection_history[:-1]:
                for past_pose in past_poses:
                    if self._poses_overlap(pose, past_pose):
                        confidence_votes += 1
                        break
            
            # Keep poses seen in at least 2 frames
            if confidence_votes >= 2:
                stable_poses.append(pose)
        
        return stable_poses

    def _poses_overlap(self, pose1, pose2, threshold=0.5):
        """Check if two pose detections overlap significantly"""
        x1, y1, x2, y2 = pose1['bbox']
        x1_p, y1_p, x2_p, y2_p = pose2['bbox']
        
        # Calculate IoU
        intersection = max(0, min(x2, x2_p) - max(x1, x1_p)) * \
                      max(0, min(y2, y2_p) - max(y1, y1_p))
        union = (x2-x1)*(y2-y1) + (x2_p-x1_p)*(y2_p-y1_p) - intersection
        
        return intersection / union > threshold if union > 0 else False

    def draw_poses(self, image, poses):
        """Draw poses with skeleton and keypoints on image"""
        for pose in poses:
            bbox = pose['bbox']
            confidence = pose['confidence']
            keypoints = pose['keypoints']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f'Person: {confidence:.2f}'
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw skeleton connections
            for connection in self.skeleton_connections:
                kp1_idx, kp2_idx = connection
                kp1 = keypoints[kp1_idx]
                kp2 = keypoints[kp2_idx]
                
                # Only draw if both keypoints are visible and confident
                if kp1['visible'] and kp2['visible'] and kp1['confidence'] > 0.3 and kp2['confidence'] > 0.3:
                    pt1 = (int(kp1['x']), int(kp1['y']))
                    pt2 = (int(kp2['x']), int(kp2['y']))
                    cv2.line(image, pt1, pt2, self.pose_colors['connections'], 2)
            
            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                if keypoint['visible'] and keypoint['confidence'] > 0.3:
                    x, y = int(keypoint['x']), int(keypoint['y'])
                    
                    # Color keypoints by body part
                    if i <= 4:  # Head keypoints
                        color = self.pose_colors['head']
                    elif i <= 10:  # Upper body
                        color = self.pose_colors['upper_body']
                    else:  # Lower body
                        color = self.pose_colors['lower_body']
                    
                    # Draw keypoint
                    cv2.circle(image, (x, y), 4, color, -1)
                    cv2.circle(image, (x, y), 6, (255, 255, 255), 1)  # White border
        
        return image

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
            'samples': len(self.inference_times),
            'input_resolution': f"{self.input_size[0]}x{self.input_size[1]}"
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
    parser = argparse.ArgumentParser(description='Multi-Resolution High-Performance YOLO Pose Detection')
    parser.add_argument('--model', required=True, help='Path to YOLO pose .tflite model (yolov8n-pose or yolov11n-pose)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--no_npu', action='store_true', help='Disable NPU acceleration')
    parser.add_argument('--no_display', action='store_true', help='Disable display for max performance')
    parser.add_argument('--input-size', nargs='+', type=int, help='Force input size: --input-size 320 or --input-size 416 416')
    parser.add_argument('--show-keypoints', action='store_true', default=True, help='Show individual keypoint names')
    parser.add_argument('--min-keypoint-conf', type=float, default=0.3, help='Minimum keypoint confidence to display')
    
    args = parser.parse_args()
    
    print("Multi-Resolution High-Performance YOLO Pose Detection")
    print("=" * 60)
    
    # Parse input size argument
    force_input_size = None
    if args.input_size:
        if len(args.input_size) == 1:
            force_input_size = args.input_size[0]  # Square: 320 -> 320x320
        elif len(args.input_size) == 2:
            force_input_size = tuple(args.input_size)  # Rectangle: 416 416 -> 416x416
        else:
            print("❌ Error: --input-size accepts 1 or 2 values (e.g., --input-size 320 or --input-size 416 416)")
            sys.exit(1)
    
    # Initialize pose detector (one-time compilation)
    detector = MultiResolutionYOLOPoseDetector(
        args.model, 
        args.threshold, 
        use_npu=not args.no_npu,
        force_input_size=force_input_size
    )
    
    # Initialize camera
    videostream = FastVideoStream(camera_index=args.camera).start()
    time.sleep(1)
    
    print(f"\n🎬 Starting multi-resolution YOLO pose detection...")
    print(f"Input resolution: {detector.input_size[0]}x{detector.input_size[1]}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Keypoint confidence threshold: {args.min_keypoint_conf}")
    print("Press 'q' to quit, 's' for performance stats, 'k' to toggle keypoint names")
    
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    show_keypoint_names = args.show_keypoints
    
    try:
        while True:
            frame = videostream.read()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Multi-resolution YOLO pose detection
            poses, inference_time = detector.detect_poses_multirez(frame)
            
            # Calculate and display FPS every 30 frames
            fps_counter += 1
            if fps_counter >= 30:
                elapsed = time.time() - fps_start
                current_fps = 30 / elapsed
                fps_start = time.time()
                fps_counter = 0
                print(f"🚀 FPS: {current_fps:.1f} | Inference: {inference_time:.1f}ms | Poses: {len(poses)} | Resolution: {detector.input_size[0]}x{detector.input_size[1]}")
            
            if not args.no_display:
                # Draw poses on frame
                detector.draw_poses(frame, poses)
                
                # Optionally draw keypoint names
                if show_keypoint_names:
                    for pose in poses:
                        for i, keypoint in enumerate(pose['keypoints']):
                            if keypoint['visible'] and keypoint['confidence'] > args.min_keypoint_conf:
                                x, y = int(keypoint['x']), int(keypoint['y'])
                                kp_name = detector.keypoint_names[i]
                                cv2.putText(frame, kp_name, (x + 8, y - 8), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                
                # Performance overlay
                cv2.putText(frame, f'YOLO-Pose-{detector.input_size[0]}: {inference_time:.1f}ms', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'NPU: {"ON" if detector.using_npu else "OFF"}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detector.using_npu else (0, 0, 255), 2)
                cv2.putText(frame, f'Poses: {len(poses)}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Legend for pose colors
                cv2.putText(frame, 'Head', (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, detector.pose_colors['head'], 2)
                cv2.putText(frame, 'Upper Body', (frame.shape[1] - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, detector.pose_colors['upper_body'], 2)
                cv2.putText(frame, 'Lower Body', (frame.shape[1] - 150, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, detector.pose_colors['lower_body'], 2)
                
                cv2.imshow('Multi-Resolution YOLO Pose Detection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = detector.get_performance_stats()
                    if stats:
                        print(f"\n📊 YOLO Pose Performance Stats ({stats['input_resolution']}):")
                        print(f"   Average: {stats['avg_inference_ms']:.1f}ms ({stats['avg_fps']:.1f} FPS)")
                        print(f"   Range: {stats['min_inference_ms']:.1f} - {stats['max_inference_ms']:.1f}ms")
                elif key == ord('k'):
                    show_keypoint_names = not show_keypoint_names
                    print(f"Keypoint names: {'ON' if show_keypoint_names else 'OFF'}")
    
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
            print(f"\n🏁 YOLO Pose Final Results:")
            print(f"   Input resolution: {final_stats['input_resolution']}")
            print(f"   Average inference: {final_stats['avg_inference_ms']:.1f}ms")
            print(f"   Theoretical FPS: {final_stats['avg_fps']:.1f}")
            print(f"   NPU: {'Active' if detector.using_npu else 'Inactive'}")
            
            # Performance tier summary
            avg_fps = final_stats['avg_fps']
            if avg_fps >= 12:
                tier = "🚀 ULTRA-FAST"
            elif avg_fps >= 8:
                tier = "⚡ FAST"
            elif avg_fps >= 4:
                tier = "🏃 MODERATE"
            else:
                tier = "🐌 ACCURATE"
            print(f"   Performance tier: {tier}")
            print(f"   Model type: YOLO Pose Detection")

if __name__ == '__main__':
    main()
