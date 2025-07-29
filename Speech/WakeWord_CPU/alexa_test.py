import wave
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




model_path="/home/scmd/tts/wakeword/alexa_v0.1.tflite"

interpreter = tflite.Interpreter(model_path=model_path, num_threads=1)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"✓ Alexa model loaded: {model_path}")
print(f"✓ Input shape: {input_details[0]['shape']}")  
print(f"✓ Output shape: {output_details[0]['shape']}")
print(f"✓ Ready to test 'Alexa' wake word detection")

def load_wav_file(filename):
    with wave.open(filename, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        return audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
# Line 7: Function to create simple audio features (basic preprocessing)
def create_features(audio_chunk):
    # Simple feature extraction - reshape audio to match model input [1, 16, 96]
    if len(audio_chunk) < 1536:  # 16 * 96 = 1536 samples needed
        audio_chunk = np.pad(audio_chunk, (0, 1536 - len(audio_chunk)))
    return audio_chunk[:1536].reshape(1, 16, 96)
# Line 8: Function to run inference on audio features
def predict_alexa(features):
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]



audio_path="/home/scmd/alexa_test.wav"
audio_data = load_wav_file(audio_path)

audio_data = load_wav_file(audio_path)
print(f"✓ Audio loaded: {len(audio_data)} samples")
chunk_size = 1536  # 16 * 96 = 1536 samples per prediction
step_size = 160    # Step by 10ms (160 samples at 16kHz) for sliding window
predictions = []
# Line 12: Run detection on overlapping audio chunks
for i in range(0, len(audio_data) - chunk_size, step_size):
    audio_chunk = audio_data[i:i + chunk_size]
    features = create_features(audio_chunk)
    score = predict_alexa(features)
    predictions.append(score)
    
# Line 13: Find the maximum confidence score and its timing
max_score = max(predictions)
max_index = predictions.index(max_score)
time_detected = (max_index * step_size) / 16000.0 
chunk_size = 1536
step_size = 160
predictions = []
for i in range(0, len(audio_data) - chunk_size, step_size):
    audio_chunk = audio_data[i:i + chunk_size]
    features = create_features(audio_chunk)
    score = predict_alexa(features)
    predictions.append(score)
predictions
max_score = max(predictions)
max_index = predictions.index(max_score)
time_detected = (max_index * step_size) / 16000.0 
threshold = 0.5
print(f"📊 Max confidence score: {max_score:.4f}")
print(f"⏰ Detected at time: {time_detected:.2f} seconds")
max_score
threshold
predictions_array = np.array(predictions)
mean_score = predictions_array.mean()
std_score = predictions_array.std()
print(f"\n📈 SCORE ANALYSIS:")
print(f"   Mean score: {mean_score:.4f}")
print(f"   Std deviation: {std_score:.4f}")
print(f"   Score range: {predictions_array.min():.4f} - {predictions_array.max():.4f}")
sorted_indices = np.argsort(predictions)[-5:][::-1]  # Top 5 indices
print(f"\n🔝 TOP 5 DETECTION PEAKS:")
for i, idx in enumerate(sorted_indices):
    score = predictions[idx]
    time_pos = (idx * step_size) / 16000.0
    print(f"   {i+1}. Score: {score:.4f} at {time_pos:.2f}s")
suggested_threshold = mean_score + (2 * std_score)
print(f"\n🎯 THRESHOLD SUGGESTIONS:")
print(f"   Current threshold: {threshold}")
print(f"   Suggested threshold: {suggested_threshold:.4f} (mean + 2*std)")
print(f"   Conservative threshold: {max_score - 0.01:.4f}")
print(f"\n📋 SUMMARY:")
print(f"   Audio duration: {len(audio_data)/16000:.2f} seconds")
print(f"   Processed {len(predictions)} audio chunks")
print(f"   Peak detection at: {time_detected:.2f}s")
