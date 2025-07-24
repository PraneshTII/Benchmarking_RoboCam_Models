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
import wave
from pathlib import Path
import struct

model_path="/home/scmd/whisper-tiny-en.tflite"
vocab_mel_path="/home/scmd/tflt-vocab-mel.bin"
audio_file="/home/scmd/input.wav"


interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model loaded. Input shape: {input_details[0]['shape']}")


def load_vocab_mel(filepath):
    with open(filepath, 'rb') as f:
        # Read magic number
        magic = struct.unpack('i', f.read(4))[0]
        assert magic == 0x74666C74, "Invalid magic number"

        # Read mel filters
        mel_rows = struct.unpack('i', f.read(4))[0]
        mel_cols = struct.unpack('i', f.read(4))[0]
        mel_filters = np.zeros((mel_rows, mel_cols), dtype=np.float32)

        for i in range(mel_rows):
            for j in range(mel_cols):
                mel_filters[i][j] = struct.unpack('f', f.read(4))[0]

        # Read tokenizer
        vocab_size = struct.unpack('i', f.read(4))[0]
        tokens = {}

        for _ in range(vocab_size):
            token_len = struct.unpack('i', f.read(4))[0]
            token_bytes = f.read(token_len)
            tokens[token_bytes] = len(tokens)

        return mel_filters, tokens

# Load preprocessing data
mel_filters, token_vocab = load_vocab_mel(vocab_mel_path)
print(f"Loaded {len(token_vocab)} tokens and mel filters shape: {mel_filters.shape}")

# Create reverse token mapping for decoding
id_to_token = {v: k for k, v in token_vocab.items()}

def load_wav(wav_path):
    with wave.open(wav_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        # Convert to numpy array
        if sample_width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Handle stereo to mono
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        return audio, sample_rate



print(f"Loading WAV audio: {audio_file}")
audio, sr = load_wav(audio_file)
print(f"Loaded audio: sample rate {sr}Hz, length {len(audio)} samples")

# Ensure 16kHz (should already be from ffmpeg conversion)
if sr != 16000:
    print(f"Warning: Expected 16kHz, got {sr}Hz")

# Pad or truncate to 30 seconds (480,000 samples at 16kHz)
target_length = 16000 * 30
if len(audio) > target_length:
    audio = audio[:target_length]
else:
    audio = np.pad(audio, (0, target_length - len(audio)))

print(f"Audio preprocessed: {len(audio)} samples, {len(audio)/16000:.1f} seconds")




def stft(audio, n_fft=400, hop_length=160):
    # Hann window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / (n_fft - 1)))

    # Pad audio
    pad_length = n_fft // 2
    audio_padded = np.pad(audio, pad_length, mode='reflect')

    # Calculate number of frames
    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length

    # Initialize output
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)

    # Compute STFT
    for i in range(n_frames):
        start = i * hop_length
        frame = audio_padded[start:start + n_fft] * window

        # FFT
        fft_frame = np.fft.fft(frame)
        stft_matrix[:, i] = fft_frame[:n_fft // 2 + 1]

    return stft_matrix





print("Computing mel spectrogram...")
n_fft = 400  # 25ms window at 16kHz
hop_length = 160  # 10ms hop at 16kHz

# Compute STFT
stft_result = stft(audio, n_fft=n_fft, hop_length=hop_length)
magnitude = np.abs(stft_result)

magnitude = np.abs(stft_result)

# Apply mel filter bank
mel_spec = np.dot(mel_filters, magnitude)

# Convert to log mel spectrogram
log_mel = np.log10(np.maximum(mel_spec, 1e-10))

# Normalize (Whisper specific normalization)
log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

# Ensure exactly 3000 frames
if log_mel.shape[1] > 3000:
    log_mel = log_mel[:, :3000]
else:
    log_mel = np.pad(log_mel, ((0, 0), (0, 3000 - log_mel.shape[1])))

# Reshape for model input [1, 80, 3000]
mel_input = log_mel.reshape(1, 80, 3000).astype(np.float32)
print(f"Mel spectrogram shape: {mel_input.shape}")

# Run inference
print("Running inference...")
interpreter.set_tensor(input_details[0]['index'], mel_input)
interpreter.invoke()
output_tokens = interpreter.get_tensor(output_details[0]['index'])

print(f"Output tokens shape: {output_tokens.shape}")

def decode_tokens(token_ids, id_to_token):
    text_parts = []
    for token_id in token_ids[0]:  # Remove batch dimension
        if token_id == 0:  # End of sequence or padding
            break
        if token_id in id_to_token:
            token_bytes = id_to_token[token_id]
            try:
                # Try to decode as UTF-8
                text_parts.append(token_bytes.decode('utf-8', errors='ignore'))
            except:
                # Skip problematic tokens
                continue

    return ''.join(text_parts)

# Get predicted token IDs (assuming logits output, take argmax)
if output_tokens.dtype == np.float32:
    predicted_ids = np.argmax(output_tokens, axis=-1)
else:
    predicted_ids = output_tokens

transcription = decode_tokens(predicted_ids, id_to_token)

print("\n" + "="*50)
print("TRANSCRIPTION:")
print("="*50)
print(transcription)
print("="*50)

# Save transcription to file
output_file = audio_file.replace('.wav', '_transcription.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(transcription)

print(f"\nTranscription saved to: {output_file}")
