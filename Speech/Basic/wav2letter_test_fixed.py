# -*- coding: utf-8 -*-
"""
Modified Wav2Letter Inference with NumPy/SciPy only
No librosa dependency - all audio processing done with NumPy/SciPy
TensorFlow only used for TFLite interpreter
"""

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


# Audio file and ground truth
audio_file = 'audio_1m.wav'
transcript = 'Wow what an audience But if I am being honest I dont care what you think of my talk I dont I care what the Internet thinks of my talk Because they are the ones who get it seen and get it shared And I think thats where most people get it wrong They are talking to you here instead of talking to you random person scrolling Facebook Thanks for the click You see back in two thousand ninteen we all had these weird little things called attention spans Yeah they are gone We killed them They are dead I am trying to think of the last time I watched an eighteen minute TED talk It is been years literally years So if you are giving a TED talk keep it quick I am doing mine in under a minute I am at forty four seconds right now That means we have got time for one final joke  Why are balloons so expensive Why Inflation Thanks'

# Alphabet setup
alphabet = "abcdefghijklmnopqrstuvwxyz' @"
alphabet_dict = {c: ind for (ind, c) in enumerate(alphabet)}
index_dict = {ind: c for (ind, c) in enumerate(alphabet)}
transcript_ints = [alphabet_dict[letter] for letter in transcript.lower()]
print(f"DEBUG: Transcript converted to integers: {transcript_ints[:50]}...")

def normalize(values):
    """Normalize values to mean 0 and std 1"""
    print(f"DEBUG: Normalizing values with shape {values.shape}")
    print(f"DEBUG: Original mean: {np.mean(values):.6f}, std: {np.std(values):.6f}")
    normalized = (values - np.mean(values)) / np.std(values)
    print(f"DEBUG: Normalized mean: {np.mean(normalized):.6f}, std: {np.std(normalized):.6f}")
    return normalized


def load_audio_scipy(audio_file, target_sr=16000):
    """Load audio file using scipy and resample to target sampling rate"""
    print(f"DEBUG: Loading audio file with scipy: {audio_file}")
    
    sample_rate, audio_data = wavfile.read(audio_file)
    print(f"DEBUG: Original audio - sample_rate: {sample_rate}, shape: {audio_data.shape}, dtype: {audio_data.dtype}")
    
    # Convert to float32 and normalize to [-1, 1] range (match librosa exactly)
    if audio_data.dtype == np.int16:
        print("DEBUG: Converting int16 to float32")
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        print("DEBUG: Converting int32 to float32") 
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        print("DEBUG: Converting uint8 to float32")
        audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
    else:
        print(f"DEBUG: Audio already in format: {audio_data.dtype}")
        audio_data = audio_data.astype(np.float32)
    
    # Convert stereo to mono if needed (match librosa's method - FIXED)
    if len(audio_data.shape) > 1:
        print(f"DEBUG: Converting stereo ({audio_data.shape[1]} channels) to mono")
        audio_data = np.mean(audio_data, axis=1).astype(np.float32)  # ✅ FIXED: Ensure float32
    
    print(f"DEBUG: Audio after conversion - shape: {audio_data.shape}, dtype: {audio_data.dtype}")
    print(f"DEBUG: Audio range: [{np.min(audio_data):.6f}, {np.max(audio_data):.6f}]")
    
    # Resample if needed - keeping the simpler method that worked better
    if sample_rate != target_sr:
        print(f"DEBUG: Resampling from {sample_rate} Hz to {target_sr} Hz")
        new_length = int(len(audio_data) * target_sr / sample_rate)
        old_indices = np.linspace(0, len(audio_data) - 1, len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        audio_data = np.interp(new_indices, old_indices, audio_data).astype(np.float32)  # ✅ FIXED: Ensure float32
        sample_rate = target_sr
        print(f"DEBUG: Resampled audio shape: {audio_data.shape}")
    
    print(f"DEBUG: Final audio - length: {len(audio_data)}, sample_rate: {sample_rate}")
    print(f"DEBUG: Final first 10 samples: {audio_data[:10]}")  
    return audio_data, sample_rate



def load_audio_scipy_old(audio_file, target_sr=16000):
    """Load audio file using scipy and resample to target sampling rate"""
    print(f"DEBUG: Loading audio file with scipy: {audio_file}")
    
    sample_rate, audio_data = wavfile.read(audio_file)
    print(f"DEBUG: Original audio - sample_rate: {sample_rate}, shape: {audio_data.shape}, dtype: {audio_data.dtype}")
    
    # Convert to float32 and normalize to [-1, 1] range
    if audio_data.dtype == np.int16:
        print("DEBUG: Converting int16 to float32")
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        print("DEBUG: Converting int32 to float32") 
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        print("DEBUG: Converting uint8 to float32")
        audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
    else:
        print(f"DEBUG: Audio already in format: {audio_data.dtype}")
        audio_data = audio_data.astype(np.float32)
    
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        print(f"DEBUG: Converting stereo ({audio_data.shape[1]} channels) to mono")
        audio_data = np.mean(audio_data, axis=1)
    
    print(f"DEBUG: Audio after conversion - shape: {audio_data.shape}, dtype: {audio_data.dtype}")
    print(f"DEBUG: Audio range: [{np.min(audio_data):.6f}, {np.max(audio_data):.6f}]")
    
    # Resample if needed (simple linear interpolation)
    if sample_rate != target_sr:
        print(f"DEBUG: Resampling from {sample_rate} Hz to {target_sr} Hz")
        new_length = int(len(audio_data) * target_sr / sample_rate)
        old_indices = np.linspace(0, len(audio_data) - 1, len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)
        audio_data = np.interp(new_indices, old_indices, audio_data)
        sample_rate = target_sr
        print(f"DEBUG: Resampled audio shape: {audio_data.shape}")
    
    print(f"DEBUG: Final audio - length: {len(audio_data)}, sample_rate: {sample_rate}")
    return audio_data, sample_rate

def stft_librosa_equivalent(y, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant'):
    """
    STFT implementation that matches librosa's behavior exactly
    """
    print(f"DEBUG: STFT - n_fft: {n_fft}, hop_length: {hop_length}, center: {center}")
    
    if win_length is None:
        win_length = n_fft
    
    if hop_length is None:
        hop_length = int(win_length // 4)
    
    # Center padding if enabled (matches librosa default)
    if center:
        padding = [(n_fft // 2, n_fft // 2)]
        if pad_mode == 'constant':
            y = np.pad(y, padding, mode='constant', constant_values=0)
        elif pad_mode == 'reflect':
            y = np.pad(y, padding, mode='reflect')
        print(f"DEBUG: After center padding: {y.shape}")
    
    # Calculate number of frames
    n_frames = 1 + (len(y) - n_fft) // hop_length
    print(f"DEBUG: Number of frames: {n_frames}")
    
    # Create window
    if isinstance(window, str):
        if window == 'hann':
            fft_window = np.hanning(win_length)
        else:
            fft_window = get_window(window, win_length)
    else:
        fft_window = window
    
    # Pad window to n_fft size (center padding)
    if len(fft_window) < n_fft:
        pad_amount = n_fft - len(fft_window)
        fft_window = np.pad(fft_window, (pad_amount // 2, pad_amount - pad_amount // 2), mode='constant')
    
    print(f"DEBUG: Window shape: {fft_window.shape}")
    
    # Initialize STFT matrix
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    
    # Compute STFT frame by frame
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        
        if end <= len(y):
            # Extract frame and apply window
            frame = y[start:end] * fft_window
            
            # Compute FFT
            fft_frame = np.fft.rfft(frame, n=n_fft)
            stft_matrix[:, i] = fft_frame
    
    print(f"DEBUG: STFT matrix shape: {stft_matrix.shape}")
    return stft_matrix

def mel_scale_slaney(frequencies):
    """Convert frequency in Hz to mel scale using Slaney formula (librosa default)"""
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)

def inverse_mel_scale_slaney(mels):
    """Convert mel scale to frequency in Hz using Slaney formula"""
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def create_mel_filterbank_librosa(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, norm='slaney'):
    """
    Create mel filterbank matrix that exactly matches librosa implementation
    """
    print(f"DEBUG: Creating mel filterbank - sr: {sr}, n_fft: {n_fft}, n_mels: {n_mels}, norm: {norm}")
    
    if fmax is None:
        fmax = float(sr) / 2
    
    print(f"DEBUG: Frequency range: {fmin} Hz to {fmax} Hz")
    
    # Number of FFT bins
    n_freqs = int(1 + n_fft // 2)
    
    # Create frequency array for FFT bins  
    fftfreqs = np.linspace(0, float(sr) / 2, n_freqs)
    print(f"DEBUG: FFT frequencies shape: {fftfreqs.shape}")
    
    # Convert to mel scale
    mel_f_min = mel_scale_slaney(fmin)
    mel_f_max = mel_scale_slaney(fmax)
    print(f"DEBUG: Mel range: {mel_f_min:.2f} to {mel_f_max:.2f}")
    
    # Create equally spaced mel points
    mels = np.linspace(mel_f_min, mel_f_max, n_mels + 2)
    
    # Convert back to Hz
    mel_freqs = inverse_mel_scale_slaney(mels)
    print(f"DEBUG: Mel filter center frequencies: {mel_freqs[:5]}...")
    
    # Create filterbank
    filterbank = np.zeros((n_mels, n_freqs))
    
    for i in range(n_mels):
        # Define the triangular filter
        left = mel_freqs[i]
        center = mel_freqs[i + 1] 
        right = mel_freqs[i + 2]
        
        # Create triangular filter
        for j, freq in enumerate(fftfreqs):
            if left <= freq <= center and center != left:
                filterbank[i, j] = (freq - left) / (center - left)
            elif center <= freq <= right and right != center:
                filterbank[i, j] = (right - freq) / (right - center)
        
        # Apply normalization
        if norm == 'slaney':
            # Slaney-style normalization: divide by width of mel band
            enorm = 2.0 / (mel_freqs[i + 2] - mel_freqs[i])
            filterbank[i] *= enorm
    
    print(f"DEBUG: Filterbank shape: {filterbank.shape}")
    print(f"DEBUG: Filterbank max values per filter (first 5): {np.max(filterbank[:5], axis=1)}")
    
    return filterbank

def power_to_db_numpy(S, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Convert power spectrogram to dB scale (matches librosa.power_to_db)
    """
    print(f"DEBUG: Converting power to dB - ref: {ref}, amin: {amin}, top_db: {top_db}")
    
    S = np.asarray(S)
    
    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = np.abs(ref)
    
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    
    print(f"DEBUG: DB conversion - input range: [{np.min(S):.6e}, {np.max(S):.6e}]")
    print(f"DEBUG: DB conversion - output range: [{np.min(log_spec):.6f}, {np.max(log_spec):.6f}]")
    
    return log_spec

def melspectrogram_numpy(y, sr=22050, n_fft=2048, hop_length=512, win_length=None, 
                        window='hann', center=True, pad_mode='constant', power=2.0, 
                        n_mels=128, fmin=0.0, fmax=None, norm='slaney'):
    """
    Compute mel spectrogram using numpy (matches librosa.feature.melspectrogram)
    """
    print(f"DEBUG: Computing mel spectrogram - n_fft: {n_fft}, hop_length: {hop_length}, power: {power}")
    
    # Compute STFT
    stft_matrix = stft_librosa_equivalent(y, n_fft=n_fft, hop_length=hop_length, 
                                         win_length=win_length, window=window, 
                                         center=center, pad_mode=pad_mode)
    
    # Calculate power/magnitude spectrogram
    if power == 1.0:
        # Magnitude spectrogram
        S = np.abs(stft_matrix)
    elif power == 2.0:
        # Power spectrogram
        S = np.abs(stft_matrix) ** 2
    else:
        S = np.abs(stft_matrix) ** power
    
    print(f"DEBUG: Power spectrogram shape: {S.shape}")
    
    # Create mel filterbank
    mel_basis = create_mel_filterbank_librosa(sr, n_fft, n_mels, fmin, fmax, norm)
    
    # Apply mel filterbank
    mel_spec = np.dot(mel_basis, S)
    print(f"DEBUG: Mel spectrogram shape: {mel_spec.shape}")
    
    return mel_spec

def mfcc_numpy(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', **kwargs):
    """
    Compute MFCC features using numpy (matches librosa.feature.mfcc)
    """
    print(f"DEBUG: Computing MFCC - n_mfcc: {n_mfcc}, dct_type: {dct_type}, norm: {norm}")
    
    if S is None:
        # Compute mel spectrogram
        mel_spec = melspectrogram_numpy(y=y, sr=sr, **kwargs)
        # Convert to dB
        S = power_to_db_numpy(mel_spec, ref=1.0, amin=1e-10, top_db=80.0)
    
    print(f"DEBUG: DB mel spectrogram shape: {S.shape}")
    print(f"DEBUG: DB mel spectrogram range: [{np.min(S):.6f}, {np.max(S):.6f}]")
    
    # Apply DCT
    M = dct(S, type=dct_type, axis=0, norm=norm)[:n_mfcc]
    
    print(f"DEBUG: Final MFCC shape: {M.shape}")
    print(f"DEBUG: MFCC range: [{np.min(M):.6f}, {np.max(M):.6f}]")
    
    return M

def delta_numpy(data, width=9, order=1, axis=-1, mode='interp', **kwargs):
    """
    Compute delta features that match librosa.feature.delta exactly
    Uses scipy.signal.savgol_filter with deriv=order
    """
    print(f"DEBUG: Computing delta features - order: {order}, width: {width}, mode: {mode}")
    print(f"DEBUG: Input data shape: {data.shape}")
    
    data = np.atleast_1d(data)
    
    # Validate parameters (same as librosa)
    if mode == "interp" and width > data.shape[axis]:
        raise ValueError(
            f"when mode='interp', width={width} "
            f"cannot exceed data.shape[axis]={data.shape[axis]}"
        )
    if width < 3 or np.mod(width, 2) != 1:
        raise ValueError("width must be an odd integer >= 3")
    if order <= 0 or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be a positive integer")
    
    # Remove 'deriv' from kwargs if present and set polyorder
    kwargs.pop("deriv", None)
    kwargs.setdefault("polyorder", order)
    
    print(f"DEBUG: Using savgol_filter with width={width}, deriv={order}, polyorder={kwargs['polyorder']}, mode={mode}")
    
    # Import here to match librosa structure
    from scipy.signal import savgol_filter
    
    # Apply Savitzky-Golay filter with derivative
    result = savgol_filter(data, width, deriv=order, axis=axis, mode=mode, **kwargs)
    
    print(f"DEBUG: Delta features shape: {result.shape}")
    print(f"DEBUG: Delta features range: [{np.min(result):.6f}, {np.max(result):.6f}]")
    
    return result

def transform_audio_to_mfcc(audio_file, transcript, n_mfcc=13, n_fft=512, hop_length=160):
    """Transform audio to MFCC features using numpy equivalents"""
    print(f"DEBUG: Starting MFCC transformation")
    
    # Load audio
    audio_data, sample_rate = load_audio_scipy(audio_file, target_sr=16000)
    
    # Compute MFCC features using exact librosa parameters
    print("DEBUG: Computing base MFCC features")
    mfcc = mfcc_numpy(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    print("DEBUG: Computing delta features")
    mfcc_delta = delta_numpy(mfcc, width=9, order=1, axis=1)
    
    print("DEBUG: Computing delta-delta features") 
    mfcc_delta2 = delta_numpy(mfcc, width=9, order=2, axis=1)
    
    print("DEBUG: Normalizing and concatenating MFCC features")
    mfcc_normalized = normalize(mfcc)
    mfcc_delta_normalized = normalize(mfcc_delta) 
    mfcc_delta2_normalized = normalize(mfcc_delta2)
    
    mfcc_combined = np.concatenate((mfcc_normalized, mfcc_delta_normalized, mfcc_delta2_normalized), axis=0)
    print(f"DEBUG: Combined MFCC features shape: {mfcc_combined.shape}")
    
    seq_length = mfcc_combined.shape[1] // 2
    print(f"DEBUG: Sequence length: {seq_length}")
    
    sequences = np.concatenate([[seq_length], transcript]).astype(np.int32)
    sequences = np.expand_dims(sequences, 0)
    print(f"DEBUG: Sequences shape: {sequences.shape}")
    
    mfcc_out = mfcc_combined.T.astype(np.float32)
    mfcc_out = np.expand_dims(mfcc_out, 0)
    print(f"DEBUG: Final MFCC output shape: {mfcc_out.shape}")
    
    return mfcc_out, sequences

def log(std):
    """Log the given string to the standard output."""
    print("******* {}".format(std), flush=True)

# NumPy implementation of CTC greedy decoder
def numpy_ctc_greedy_decoder(logits, sequence_lengths, blank_label=28, merge_repeated=True):
    """Pure NumPy implementation of CTC greedy decoder"""
    print(f"DEBUG: CTC decoder input logits shape: {logits.shape}")
    print(f"DEBUG: CTC decoder sequence lengths: {sequence_lengths}")
    print(f"DEBUG: Using blank label: {blank_label}")
    print(f"DEBUG: Logits min: {np.min(logits):.6f}, max: {np.max(logits):.6f}")
    print(f"DEBUG: Logits mean: {np.mean(logits):.6f}, std: {np.std(logits):.6f}")
    
    # Get the most likely class at each timestep
    max_indices = np.argmax(logits, axis=2).T  # (batch_size, time_steps)
    print(f"DEBUG: Max indices shape: {max_indices.shape}")
    print(f"DEBUG: Max indices min: {np.min(max_indices)}, max: {np.max(max_indices)}")
    print(f"DEBUG: Max indices first 10 values: {max_indices.flatten()[:10]}")
    
    # Check for invalid indices
    num_classes = logits.shape[2] if len(logits.shape) > 2 else logits.shape[1]
    invalid_indices = max_indices >= num_classes
    if np.any(invalid_indices):
        print(f"WARNING: Found {np.sum(invalid_indices)} invalid indices >= {num_classes}")
        print(f"DEBUG: Invalid indices sample: {max_indices[invalid_indices][:10]}")
        # Clip invalid indices to blank label
        max_indices = np.clip(max_indices, 0, num_classes - 1)
        print(f"DEBUG: After clipping - max indices max: {np.max(max_indices)}")
    
    decoded_sequences = []
    for batch_idx, sequence in enumerate(max_indices):
        print(f"DEBUG: Processing batch {batch_idx}")
        seq_len = sequence_lengths[batch_idx] if batch_idx < len(sequence_lengths) else len(sequence)
        sequence = sequence[:seq_len]  # Trim to actual sequence length
        
        print(f"DEBUG: Batch {batch_idx} sequence length: {seq_len}")
        print(f"DEBUG: Batch {batch_idx} sequence min: {np.min(sequence)}, max: {np.max(sequence)}")
        
        decoded = []
        prev_token = None
        
        for token in sequence:
            if token != blank_label:  # Not blank
                if not merge_repeated or token != prev_token:
                    # Additional safety check
                    if 0 <= token < num_classes:
                        decoded.append(int(token))
                    else:
                        print(f"WARNING: Skipping invalid token {token}")
            prev_token = token
        
        print(f"DEBUG: Batch {batch_idx} decoded sequence length: {len(decoded)}")
        if len(decoded) > 0:
            print(f"DEBUG: Batch {batch_idx} decoded min: {min(decoded)}, max: {max(decoded)}")
        decoded_sequences.append(np.array(decoded))
    
    print(f"DEBUG: Total decoded sequences: {len(decoded_sequences)}")
    return decoded_sequences

# NumPy implementation of edit distance (Levenshtein distance)
def numpy_edit_distance(seq1, seq2):
    """Calculate edit distance between two sequences using NumPy"""
    print(f"DEBUG: Computing edit distance between sequences of length {len(seq1)} and {len(seq2)}")
    
    m, n = len(seq1), len(seq2)
    
    # Create distance matrix
    dist_matrix = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize first row and column
    dist_matrix[0, :] = np.arange(n + 1)
    dist_matrix[:, 0] = np.arange(m + 1)
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dist_matrix[i, j] = min(
                dist_matrix[i-1, j] + 1,      # deletion
                dist_matrix[i, j-1] + 1,      # insertion
                dist_matrix[i-1, j-1] + cost  # substitution
            )
    
    edit_dist = dist_matrix[m, n]
    print(f"DEBUG: Edit distance: {edit_dist}")
    return edit_dist

def ctc_preparation_numpy(tensor, y_predict):
    """Prepare inputs for CTC decoder - NumPy version"""
    print(f"DEBUG: CTC preparation - tensor shape: {tensor.shape}")
    print(f"DEBUG: CTC preparation - y_predict shape: {y_predict.shape}")
    
    if len(y_predict.shape) == 4:
        y_predict = np.squeeze(y_predict, axis=1)
        print(f"DEBUG: After squeeze: {y_predict.shape}")
    
    # Check the correct sequence length from actual predictions
    actual_time_steps = y_predict.shape[1]  # Should be the middle dimension
    print(f"DEBUG: Actual time steps in predictions: {actual_time_steps}")
    
    # Transpose to (time_steps, batch_size, num_classes)
    # Original shape should be (batch_size, time_steps, num_classes)
    y_predict = np.transpose(y_predict, (1, 0, 2))
    print(f"DEBUG: After transpose: {y_predict.shape}")
    
    sequence_lengths = tensor[:, 0]
    labels = tensor[:, 1:]
    print(f"DEBUG: Original sequence lengths from tensor: {sequence_lengths}")
    print(f"DEBUG: Labels shape: {labels.shape}")
    
    # Use actual prediction length instead of the sequence length from preprocessing
    # The sequence length in tensor might be from audio preprocessing, not model output
    actual_sequence_lengths = np.array([actual_time_steps] * len(sequence_lengths))
    print(f"DEBUG: Using actual sequence lengths: {actual_sequence_lengths}")
    
    # Create sparse labels (remove padding tokens)
    sparse_labels = []
    for batch_idx, seq_len in enumerate(sequence_lengths):
        batch_labels = labels[batch_idx]
        # Remove padding (assuming 28 is the padding token)
        valid_labels = batch_labels[batch_labels != 28]
        sparse_labels.append(valid_labels)
        print(f"DEBUG: Batch {batch_idx} valid labels length: {len(valid_labels)}")
    
    return sparse_labels, actual_sequence_lengths, y_predict

def ctc_ler_numpy(y_true, y_predict):
    """Calculate Label Error Rate using NumPy CTC decoder"""
    print("DEBUG: Computing CTC LER with NumPy")
    sparse_labels, sequence_lengths, y_predict = ctc_preparation_numpy(y_true, y_predict)
    
    # Decode using NumPy greedy decoder
    decoded_sequences = numpy_ctc_greedy_decoder(y_predict, sequence_lengths)
    
    if len(decoded_sequences) == 0 or len(sparse_labels) == 0:
        print("DEBUG: Empty sequences, returning 1.0 error rate")
        return 1.0, np.array([])
    
    # Calculate edit distance
    true_seq = sparse_labels[0]  # Assuming single batch
    pred_seq = decoded_sequences[0]
    
    print(f"DEBUG: True sequence length: {len(true_seq)}")
    print(f"DEBUG: Predicted sequence length: {len(pred_seq)}")
    
    edit_dist = numpy_edit_distance(true_seq, pred_seq)
    
    # Calculate LER
    ler = edit_dist / max(len(true_seq), 1)
    print(f"DEBUG: Label Error Rate: {ler}")
    
    return ler, decoded_sequences[0]

def trans_int_to_string(trans_int):
    """Convert integer sequence to string"""
    print(f"DEBUG: Converting {len(trans_int)} integers to string")
    if len(trans_int) > 0:
        print(f"DEBUG: Input integers min: {min(trans_int)}, max: {max(trans_int)}")
        print(f"DEBUG: First 10 integers: {trans_int[:10] if len(trans_int) >= 10 else trans_int}")
    
    string = ""
    alphabet = "abcdefghijklmnopqrstuvwxyz' @"
    alphabet_dict = {}
    count = 0
    for x in alphabet:
        alphabet_dict[count] = x
        count += 1
    
    print(f"DEBUG: Alphabet dictionary size: {len(alphabet_dict)} (0-{len(alphabet_dict)-1})")
    
    for i, letter in enumerate(trans_int):
        if isinstance(letter, np.ndarray):
            letter_val = letter.item()
        else:
            letter_val = int(letter)
        
        if letter_val != 28:  # Skip blank tokens
            if letter_val in alphabet_dict:
                string += alphabet_dict[letter_val]
            else:
                print(f"WARNING: Invalid letter value {letter_val} at position {i}, skipping")
                # Skip invalid values instead of crashing
                continue
    
    print(f"DEBUG: Converted string length: {len(string)}")
    return string

def ctc_wer_numpy(y_true, y_predict):
    """Calculate Word Error Rate using NumPy"""
    print("DEBUG: Computing CTC WER with NumPy")
    sparse_labels, sequence_lengths, y_predict = ctc_preparation_numpy(y_true, y_predict)
    
    # Decode using NumPy greedy decoder
    decoded_sequences = numpy_ctc_greedy_decoder(y_predict, sequence_lengths)
    
    if len(decoded_sequences) == 0 or len(sparse_labels) == 0:
        print("DEBUG: Empty sequences, returning 1.0 error rate")
        return 1.0
    
    # Convert to strings
    true_sentence = trans_int_to_string(sparse_labels[0])
    pred_sentence = trans_int_to_string(decoded_sequences[0])
    
    print(f"DEBUG: True sentence: '{true_sentence[:100]}...'")  # First 100 chars
    print(f"DEBUG: Predicted sentence: '{pred_sentence[:100]}...'")  # First 100 chars
    
    # Calculate WER 
    wer_score = wer(true_sentence, pred_sentence)
    print(f"DEBUG: Word Error Rate: {wer_score}")
    
    return wer_score
import numpy as np
from typing import List, Union

def simple_tokenize(sentence: str) -> List[str]:
    """Basic whitespace tokenizer."""
    return sentence.lower().strip().split()

def levenshtein_alignment(ref: List[str], hyp: List[str]):
    """Compute Levenshtein distance with backtrace for alignment."""
    n = len(ref)
    m = len(hyp)

    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution / Match
            )

    # Backtrace to get opcodes
    i, j = n, m
    substitutions = deletions = insertions = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1

    return substitutions, deletions, insertions, hits

def wer(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]]
) -> float:
    # Handle list of strings
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]

    if len(reference) != len(hypothesis):
        raise ValueError("Reference and hypothesis lengths must match")

    total_subs = total_dels = total_ins = total_hits = total_ref_words = 0

    for ref_sent, hyp_sent in zip(reference, hypothesis):
        ref_words = simple_tokenize(ref_sent)
        hyp_words = simple_tokenize(hyp_sent)

        subs, dels, ins, hits = levenshtein_alignment(ref_words, hyp_words)

        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_hits += hits
        total_ref_words += len(ref_words)

    if total_ref_words == 0:
        if len(hypothesis) == 0 or all(len(simple_tokenize(h)) == 0 for h in hypothesis):
            return 0.0  # silence match
        return 1.0  # silence mismatch

    wer_value = (total_subs + total_dels + total_ins) / total_ref_words
    return wer_value

def evaluate_tflite(tflite_path, input_window_length=296):
    """Evaluates tflite model using NumPy/SciPy for preprocessing"""
    print(f"DEBUG: Starting TFLite evaluation with window length {input_window_length}")
    results = []
    data, label = transform_audio_to_mfcc(audio_file, transcript_ints)
    
    print("DEBUG: Initializing TFLite interpreter")
    interpreter = tflite.Interpreter(
                    model_path=tflite_path,
                    experimental_delegates=[
                        tflite.load_delegate('/nix/store/96bsy96b042wsqgzazpdhcdkqhai9k7n-vx-delegate-aarch64-unknown-linux-gnu-v-tf2.14.0/lib/libvx_delegate.so')
                    ]
                )
    interpreter.allocate_tensors()
    input_chunk = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_shape = input_chunk["shape"]
    log("eval_model() - input_shape: {}".format(input_shape))
    input_dtype = input_chunk["dtype"]
    output_dtype = output_details["dtype"]
    
    print(f"DEBUG: Input dtype: {input_dtype}, Output dtype: {output_dtype}")
    
    # Check if the input/output type is quantized
    if input_dtype != np.float32:
        input_scale, input_zero_point = input_chunk["quantization"]
        print(f"DEBUG: Input quantization - scale: {input_scale}, zero_point: {input_zero_point}")
    else:
        input_scale, input_zero_point = 1, 0
        print("DEBUG: No input quantization")
    
    if output_dtype != np.float32:
        output_scale, output_zero_point = output_details["quantization"]
        print(f"DEBUG: Output quantization - scale: {output_scale}, zero_point: {output_zero_point}")
    else:
        output_scale, output_zero_point = 1, 0
        print("DEBUG: No output quantization")
    
    print("DEBUG: Applying input quantization")
    data = data / input_scale + input_zero_point
    
    # Round the data if dtype is not float32
    if input_dtype is not np.float32:
        print("DEBUG: Rounding quantized data")
        data = np.round(data)
    
    print(f"DEBUG: Data shape before padding: {data.shape}")
    
    # Pad data if needed
    while data.shape[1] < input_window_length:
        print("DEBUG: Padding data to minimum window length")
        data = np.append(data, data[:, -2:-1, :], axis=1)
    
    # Zero-pad any odd-length inputs
    if data.shape[1] % 2 == 1:
        print("DEBUG: Zero-padding odd-length input")
        data = np.concatenate([data, np.zeros((1, 1, data.shape[2]), dtype=input_dtype)], axis=1)
    
    print(f"DEBUG: Final data shape: {data.shape}")
    
    context = 24 + 2 * (7 * 3 + 16)  # = 98 - theoretical max receptive field
    size = input_chunk['shape'][1]
    inner = size - 2 * context
    data_end = data.shape[1]
    
    print(f"DEBUG: Context: {context}, Size: {size}, Inner: {inner}, Data end: {data_end}")
    
    # Initialize variables for the sliding window loop
    data_pos = 0
    outputs = []
    window_count = 0
    
    print("DEBUG: Starting sliding window inference")
    while data_pos < data_end:
        window_count += 1
        print(f"DEBUG: Processing window {window_count}")
        
        if data_pos == 0:
            # First window
            start = data_pos
            end = start + size
            y_start = 0
            y_end = y_start + (size - context) // 2
            data_pos = end - context
            print(f"DEBUG: First window - start: {start}, end: {end}, y_start: {y_start}, y_end: {y_end}")
        elif data_pos + inner + context >= data_end:
            # Final window
            shift = (data_pos + inner + context) - data_end
            start = data_pos - context - shift
            end = start + size
            assert start >= 0, f"Start position {start} is negative"
            y_start = (shift + context) // 2
            y_end = size // 2
            data_pos = data_end
            print(f"DEBUG: Final window - start: {start}, end: {end}, y_start: {y_start}, y_end: {y_end}")
        else:
            # Middle windows
            start = data_pos - context
            end = start + size
            y_start = context // 2
            y_end = y_start + inner // 2
            data_pos = end - context
            print(f"DEBUG: Middle window - start: {start}, end: {end}, y_start: {y_start}, y_end: {y_end}")
        
        # Run inference
        window_data = data[:, start:end, :].astype(input_dtype)
        print(f"DEBUG: Window data shape: {window_data.shape}")
        
        interpreter.set_tensor(input_chunk["index"], window_data)
        interpreter.invoke()
        cur_output_data = interpreter.get_tensor(output_details["index"])[:, :, y_start:y_end, :]
        
        # Dequantize if needed
        cur_output_data = output_scale * (
                cur_output_data.astype(np.float32) - output_zero_point
        )
        print(f"DEBUG: Window output shape: {cur_output_data.shape}")
        outputs.append(cur_output_data)
    
    print("DEBUG: Concatenating window outputs")
    complete = np.concatenate(outputs, axis=2)
    print(f"DEBUG: Complete output shape: {complete.shape}")
    
    print("DEBUG: Computing LER and WER")
    LER, output = ctc_ler_numpy(label, complete)
    WER = ctc_wer_numpy(label, complete)
    
    return output, LER, WER

# Main execution
print("DEBUG: Starting main execution")
wav2letter_tflite_path = "wav2letter_pruned_int8.tflite"
output, LER, WER = evaluate_tflite(wav2letter_tflite_path)

decoded_output = [index_dict[value] for value in output]
log(f'Transcribed File: {"".join(decoded_output)}')
log(f'Letter Error Rate is {LER}')
log(f'Word Error Rate is {WER}')

print("DEBUG: Execution completed")
