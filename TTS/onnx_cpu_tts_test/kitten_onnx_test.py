#!/usr/bin/env python3

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
from scipy.special import logsumexp
import sys
sys.path.insert(0,'/home/scmd/audio_test_python_setup')
import tensorflow as tf
import librosa
from jiwer import wer
from collections import namedtuple
from collections import defaultdict, namedtuple
from pygtrie import Trie
import heapq
import difflib
import re

from misaki import en, espeak
import numpy as np
import phonemizer
import soundfile as sf
import onnxruntime as ort


def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        
        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i

        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


class KittenTTS_1_Onnx:
    def __init__(self, model_path="/home/scmd/onnx_tts_test/kitten_tts_nano_v0_1.onnx", voices_path="voices.npz"):
        """Initialize KittenTTS with model and voice data.
        
        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the voices NPZ file
        """
        self.model_path = model_path
        self.voices = np.load(voices_path)
        self.session = ort.InferenceSession(model_path)
        
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        
        # Available voices
        self.available_voices = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f', 
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f'
        ]
    
    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict:
        """Prepare ONNX model inputs from text and voice parameters."""
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")
        
        # Phonemize the input text
        phonemes_list = self.phonemizer.phonemize([text])
        
        # Process phonemes to get token IDs
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = ' '.join(phonemes)
        tokens = self.text_cleaner(phonemes)
        
        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(0)
        
        input_ids = np.array([tokens], dtype=np.int64)
        ref_s = self.voices[voice]
        
        return {
            "input_ids": input_ids,
            "style": ref_s,
            "speed": np.array([speed], dtype=np.float32),
        }
    
    def generate(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            
        Returns:
            Audio data as numpy array
        """
        onnx_inputs = self._prepare_inputs(text, voice, speed)
        
        outputs = self.session.run(None, onnx_inputs)
        
        # Trim audio
        audio = outputs[0][5000:-10000]

        return audio
    
    def generate_to_file(self, text: str, output_path: str, voice: str = "expr-voice-5-m", 
                          speed: float = 1.0, sample_rate: int = 24000) -> None:
        """Synthesize speech and save to file.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
        """
        audio = self.generate(text, voice, speed)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")


# Example usage
if __name__ == "__main__":
    model_path="/home/scmd/onnx_tts_test/kitten_tts_nano_v0_1.onnx"
    voices_path="/home/scmd/onnx_tts_test/voices.npz"
    ttsmodel = KittenTTS_1_Onnx(model_path=model_path, voices_path=voices_path)
    
    text = """
    It begins with an "Ugh!" Another mysterious stain appears on a favorite shirt. Every trick has been tried, but the stain persists.
    """

    ttsmodel.generate_to_file(text, "inference_output25.wav", voice="expr-voice-5-m")
