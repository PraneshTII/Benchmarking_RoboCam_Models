Repo referred are : [https://github.com/dscripka/openWakeWord/](https://github.com/dscripka/openWakeWord/releases/)
 
Model download link: https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.tflite

Instructions for training on custom wakeword are given here:  https://github.com/Mic92/openWakeWord/tree/main

```
$ python alexa_test.py 

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
✓ Alexa model loaded: /home/scmd/tts/wakeword/alexa_v0.1.tflite
✓ Input shape: [ 1 16 96]
✓ Output shape: [1 1]
✓ Ready to test 'Alexa' wake word detection
✓ Audio loaded: 48000 samples
📊 Max confidence score: 0.4901
⏰ Detected at time: 1.99 seconds

📈 SCORE ANALYSIS:
   Mean score: 0.4877
   Std deviation: 0.0005
   Score range: 0.4853 - 0.4901

🔝 TOP 5 DETECTION PEAKS:
   1. Score: 0.4901 at 1.99s
   2. Score: 0.4900 at 1.74s
   3. Score: 0.4895 at 1.96s
   4. Score: 0.4894 at 1.80s
   5. Score: 0.4893 at 1.61s

🎯 THRESHOLD SUGGESTIONS:
   Current threshold: 0.5
   Suggested threshold: 0.4887 (mean + 2*std)
   Conservative threshold: 0.4801

📋 SUMMARY:
   Audio duration: 3.00 seconds
   Processed 291 audio chunks
   Peak detection at: 1.99s
```
