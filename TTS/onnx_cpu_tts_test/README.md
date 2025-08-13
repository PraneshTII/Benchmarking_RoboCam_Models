#Download the Piper models and JSON
```
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```
Kitten TTS Models will downloaded during load from the HF Repo: https://huggingface.co/KittenML/kitten-tts-nano-0.1/tree/main

```
[scmd@nixos:~/onnx_tts_test]$ python run_tts_benchmark.py 
🚀 Starting TTS Performance Benchmark
==================================================
📝 Testing with 4 texts
🔄 3 runs per text (+ 1 warmup)

🐱 Setting up KittenTTS...

🔥 Benchmarking KittenTTS
==================================================

Text 1/4: 'Hello world, this is a quick test.'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 5.022, Latency: 10.045s
  Run 2/3... ✅ RTF: 4.701, Latency: 9.401s
  Run 3/3... ✅ RTF: 4.998, Latency: 9.997s

Text 2/4: 'Welcome to the world of speech synthesis technolog...'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 4.546, Latency: 12.730s
  Run 2/3... ✅ RTF: 4.759, Latency: 13.324s
  Run 3/3... ✅ RTF: 4.714, Latency: 13.199s

Text 3/4: 'This high quality text-to-speech model works effic...'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 3.574, Latency: 24.034s
  Run 2/3... ✅ RTF: 3.823, Latency: 25.707s
  Run 3/3... ✅ RTF: 3.800, Latency: 25.557s

Text 4/4: 'The development of modern TTS systems has revoluti...'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 3.535, Latency: 33.756s
  Run 2/3... ✅ RTF: 3.558, Latency: 33.978s
  Run 3/3... ✅ RTF: 3.367, Latency: 32.158s
🎵 Setting up Piper TTS...

🔥 Benchmarking Piper TTS
==================================================

Text 1/4: 'Hello world, this is a quick test.'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 1.291, Latency: 2.743s
  Run 2/3... ✅ RTF: 1.299, Latency: 2.804s
  Run 3/3... ✅ RTF: 1.319, Latency: 2.834s

Text 2/4: 'Welcome to the world of speech synthesis technolog...'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 1.305, Latency: 3.514s
  Run 2/3... ✅ RTF: 1.329, Latency: 3.903s
  Run 3/3... ✅ RTF: 1.281, Latency: 3.464s

Text 3/4: 'This high quality text-to-speech model works effic...'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 0.964, Latency: 6.302s
  Run 2/3... ✅ RTF: 1.092, Latency: 7.377s
  Run 3/3... ✅ RTF: 0.950, Latency: 6.356s

Text 4/4: 'The development of modern TTS systems has revoluti...'
  Warming up... (1 runs)
  Run 1/3... ✅ RTF: 1.020, Latency: 8.490s
  Run 2/3... ✅ RTF: 1.044, Latency: 8.859s
  Run 3/3... ✅ RTF: 1.051, Latency: 8.809s

================================================================================
📊 BENCHMARK SUMMARY
================================================================================

🤖 KittenTTS
------------------------------------------------------------
⚡ SPEED METRICS:
   Real-Time Factor (RTF):     4.200 (Slower than real-time)
   Average Inference Latency:  20.324 seconds
   Throughput (chars/sec):     4.1
   Throughput (words/sec):     0.6

💻 EFFICIENCY METRICS:
   Peak CPU Usage:             376.3%
   Average CPU Usage:          195.9%
   Peak Memory Usage:          966.4 MB
   Memory Increase:            9.6 MB

🤖 Piper TTS
------------------------------------------------------------
⚡ SPEED METRICS:
   Real-Time Factor (RTF):     1.162 (Slower than real-time)
   Average Inference Latency:  5.454 seconds
   Throughput (chars/sec):     15.2
   Throughput (words/sec):     2.3

💻 EFFICIENCY METRICS:
   Peak CPU Usage:             323.2%
   Average CPU Usage:          225.1%
   Peak Memory Usage:          1316.2 MB
   Memory Increase:            10.6 MB

🏆 COMPARISON
------------------------------------------------------------
🥇 Fastest (RTF):           Piper TTS (1.162)
🥇 Lowest Latency:          Piper TTS (5.454s)
🥇 Most CPU Efficient:      KittenTTS (195.9%)
🥇 Most Memory Efficient:   KittenTTS (966.4 MB)

💾 Detailed results saved to: tts_benchmark_results.csv

✨ Benchmark complete!

Run with --custom flag to test your own texts
```
