# 🎙️ Nikudon - Whisper Model for Diacritic Hebrew

[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/MayBog/whisper-hebrew-nikud-v1)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**The first Hebrew speech-to-text model that directly transcribes audio to text with diacritics (niqqud).**

## 🌟 Overview

This project presents **whisper-hebrew-nikud-v1**, a groundbreaking speech recognition model that solves a long-standing challenge in Hebrew NLP: direct transcription of spoken Hebrew to properly vocalized text with diacritics (niqqud).

### The Problem

Hebrew text is typically written without vowel marks (niqqud/diacritics), which can lead to ambiguity. While readers familiar with Hebrew can infer the correct pronunciation from context, this creates several challenges:

1. **Text-to-Speech**: TTS systems need nikud to pronounce words correctly
2. **Educational Tools**: Language learning applications require properly vocalized text
3. **Two-Step Pipeline**: Existing solutions require transcription first, then separate nikud restoration - introducing latency and compounding errors

### The Solution

**whisper-hebrew-nikud-v1** is the **first model** to directly transcribe Hebrew speech to text with nikud in a single step, eliminating the need for post-processing nikud restoration. This approach:

- ✅ **Improves accuracy** - nikud is predicted directly from audio, not inferred from ambiguous text
- ✅ **Simplifies deployment** - one model instead of separate ASR + nikud models
- ✅ **Enables real-time applications** - fast enough for live transcription with diacritics

## 🎯 Model Details

- **Base Model**: Built on [ivrit.ai's whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo)
- **Architecture**: Whisper Large v3 Turbo
- **Task**: Automatic Speech Recognition with Diacritics
- **Training Data**: Hebrew speech paired with vocalized text
- **Model Size**: ~809 million parameters

## 🙏 Acknowledgments

This project builds upon the excellent work of:

- **[ivrit.ai](https://ivrit.ai/)** - For their foundational Hebrew Whisper model that served as the base for this work
- **[thewhiteagle](https://github.com/thewhiteagle)** - For the original idea and the [phonikud](https://github.com/thewhiteagle/phonikud) project, which inspired the data preprocessing approach and methodology used in creating the training dataset

Without their contributions, this project would not have been possible.

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch gradio
```

### Basic Usage

```python
from transformers import pipeline
import torch

# Load model
device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model="MayBog/whisper-hebrew-nikud-v1",
    chunk_length_s=30,
    device=device,
)

# Transcribe audio
result = pipe(
    "audio.wav",
    generate_kwargs={"language": "hebrew", "task": "transcribe"}
)

print(result["text"])  # Output with nikud!
```

## 📦 Scripts Included

This repository includes several ready-to-use scripts:

### 1. `inference.py` - Command-line Inference

Simple command-line tool for transcribing audio files:

```bash
# Basic usage
python inference.py --audio sample.wav

# Specify model
python inference.py --audio sample.mp3 --model MayBog/whisper-hebrew-nikud-v1

# Save output to file
python inference.py --audio sample.wav --output transcription.txt
```

### 2. `gradio.py` - Web Interface

Interactive web interface with microphone support:

```bash
# Launch local interface
python gradio.py

# Create public share link
python gradio.py --share

# Custom port
python gradio.py --port 8080
```


### 3. `streaming.py` - Streaming Inference

Real-time transcription with chunk-by-chunk output:

```bash
# Stream transcription with timestamps
python streaming.py --audio long_audio.wav

# Adjust chunk size for faster streaming
python streaming.py --audio speech.mp3 --chunk-length 10

# Save final output
python streaming.py --audio audio.wav --output transcription.txt
```

### 4. `post_process.py` - Text Cleaning

Filter unwanted characters from transcription output:

```bash
# Clean text from command line
python post_process.py --text "הַבַּיְתָה"

# Process files
python post_process.py --input raw.txt --output cleaned.txt
```

Removes:
- `|` (pipe character)
- `\u05ab` (Hebrew Accent Ole)
- `\u05af` (Hebrew Mark Masora Circle)


## 📈 Limitations & Future Work

### Current Limitations

- **Speed**: Real-time factor depends on hardware (GPU recommended)
- **Nikud Consistency**: May occasionally produce inconsistent nikd patterns

### Planned Improvements

- [ ] Support for different Hebrew dialects and accents
- [ ] Enhanced noise robustness
- [ ] Smaller whisper for faster streaming


## 🤝 Contributing

Contributions are welcome!

## 📚 Citation
If you use this model in your research or applications, please cite this repository 

## 🔗 Links

- 🤗 [Model on Hugging Face](https://huggingface.co/MayBog/whisper-hebrew-nikud-v1)
- 🌐 [ivrit.ai - Hebrew NLP Resources](https://ivrit.ai/)
- 💡 [phonikud - Phoneme-based Nikud Restoration](https://github.com/thewhiteagle/phonikud)

**Made with ❤️ for the Hebrew NLP community**
