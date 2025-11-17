"""

Inference script for Whisper Hebrew Nikud Model

Usage:

    python inference.py --audio path/to/audio.wav

    python inference.py --audio path/to/audio.mp3 --model MayBog/whisper-hebrew-nikud-v1

"""

import torch

from transformers import pipeline

import argparse

import sys

from pathlib import Path

def transcribe_audio(audio_path, model_name="MayBog/whisper-hebrew-nikud-v1"):

    """

    Transcribe audio file to Hebrew text with diacritics (nikud)

    

    Args:

        audio_path: Path to audio file

        model_name: HuggingFace model name or local path

        

    Returns:

        Transcribed text with nikud

    """

    print("="*80)

    print("WHISPER HEBREW NIKUD - INFERENCE")

    print("="*80)

    print(f"Model: {model_name}")

    print(f"Audio: {audio_path}")

    print("="*80)

    

    # Check if audio file exists

    if not Path(audio_path).exists():

        print(f"❌ Error: Audio file not found: {audio_path}")

        sys.exit(1)

    

    # Load model

    print("\n📥 Loading model...")

    device = 0 if torch.cuda.is_available() else "cpu"

    

    try:

        pipe = pipeline(

            task="automatic-speech-recognition",

            model=model_name,

            chunk_length_s=30,

            device=device,

        )

        print(f"✅ Model loaded successfully on {device}")

    except Exception as e:

        print(f"❌ Error loading model: {e}")

        sys.exit(1)

    

    # Transcribe

    print(f"\n🎤 Transcribing audio...")

    try:

        result = pipe(

            audio_path, 

            generate_kwargs={"language": "hebrew", "task": "transcribe"}

        )

        text = result["text"]

        

        print(f"\n📝 Transcription Result:")

        print("-"*80)

        print(text)

        print("-"*80)

        

        return text

        

    except Exception as e:

        print(f"❌ Error during transcription: {e}")

        sys.exit(1)

def main():

    parser = argparse.ArgumentParser(

        description="Transcribe Hebrew audio to text with diacritics (nikud)"

    )

    parser.add_argument(

        "--audio",

        type=str,

        required=True,

        help="Path to audio file (wav, mp3, etc.)"

    )

    parser.add_argument(

        "--model",

        type=str,

        default="MayBog/whisper-hebrew-nikud-v1",

        help="HuggingFace model name or local model path"

    )

    parser.add_argument(

        "--output",

        type=str,

        default=None,

        help="Optional: Save transcription to text file"

    )

    

    args = parser.parse_args()

    

    # Transcribe

    transcription = transcribe_audio(args.audio, args.model)

    

    # Save to file if requested

    if args.output:

        try:

            with open(args.output, 'w', encoding='utf-8') as f:

                f.write(transcription)

            print(f"\n💾 Transcription saved to: {args.output}")

        except Exception as e:

            print(f"❌ Error saving file: {e}")

    

    print("\n✅ Done!")

if __name__ == "__main__":

    main()