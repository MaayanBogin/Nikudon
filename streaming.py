"""
Streaming inference example for Whisper Hebrew Nikud Model

Demonstrates real-time transcription with chunk-by-chunk streaming.

Usage:
    python streaming.py --audio path/to/audio.wav
    python streaming.py --audio long_audio.mp3 --chunk-length 10
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import argparse
import sys
from pathlib import Path

def streaming_transcribe(audio_path, model_name="MayBog/whisper-hebrew-nikud-v1", chunk_length_s=30):
    """
    Transcribe audio with streaming output (chunk by chunk)
    
    Args:
        audio_path: Path to audio file
        model_name: HuggingFace model name or local path
        chunk_length_s: Length of audio chunks in seconds
        
    Returns:
        Full transcription
    """
    print("="*80)
    print("WHISPER HEBREW NIKUD - STREAMING INFERENCE")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Audio: {audio_path}")
    print(f"Chunk length: {chunk_length_s}s")
    print("="*80)
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"\n📥 Loading model on {device}...")
    
    try:
        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Create pipeline with streaming enabled
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,  # Enable timestamps for streaming
        )
        
        print(f"✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Transcribe with streaming
    print(f"\n🎤 Transcribing with streaming output...")
    print("-"*80)
    print("📝 Transcription (real-time):\n")
    
    try:
        # Generate with streaming
        result = pipe(
            audio_path,
            generate_kwargs={
                "language": "hebrew",
                "task": "transcribe",
            }
        )
        
        full_text = ""
        
        # If timestamps are available, show chunk by chunk
        if "chunks" in result:
            for i, chunk in enumerate(result["chunks"]):
                chunk_text = chunk["text"]
                timestamp = chunk.get("timestamp", (0, 0))
                
                print(f"[{timestamp[0]:.1f}s - {timestamp[1]:.1f}s] {chunk_text}")
                full_text += chunk_text
        else:
            # Fallback to full text
            full_text = result["text"]
            print(full_text)
        
        print("\n" + "-"*80)
        print("\n✅ Transcription complete!")
        print("\n📄 Full transcription:")
        print("-"*80)
        print(full_text)
        print("-"*80)
        
        return full_text
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Stream transcription of Hebrew audio with nikud"
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
        "--chunk-length",
        type=int,
        default=30,
        help="Audio chunk length in seconds (default: 30)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save transcription to text file"
    )
    
    args = parser.parse_args()
    
    # Transcribe with streaming
    transcription = streaming_transcribe(
        args.audio, 
        args.model, 
        args.chunk_length
    )
    
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