"""

Gradio web interface for Whisper Hebrew Nikud Model

Usage:

    python gradio.py

    python gradio.py --model MayBog/whisper-hebrew-nikud-v1

    python gradio.py --share  # Create public link

"""

import torch

from transformers import pipeline

import gradio as gr

import argparse

def load_model(model_name):

    """Load the Whisper model"""

    print(f"📥 Loading model: {model_name}")

    device = 0 if torch.cuda.is_available() else "cpu"

    

    pipe = pipeline(

        task="automatic-speech-recognition",

        model=model_name,

        chunk_length_s=30,

        device=device,

    )

    

    print(f"✅ Model loaded on {device}")

    return pipe

def create_gradio_interface(model_name="MayBog/whisper-hebrew-nikud-v1"):

    """Create and return Gradio interface"""

    

    # Load model once at startup

    pipe = load_model(model_name)

    

    def transcribe(audio):

        """Transcribe audio file or recording"""

        if audio is None:

            return "⚠️ נא להעלות קובץ אודיו או להקליט\n\nPlease upload an audio file or record audio"

        

        try:

            result = pipe(

                audio, 

                generate_kwargs={"language": "hebrew", "task": "transcribe"}

            )

            return result["text"]

        except Exception as e:

            return f"❌ Error during transcription: {str(e)}"

    

    # Create interface

    interface = gr.Interface(

        fn=transcribe,

        inputs=gr.Audio(

            sources=["upload", "microphone"], 

            type="filepath", 

            label="Upload Audio File or Record / העלה קובץ אודיו או הקלט"

        ),

        outputs=gr.Textbox(

            label="Hebrew Transcription with Nikud / תמלול עברית עם ניקוד",

            lines=10,

            rtl=True,

            show_copy_button=True,

            placeholder="התמלול יופיע כאן...\n\nTranscription will appear here..."

        ),

        title="🎙️ Hebrew Speech-to-Text with Nikud | תמלול עברית עם ניקוד",

        description=f"""

        Transcribe Hebrew audio to text with diacritics (nikud) using Whisper model.

        

        **Model:** {model_name}

        

        תמלול אודיו עברית לטקסט עם ניקוד באמצעות מודל Whisper.

        """,

        theme="soft",

        examples=[

            # Add example audio files here if available

            # ["examples/sample1.wav"],

            # ["examples/sample2.mp3"],

        ],

        allow_flagging="never",

        analytics_enabled=False

    )

    

    return interface

def main():

    parser = argparse.ArgumentParser(

        description="Launch Gradio interface for Hebrew speech transcription with nikud"

    )

    parser.add_argument(

        "--model",

        type=str,

        default="MayBog/whisper-hebrew-nikud-v1",

        help="HuggingFace model name or local model path"

    )

    parser.add_argument(

        "--share",

        action="store_true",

        help="Create a public share link"

    )

    parser.add_argument(

        "--port",

        type=int,

        default=7860,

        help="Port to run the server on (default: 7860)"

    )

    parser.add_argument(

        "--host",

        type=str,

        default="0.0.0.0",

        help="Host to run the server on (default: 0.0.0.0)"

    )

    

    args = parser.parse_args()

    

    print("="*80)

    print("WHISPER HEBREW NIKUD - GRADIO INTERFACE")

    print("="*80)

    print(f"Model: {args.model}")

    print(f"Host: {args.host}")

    print(f"Port: {args.port}")

    print(f"Share: {args.share}")

    print("="*80)

    

    # Create and launch interface

    interface = create_gradio_interface(args.model)

    

    print("\n🚀 Launching Gradio interface...")

    interface.launch(

        share=args.share,

        server_name=args.host,

        server_port=args.port

    )

if __name__ == "__main__":

    main()
