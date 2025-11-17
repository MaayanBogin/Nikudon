"""
Post-processing script for Hebrew text with nikud

Filters specific unwanted characters from transcription output.

Usage:
    python post_process.py --text "הַבַּיְתָה"
    python post_process.py --input input.txt --output output.txt
"""

import argparse

# Characters to filter out
CHARS_TO_REMOVE = {
    '|',        # Pipe character
    '\u05ab',   # HEBREW ACCENT OLE
    '\u05af',   # HEBREW MARK MASORA CIRCLE
}

def post_process(text: str) -> str:
    """
    Remove unwanted characters from Hebrew text with nikud
    
    Args:
        text: Input text with nikud
        
    Returns:
        Processed text with filtered characters removed
    """
    processed = ''
    for c in text:
        if c not in CHARS_TO_REMOVE:
            processed += c
    return processed

def main():
    parser = argparse.ArgumentParser(
        description="Post-process Hebrew text with nikud - removes specific unwanted characters"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to process (alternative to --input)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (if not specified, prints to stdout)"
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.text:
        input_text = args.text
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        print("❌ Error: Please provide --text or --input")
        return
    
    # Process - remove unwanted characters
    processed = post_process(input_text)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(processed)
        print(f"✅ Processed text saved to: {args.output}")
        print(f"📊 Removed characters: | \\u05ab \\u05af")
    else:
        print(processed)

if __name__ == "__main__":
    main()