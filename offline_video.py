
import os
import sys
import argparse
import subprocess
from pathlib import Path
import warnings
import ssl

# Suppress warnings
warnings.filterwarnings("ignore")

# Bypass SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

def check_dependencies():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH. Please install it (e.g., 'brew install ffmpeg').")
        sys.exit(1)

def download_video(url, output_dir):
    """Download video and audio using yt-dlp."""
    print(f"Downloading video from {url}...")
    
    # Template for output filename
    output_template = os.path.join(output_dir, '%(title)s.%(ext)s')
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--output", output_template,
        "--no-playlist",
        "--write-thumbnail",
        url
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        sys.exit(1)
        
    # Find the downloaded file
    # This is a bit simplistic, might need improvement to catch the exact filename yt-dlp used
    # But for now, we'll search for the most recent mp4 file in the directory
    files = list(Path(output_dir).glob("*.mp4"))
    if not files:
        print("Error: Could not find downloaded video file.")
        sys.exit(1)
        
    # Return the most recently modified file
    return max(files, key=os.path.getmtime)

def transcribe_audio(video_path, model_size="base"):
    """Transcribe audio using OpenAI Whisper."""
    print(f"Transcribing {video_path} using Whisper ({model_size})...")
    
    import whisper
    from whisper.utils import get_writer

    model = whisper.load_model(model_size)
    result = model.transcribe(str(video_path))
    
    # Save SRT
    output_dir = os.path.dirname(video_path)
    audio_basename = os.path.basename(video_path).rsplit('.', 1)[0]
    
    writer = get_writer("srt", output_dir)
    writer(result, audio_basename)
    
    srt_path = os.path.join(output_dir, f"{audio_basename}.srt")
    print(f"Transcription saved to {srt_path}")
    return srt_path

def translate_srt(srt_path, target_lang="zh-CN"):
    """Translate SRT file to target language."""
    print(f"Translating {srt_path} to {target_lang}...")
    
    from deep_translator import GoogleTranslator
    
    translator = GoogleTranslator(source='auto', target=target_lang)
    
    translated_lines = []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Simple SRT parser: translate only lines that are text (not index or simple timestamps)
    # A more robust parser would be better, but this works for standard Whisper output
    
    # We'll assume the structure:
    # Index
    # Timestamp
    # Text
    # (Blank line)
    
    is_text = False
    
    translated_srt_path = srt_path.replace('.srt', f'.{target_lang}.srt')
    
    with open(translated_srt_path, 'w', encoding='utf-8') as f_out:
        for line in lines:
            stripped = line.strip()
            
            # Identify if line is text to translate
            # Heuristic: It's text if it's not a number (index) and doesn't contain "-->" (timestamp) and is not empty
            if stripped and not stripped.isdigit() and "-->" not in stripped:
                try:
                    translated = translator.translate(stripped)
                    f_out.write(f"{translated}\n")
                except Exception as e:
                    print(f"Translation warning: {e}")
                    f_out.write(line)
            else:
                f_out.write(line)

    print(f"Translation saved to {translated_srt_path}")
    return translated_srt_path

def embed_subtitles(video_path, srt_path):
    """Embed subtitles into MP4 for QuickTime compatibility."""
    print(f"Embedding subtitles into {video_path}...")
    
    output_path = str(video_path).replace(".mp4", ".embedded.mp4")
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", srt_path,
        "-c", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=zho",
        output_path,
        "-y"
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Embedded video saved to {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        print("Warning: Failed to embed subtitles.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download video and generate Chinese subtitles.")
    parser.add_argument("url", help="Video URL (YouTube/Vimeo)")
    parser.add_argument("--dir", default=".", help="Output directory")
    parser.add_argument("--model", default="turbo", help="Whisper model size (tiny, base, small, medium, large, turbo)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.dir, exist_ok=True)
    
    check_dependencies()
    
    try:
        video_path = download_video(args.url, args.dir)
        srt_path = transcribe_audio(video_path, args.model)
        translated_srt_path = translate_srt(srt_path)
        embed_subtitles(video_path, translated_srt_path)
        
        print("\nDone! Video and subtitles are ready.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
