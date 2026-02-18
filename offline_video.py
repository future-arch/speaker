
import os
import sys
import argparse
import subprocess
from pathlib import Path
import warnings
import ssl
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Bypass SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

def check_dependencies():
    """Check if ffmpeg is installed and supports libass (for burning subtitles)."""
    try:
        # Check basic ffmpeg
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
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
    return str(max(files, key=os.path.getmtime))

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

def post_process_subtitles(srt_path):
    """
    Resize English words and numbers in the SRT file to be smaller than Chinese characters.
    Uses ASS/SSA override tags {\fscx80\fscy80} supported by libass/ffmpeg.
    """
    print(f"Post-processing subtitles in {srt_path}...")
    
    import re
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Regex to find English words, numbers, and associated punctuation
    # We want to match continuous blocks of Latin characters/numbers
    # Avoid matching SRT timestamps (00:00:00,000) or index numbers
    # But since we are processing the whole file, we need to be careful not to break the structure.
    # Safer approach: Process line by line, and identify 'text' lines again.
    
    lines = content.splitlines()
    processed_lines = []
    
    # Simple state machine to identify text lines
    # 1. Index (digits)
    # 2. Timestamp (contains -->)
    # 3. Text (everything else until blank line)
    
    for line in lines:
        stripped = line.strip()
        
        # Check if it's a metadata line (index or timestamp)
        is_metadata = False
        if stripped.isdigit():
            is_metadata = True
        elif "-->" in stripped:
            is_metadata = True
        elif not stripped: # Empty line
            is_metadata = True
            
        if not is_metadata:
            # It's a text line. Apply formatting.
            # Match sequences of [a-zA-Z0-9] and basic punctuation that looks like English/Code
            # We want to wrap them in {\fscx75\fscy75} ... {\fscx100\fscy100}
            # Note: 80% might still be large, trying 75%
            
            # This regex matches words containing at least one latin char or number, 
            # allowing for some punctuation but trying to avoid capturing Chinese punctuation if possible.
            # \u0000-\u007F covers Basic Latin.
            # We'll target contiguous runs of non-Chinese/non-CJ characters that contain at least one alphanumeric.
            
            def resize_match(match):
                text = match.group(0)
                # Don't resize if it's just a space or purely punctuation
                if not any(c.isalnum() for c in text):
                    return text
                return f"{{\\fscx75\\fscy75}}{text}{{\\fscx100\\fscy100}}"
            
            # Pattern: non-Chinese characters (broadly). 
            # \u4e00-\u9fff is CJK Unified Ideographs.
            # We'll keep it simple: [a-zA-Z0-9\.\-\s]+
            # But we must ensure we don't match the whole line if it's mixed.
            
            # Let's match explicit English/Number blocks.
            # [a-zA-Z0-9]+ plus optional punctuation/spaces around it, but usually standard English text
            # regex: ([a-zA-Z0-9][a-zA-Z0-9\s\.\,\:\-\(\)]*[a-zA-Z0-9]) | ([a-zA-Z0-9]+)
            
            # Simpler: [a-zA-Z0-9\.\,\:\-\(\)\s]+
            # But avoid capturing Chinese spaces if possible (though they are usually different unicode)
            
            # Regex explanation:
            # [a-zA-Z0-9] : Start with alphanum
            # [a-zA-Z0-9\s\.\,\-\(\)\']* : allow content
            # (?<!\s) : Don't end with space (trim)
            
            # Actually, simply matching [a-zA-Z0-9]+ is too fragmented ("open" "BIM" vs "openBIM").
            # Let's try to match broader phrases.
            
            # Substitution
            new_line = re.sub(r'([a-zA-Z0-9][a-zA-Z0-9\s\.,\-\(\)\']*[a-zA-Z0-9]|[a-zA-Z0-9]+)', resize_match, line)
            
            # Cleanup: If we created adjacent tags like } { , merge them? 
            # e.g. {\fscx75\fscy75}Word{\fscx100\fscy100} {\fscx75\fscy75}Word2{\fscx100\fscy100}
            # It's fine for rendering, effectively resets and starts again.
            
            processed_lines.append(new_line)
        else:
            processed_lines.append(line)
            
    # Write back
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(processed_lines))
        
    print("Subtitle post-processing complete.")

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

def burn_subtitles(video_path, srt_path):
    """Burn subtitles into video (Hardsub) for web compatibility."""
    print(f"Burning subtitles into {video_path}...")
    
    # Use temporary simple filenames to avoid ffmpeg escaping issues
    video_dir = os.path.dirname(video_path)
    temp_video = os.path.join(video_dir, "temp_video_input.mp4")
    temp_srt = os.path.join(video_dir, "temp_subs_input.srt")
    temp_output = os.path.join(video_dir, "temp_burning_output.mp4")
    
    final_output = str(video_path).replace(".mp4", ".burned.mp4")
    
    # Rename originals to temp
    try:
        os.rename(video_path, temp_video)
        os.rename(srt_path, temp_srt)
        
        cmd = [
            "ffmpeg",
            "-i", temp_video,
            "-vf", f"subtitles={os.path.basename(temp_srt)}",
            "-c:a", "copy",
            temp_output,
            "-y"
        ]
        
        # Run ffmpeg in the directory to keep paths simple for the filter
        subprocess.run(cmd, cwd=video_dir, check=True)
        
        # Rename output
        if os.path.exists(temp_output):
            os.rename(temp_output, final_output)
            print(f"Burned video saved to {final_output}")
            return final_output
        else:
            print("Error: ffmpeg did not produce output.")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error during burning: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during burning: {e}")
        return None
    finally:
        # metadata restoration: Rename inputs back
        if os.path.exists(temp_video):
            os.rename(temp_video, video_path)
        if os.path.exists(temp_srt):
            os.rename(temp_srt, srt_path)
        # Clean up partial output if failed
        if os.path.exists(temp_output):
            os.remove(temp_output)

def main():
    parser = argparse.ArgumentParser(description="Download video and generate Chinese subtitles.")
    parser.add_argument("url", help="Video URL (YouTube/Vimeo)")
    parser.add_argument("--dir", default=".", help="Output directory")
    parser.add_argument("--model", default="turbo", help="Whisper model size (tiny, base, small, medium, large, turbo)")
    parser.add_argument("--burn", action="store_true", help="Burn subtitles into video (hardsub) for web compatibility")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.dir, exist_ok=True)
    
    check_dependencies()
    
    try:
        video_path = download_video(args.url, args.dir)
        srt_path = transcribe_audio(video_path, args.model)
        translated_srt_path = translate_srt(srt_path)
        
        # Post-process for font styling (resize English/Numbers)
        post_process_subtitles(translated_srt_path)
        
        # Always create embedded (soft) for QuickTime
        embed_subtitles(video_path, translated_srt_path)
        
        # Optionally create burned (hard) for Web
        if args.burn:
            burn_subtitles(video_path, translated_srt_path)
        
        print("\nDone! Video and subtitles are ready.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
