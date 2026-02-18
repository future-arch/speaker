
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
    files = list(Path(output_dir).glob("*.mp4"))
    if not files:
        print("Error: Could not find downloaded video file.")
        sys.exit(1)
        
    # Return the most recently modified file
    # Exclude .embedded.mp4 and .burned.mp4 from detection to find the source
    source_files = [f for f in files if not f.name.endswith('.embedded.mp4') and not f.name.endswith('.burned.mp4')]
    if not source_files:
        return str(max(files, key=os.path.getmtime))
    return str(max(source_files, key=os.path.getmtime))

def transcribe_audio(video_path, model_size="base", skip_existing=False):
    """Transcribe audio using OpenAI Whisper."""
    output_dir = os.path.dirname(video_path)
    audio_basename = os.path.basename(video_path).rsplit('.', 1)[0]
    srt_path = os.path.join(output_dir, f"{audio_basename}.srt")

    if skip_existing and os.path.exists(srt_path):
        print(f"SRT file already exists at {srt_path}, skipping transcription.")
        return srt_path

    print(f"Transcribing {video_path} using Whisper ({model_size})...")
    
    import whisper
    from whisper.utils import get_writer

    model = whisper.load_model(model_size)
    result = model.transcribe(str(video_path))
    
    writer = get_writer("srt", output_dir)
    writer(result, audio_basename)
    
    print(f"Transcription saved to {srt_path}")
    return srt_path

def translate_srt(srt_path, target_lang="zh-CN", skip_existing=False):
    """Translate SRT file to target language."""
    translated_srt_path = srt_path.replace('.srt', f'.{target_lang}.srt')
    
    if skip_existing and os.path.exists(translated_srt_path):
        print(f"Translated SRT file already exists at {translated_srt_path}, skipping translation.")
        return translated_srt_path

    print(f"Translating {srt_path} to {target_lang}...")
    
    from deep_translator import GoogleTranslator
    
    translator = GoogleTranslator(source='auto', target=target_lang)
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(translated_srt_path, 'w', encoding='utf-8') as f_out:
        for line in lines:
            stripped = line.strip()
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

def burn_subtitles(video_path, srt_path, font_name=None):
    """Burn subtitles into video (Hardsub) for web compatibility."""
    print(f"Burning subtitles into {video_path}..." + (f" using font '{font_name}'" if font_name else ""))
    
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
        
        # Build subtitle filter string
        sub_filter = f"subtitles={os.path.basename(temp_srt)}"
        if font_name:
            # Add force_style to specify font
            # Example: subtitles=filename.srt:force_style='FontName=Arial'
            sub_filter += f":force_style='FontName={font_name}'"

        cmd = [
            "ffmpeg",
            "-i", temp_video,
            "-vf", sub_filter,
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
    parser.add_argument("--font", help="Specific font for burned subtitles (e.g., 'Heiti SC', 'Arial')")
    parser.add_argument("--skip-existing", action="store_true", help="Skip transcription/translation if files exist")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.dir, exist_ok=True)
    
    check_dependencies()
    
    try:
        video_path = download_video(args.url, args.dir)
        srt_path = transcribe_audio(video_path, args.model, args.skip_existing)
        translated_srt_path = translate_srt(srt_path, skip_existing=args.skip_existing)
        
        # Always create embedded (soft) for QuickTime
        embed_subtitles(video_path, translated_srt_path)
        
        # Optionally create burned (hard) for Web
        if args.burn:
            burn_subtitles(video_path, translated_srt_path, font_name=args.font)
        
        print("\nDone! Video and subtitles are ready.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
