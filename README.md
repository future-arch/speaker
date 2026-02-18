
# Video Offlining and Translation Tool

A tool to download videos from buildingSMART Education (via YouTube/Vimeo), generate subtitles using OpenAI Whisper, translate them to Chinese, and embed them for offline viewing.

## Features
- **Download**: Supports YouTube and Vimeo links.
- **Transcribe**: Uses OpenAI Whisper (local models) for high-accuracy English transcription.
- **Translate**: Translates subtitles to Chinese (Simplified).
- **Embed**: Automatically embeds Chinese subtitles into MP4 for QuickTime compatibility.
- **Turbo Model**: Uses `large-v3-turbo` by default for the best balance of speed and accuracy.

## Prerequisites

1.  **Python 3.8+**
2.  **FFmpeg**: Required for audio processing and subtitle embedding.
    ```bash
    brew install ffmpeg
    ```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/speaker.git
    cd speaker
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script with a video URL:

```bash
python3 offline_video.py "YOUR_VIDEO_URL_HERE" --model turbo
```

### Arguments
- `URL`: The YouTube or Vimeo URL of the video (Required).
- `--dir`: Output directory (Default: current directory).
- `--model`: Whisper model size. Options: `tiny`, `base`, `small`, `medium`, `large`, `turbo`. `turbo` is recommended. (Default: `turbo`).
- `--burn`: Burn subtitles into the video (hardsub) for web compatibility. This takes longer but ensures subtitles show on all players.

## Example

To download "What is openBIM?" and burn subtitles:

```bash
python3 offline_video.py "https://www.youtube.com/watch?v=LsV3z27iSGc" --dir "downloads/openbim" --burn
```

### Output Files
The script generates:
1.  `Video Title.mp4`: The original video.
2.  `Video Title.srt`: English transcript.
3.  `Video Title.zh-CN.srt`: Chinese translation.
4.  `Video Title.embedded.mp4`: Video with embedded Chinese subtitles (Ready for QuickTime).

## How to Watch with Subtitles

### QuickTime Player (Mac Default)
**Use the `.embedded.mp4` file.**
1.  Open `...embedded.mp4` in QuickTime.
2.  Click the **Subtitle icon** (speech bubble) in the playback controls.
3.  Select **"Chinese"** (or "Unknown" if language isn't detected) to enable them.

### Other Players (IINA, VLC)
1.  **Automatic**: Ensure the video file (`.mp4`) and the subtitle file (`.zh-CN.srt`) are in the same folder and have the same name. Most players (IINA, VLC) will load it automatically.
2.  **Manual**: Open the video, then drag and drop the `.zh-CN.srt` file directly into the player window.

## Troubleshooting
- **SSL Errors**: The script handles SSL certificate issues specifically for macOS python environments.
- **Download Fails**: Ensure you have a stable internet connection. Some videos may be region-locked.
- **Translation Quality**: Machine translation is used. It may struggle with technical terms (e.g., "IFC Stair" becoming "IFC Stare").
    - **Tip**: You can manually edit the `.zh-CN.srt` file to fix errors, then re-embed or use the external file.
