#!/usr/bin/env python3
"""Extract transcript and structured information from large videos via Gemini."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - dependency validation path
    raise SystemExit(
        "Missing dependency: google-genai. Install with: pip install google-genai"
    ) from exc


READY_STATES = {"ACTIVE", "READY", "SUCCEEDED"}
FAILED_STATES = {"FAILED", "ERROR", "CANCELLED"}
TOKEN_LIMIT_MARKER = "input token count"
DEFAULT_VIDEO = Path("sources/呼和运维_第三次线下培训-运维实操_魏来.mov")


@dataclass
class SegmentInfo:
    """Information about one segment used for Gemini extraction."""

    index: int
    path: str
    mime_type: str
    offset_seconds: int
    duration_seconds: int


@dataclass
class ExtractionMetadata:
    """Metadata for one extraction run."""

    run_id: str
    mode: str
    model: str
    source_video_path: str
    prompt_requirements: str
    generated_at_utc: str
    segment_count: int
    fallback_triggered: bool


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Use Gemini Files API for large video transcript extraction."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Video path.")
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model name, e.g. gemini-2.0-flash or gemini-3-flash-preview.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "single", "segments"],
        default="auto",
        help="auto: single then fallback to segments on token overflow.",
    )
    parser.add_argument(
        "--segment-dir",
        type=Path,
        default=None,
        help="Existing segment directory (audio/video files).",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=600,
        help="Seconds per segment for auto split or offset estimate.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Max segments to process (0 means all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/A/gemini_video_extract"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--requirements",
        default=(
            "请输出中文结果，包含："
            "1) 带时间信息的尽量完整文字稿；"
            "2) 关键主题与术语（术语给出一句定义）；"
            "3) 关键结论与证据片段（标注时间）；"
            "4) 可执行行动项（面向运维培训场景）。"
        ),
        help="User requirements appended to prompt.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional .env file containing GEMINI_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=8,
        help="Seconds between file status checks.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=1800,
        help="Max seconds for one uploaded file to become ready.",
    )
    parser.add_argument(
        "--delete-remote-file",
        action="store_true",
        help="Delete each uploaded file on Gemini side after use.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env file."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    """Return Gemini API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment/.env."
        )
    return api_key


def resolve_mime_type(path: Path) -> str:
    """Resolve MIME type for local media files."""
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() == ".m4a":
        return "audio/mp4"
    if path.suffix.lower() == ".mp3":
        return "audio/mpeg"
    return "application/octet-stream"


def get_file_state_name(file_obj: Any) -> str:
    """Normalize Gemini file state string."""
    state = getattr(file_obj, "state", None)
    if state is None:
        return "UNKNOWN"
    state_name = getattr(state, "name", state)
    return str(state_name).upper()


def to_ascii_upload_path(path: Path) -> tuple[Path, Path | None]:
    """Return ASCII-safe upload path and optional temp dir for cleanup."""
    try:
        path.name.encode("ascii")
        return path, None
    except UnicodeEncodeError:
        temp_dir = Path(tempfile.mkdtemp(prefix="gemini_upload_"))
        suffix = path.suffix or ".bin"
        staged = temp_dir / f"media_upload{suffix}"
        try:
            staged.symlink_to(path.resolve())
        except OSError:
            shutil.copy2(path, staged)
        return staged, temp_dir


def upload_file(client: genai.Client, media_path: Path, mime_type: str) -> Any:
    """Upload media via Gemini Files API."""
    try:
        return client.files.upload(file=str(media_path), config={"mime_type": mime_type})
    except TypeError:
        return client.files.upload(file=str(media_path))


def wait_for_file_ready(
    client: genai.Client,
    file_name: str,
    poll_interval: int,
    poll_timeout: int,
) -> Any:
    """Poll Gemini file until READY or failure."""
    start = time.time()
    while True:
        current = client.files.get(name=file_name)
        state_name = get_file_state_name(current)
        if state_name in READY_STATES:
            return current
        if state_name in FAILED_STATES:
            raise RuntimeError(f"File processing failed with state: {state_name}")
        elapsed = time.time() - start
        if elapsed >= poll_timeout:
            raise TimeoutError(
                f"Timeout waiting for file readiness. Last state: {state_name}"
            )
        print(f"[wait] state={state_name}, elapsed={int(elapsed)}s", flush=True)
        time.sleep(poll_interval)


def maybe_delete_remote_file(client: genai.Client, file_name: str, enabled: bool) -> None:
    """Delete uploaded Gemini file if enabled."""
    if enabled and file_name:
        client.files.delete(name=file_name)
        print(f"[cleanup] deleted remote file: {file_name}")


def response_to_text(response: Any) -> str:
    """Extract generated text from Gemini response object."""
    text = getattr(response, "text", None)
    if text:
        return text
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []) if content else []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    if chunks:
        return "\n".join(chunks)
    return json.dumps(response_to_mapping(response), ensure_ascii=False, indent=2)


def response_to_mapping(response: Any) -> dict[str, Any]:
    """Convert Gemini response object to serializable mapping across SDK versions."""
    if hasattr(response, "to_dict") and callable(response.to_dict):
        return response.to_dict()
    if hasattr(response, "model_dump") and callable(response.model_dump):
        return response.model_dump()
    if hasattr(response, "dict") and callable(response.dict):
        return response.dict()
    try:
        return json.loads(json.dumps(response, default=str))
    except TypeError:
        return {"raw_response": str(response)}


def build_single_prompt(requirements: str) -> str:
    """Build prompt for full-file processing."""
    return (
        "你是中文技术培训视频分析助手。严格基于输入内容输出，不得臆造。"
        "若不确定请明确标注“不确定”并给出补证路径。"
        "输出 Markdown，结构如下："
        "## 1. 文字稿"
        "## 2. 关键信息摘要"
        "## 3. 主题/术语表"
        "## 4. 关键结论与证据（时间）"
        "## 5. 行动项"
        f"额外要求：{requirements}"
    )


def build_segment_prompt(requirements: str, segment: SegmentInfo) -> str:
    """Build prompt for one segment extraction."""
    return (
        f"你正在处理完整培训视频的第 {segment.index:03d} 分段。"
        f"该分段起始偏移约 {segment.offset_seconds} 秒，分段时长约 {segment.duration_seconds} 秒。"
        "请严格基于分段内容输出中文 Markdown："
        "## 1. 分段文字稿（尽量完整，保留关键时间线）"
        "## 2. 分段关键主题与术语"
        "## 3. 分段关键结论与证据（引用分段内时间点）"
        "## 4. 分段行动项"
        f"额外要求：{requirements}"
    )


def generate_with_uri(
    client: genai.Client,
    model: str,
    prompt: str,
    file_uri: str,
    mime_type: str,
) -> Any:
    """Call Gemini generate_content with file URI + prompt."""
    part = types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)
    return client.models.generate_content(model=model, contents=[part, prompt])


def run_one_media(
    client: genai.Client,
    model: str,
    media_path: Path,
    mime_type: str,
    prompt: str,
    poll_interval: int,
    poll_timeout: int,
    delete_remote: bool,
) -> tuple[str, dict[str, Any], Any]:
    """Upload one media file and return extracted text and response."""
    upload_path, temp_dir = to_ascii_upload_path(media_path)
    if upload_path != media_path:
        print(f"[prep] staged upload path: {upload_path}")
    uploaded = upload_file(client=client, media_path=upload_path, mime_type=mime_type)
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    remote_name = getattr(uploaded, "name", "")
    print(f"[upload] remote file={remote_name}")
    ready = wait_for_file_ready(client, remote_name, poll_interval, poll_timeout)
    file_uri = getattr(ready, "uri", "")
    print(f"[ready] file uri={file_uri}")

    try:
        response = generate_with_uri(
            client=client,
            model=model,
            prompt=prompt,
            file_uri=file_uri,
            mime_type=mime_type,
        )
    finally:
        maybe_delete_remote_file(client, remote_name, delete_remote)

    text = response_to_text(response)
    remote_meta = {
        "remote_file_name": remote_name,
        "remote_file_uri": file_uri,
        "mime_type": mime_type,
    }
    return text, remote_meta, response


def is_token_limit_error(exc: Exception) -> bool:
    """Check whether exception indicates model token limit overflow."""
    return TOKEN_LIMIT_MARKER in str(exc).lower()


def run_cmd(cmd: list[str]) -> None:
    """Run command and raise detailed error on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def has_avconvert() -> bool:
    """Return whether avconvert is available."""
    return shutil.which("avconvert") is not None


def get_video_duration_seconds(video_path: Path) -> int:
    """Get video duration in seconds with mdls."""
    result = subprocess.run(
        ["mdls", "-name", "kMDItemDurationSeconds", str(video_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read duration with mdls: {result.stderr}")
    match = re.search(r"=\s*([0-9.]+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse duration from mdls output: {result.stdout}")
    return int(float(match.group(1)) + 0.999)


def split_video_with_avconvert(
    video_path: Path,
    segment_seconds: int,
    segment_dir: Path,
    max_segments: int,
) -> list[SegmentInfo]:
    """Split video into M4A audio segments using avconvert."""
    duration = get_video_duration_seconds(video_path)
    segment_dir.mkdir(parents=True, exist_ok=True)

    segments: list[SegmentInfo] = []
    index = 0
    for offset in range(0, duration, segment_seconds):
        if max_segments > 0 and index >= max_segments:
            break
        current_duration = min(segment_seconds, duration - offset)
        output = segment_dir / f"segment_{index:03d}.m4a"
        cmd = [
            "avconvert",
            "--source",
            str(video_path),
            "--preset",
            "PresetAppleM4A",
            "--output",
            str(output),
            "--start",
            str(float(offset)),
            "--duration",
            str(float(current_duration)),
            "--replace",
        ]
        print(f"[split] creating segment {index:03d} offset={offset}s")
        run_cmd(cmd)
        segments.append(
            SegmentInfo(
                index=index,
                path=str(output),
                mime_type="audio/mp4",
                offset_seconds=offset,
                duration_seconds=current_duration,
            )
        )
        index += 1
    return segments


def list_existing_segments(
    segment_dir: Path,
    segment_seconds: int,
    max_segments: int,
) -> list[SegmentInfo]:
    """Load existing segment files from directory."""
    exts = {".mp3", ".m4a", ".wav", ".aac", ".mp4", ".mov"}
    files = [p for p in sorted(segment_dir.iterdir()) if p.suffix.lower() in exts]
    if max_segments > 0:
        files = files[:max_segments]
    if not files:
        raise RuntimeError(f"No segment files found in: {segment_dir}")

    segments: list[SegmentInfo] = []
    for idx, file_path in enumerate(files):
        segments.append(
            SegmentInfo(
                index=idx,
                path=str(file_path),
                mime_type=resolve_mime_type(file_path),
                offset_seconds=idx * segment_seconds,
                duration_seconds=segment_seconds,
            )
        )
    return segments


def run_segment_mode(
    client: genai.Client,
    args: argparse.Namespace,
    run_id: str,
    source_video_path: Path,
) -> tuple[Path, Path, Path]:
    """Run extraction over segments and write merged outputs."""
    if args.segment_dir:
        segments = list_existing_segments(
            segment_dir=args.segment_dir,
            segment_seconds=args.segment_seconds,
            max_segments=args.max_segments,
        )
        split_origin = "existing"
    else:
        if not has_avconvert():
            raise RuntimeError(
                "avconvert not found and no --segment-dir provided. "
                "Install ffmpeg or provide pre-split segment files."
            )
        segment_root = args.output_dir / "generated_segments"
        segments = split_video_with_avconvert(
            video_path=source_video_path,
            segment_seconds=args.segment_seconds,
            segment_dir=segment_root,
            max_segments=args.max_segments,
        )
        split_origin = "avconvert"

    merged_lines: list[str] = [
        f"# Gemini 分段提取结果",
        "",
        f"- 模型: `{args.model}`",
        f"- 分段来源: `{split_origin}`",
        f"- 分段数量: `{len(segments)}`",
        "",
    ]
    raw_payload: list[dict[str, Any]] = []

    segment_text_dir = args.output_dir / "segment_outputs"
    segment_text_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        segment_path = Path(segment.path)
        print(f"[segment] {segment.index:03d} => {segment_path.name}")
        prompt = build_segment_prompt(args.requirements, segment)
        text, remote_meta, response = run_one_media(
            client=client,
            model=args.model,
            media_path=segment_path,
            mime_type=segment.mime_type,
            prompt=prompt,
            poll_interval=args.poll_interval,
            poll_timeout=args.poll_timeout,
            delete_remote=args.delete_remote_file,
        )

        segment_md = segment_text_dir / f"segment_{segment.index:03d}.md"
        segment_md.write_text(text, encoding="utf-8")
        merged_lines.extend(
            [
                f"## Segment {segment.index:03d}",
                "",
                f"- 文件: `{segment_path.name}`",
                f"- 起始偏移(估计): `{segment.offset_seconds}` 秒",
                "",
                text,
                "",
            ]
        )
        raw_payload.append(
            {
                "segment": asdict(segment),
                "remote_meta": remote_meta,
                "response": response_to_mapping(response),
            }
        )

    merged_path = args.output_dir / f"{run_id}_transcript_segmented.md"
    raw_path = args.output_dir / f"{run_id}_raw_segment_responses.json"
    meta_path = args.output_dir / f"{run_id}_meta.json"

    metadata = ExtractionMetadata(
        run_id=run_id,
        mode="segments",
        model=args.model,
        source_video_path=str(source_video_path),
        prompt_requirements=args.requirements,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        segment_count=len(segments),
        fallback_triggered=False,
    )

    merged_path.write_text("\n".join(merged_lines), encoding="utf-8")
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return merged_path, meta_path, raw_path


def run_single_mode(
    client: genai.Client,
    args: argparse.Namespace,
    run_id: str,
    source_video_path: Path,
    fallback_triggered: bool,
) -> tuple[Path, Path, Path]:
    """Run extraction for a single full media file."""
    mime_type = resolve_mime_type(source_video_path)
    video_size = source_video_path.stat().st_size
    print(f"[start] single-file mode: {source_video_path} ({video_size / (1024**2):.1f} MB)")

    prompt = build_single_prompt(args.requirements)
    text, remote_meta, response = run_one_media(
        client=client,
        model=args.model,
        media_path=source_video_path,
        mime_type=mime_type,
        prompt=prompt,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
        delete_remote=args.delete_remote_file,
    )

    transcript_path = args.output_dir / f"{run_id}_transcript.md"
    raw_path = args.output_dir / f"{run_id}_raw_response.json"
    meta_path = args.output_dir / f"{run_id}_meta.json"

    metadata = ExtractionMetadata(
        run_id=run_id,
        mode="single",
        model=args.model,
        source_video_path=str(source_video_path),
        prompt_requirements=args.requirements,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        segment_count=1,
        fallback_triggered=fallback_triggered,
    )

    transcript_path.write_text(text, encoding="utf-8")
    raw_path.write_text(
        json.dumps(response_to_mapping(response), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    payload = {"metadata": asdict(metadata), "remote_meta": remote_meta}
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return transcript_path, meta_path, raw_path


def validate_paths(args: argparse.Namespace) -> None:
    """Validate user input paths."""
    if args.mode in {"auto", "single"}:
        if not args.video.exists() or not args.video.is_file():
            raise FileNotFoundError(f"Video not found: {args.video}")
    if args.segment_dir and not args.segment_dir.exists():
        raise FileNotFoundError(f"Segment directory not found: {args.segment_dir}")


def main() -> int:
    """Execute extraction workflow."""
    args = parse_args()
    load_env_file(args.env_file)
    validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.mode == "segments":
        merged_path, meta_path, raw_path = run_segment_mode(client, args, run_id, args.video)
        print(f"[done] merged transcript: {merged_path}")
        print(f"[done] metadata: {meta_path}")
        print(f"[done] raw responses: {raw_path}")
        return 0

    if args.mode == "single":
        transcript_path, meta_path, raw_path = run_single_mode(
            client, args, run_id, args.video, fallback_triggered=False
        )
        print(f"[done] transcript: {transcript_path}")
        print(f"[done] metadata: {meta_path}")
        print(f"[done] raw response: {raw_path}")
        return 0

    try:
        transcript_path, meta_path, raw_path = run_single_mode(
            client, args, run_id, args.video, fallback_triggered=False
        )
        print(f"[done] transcript: {transcript_path}")
        print(f"[done] metadata: {meta_path}")
        print(f"[done] raw response: {raw_path}")
        return 0
    except Exception as exc:
        if not is_token_limit_error(exc):
            raise
        print(f"[warn] single-file token limit hit: {exc}")
        print("[warn] fallback to segment mode.")

    merged_path, meta_path, raw_path = run_segment_mode(client, args, run_id, args.video)
    print(f"[done] merged transcript: {merged_path}")
    print(f"[done] metadata: {meta_path}")
    print(f"[done] raw responses: {raw_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level failure path
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
