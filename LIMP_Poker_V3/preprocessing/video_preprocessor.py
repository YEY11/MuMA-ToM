"""
Video Preprocessing Module
Extracts frames and audio from poker game videos
"""

import os
import subprocess
from loguru import logger
from LIMP_Poker_V3.config import config


def preprocess_video(
    video_path: str,
    output_dir: str,
    fps: int = None,
    extract_audio: bool = True,
) -> dict:
    """
    Extract frames and audio from a video file using ffmpeg.

    Args:
        video_path: Path to the input MP4 file
        output_dir: Directory to save extracted data
        fps: Frames per second for extraction (default: config.FPS)
        extract_audio: Whether to extract audio track

    Returns:
        dict with paths to extracted data
    """
    if fps is None:
        fps = config.FPS

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return {"success": False, "error": "Video file not found"}

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, "audio.wav")

    logger.info(f"Processing {video_name} -> {output_dir}")

    result = {
        "success": True,
        "video_name": video_name,
        "frames_dir": frames_dir,
        "audio_path": None,
        "frame_count": 0,
    }

    # 1. Extract Frames
    logger.info(f"Extracting frames at {fps} fps...")
    cmd_frames = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        os.path.join(frames_dir, "frame_%04d.jpg"),
        "-y",  # Overwrite
        "-loglevel",
        "error",
    ]

    try:
        subprocess.run(cmd_frames, check=True)
        # Count extracted frames
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
        result["frame_count"] = len(frame_files)
        logger.info(f"Extracted {result['frame_count']} frames successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract frames: {e}")
        result["success"] = False
        result["error"] = str(e)
        return result

    # 2. Extract Audio (for Ground Truth generation)
    if extract_audio:
        logger.info("Extracting audio (16kHz mono wav)...")
        cmd_audio = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",  # 16kHz
            "-ac",
            "1",  # Mono
            audio_path,
            "-y",
            "-loglevel",
            "error",
        ]

        try:
            subprocess.run(cmd_audio, check=True)
            result["audio_path"] = audio_path
            logger.info("Audio extracted successfully.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to extract audio: {e}")
            # Audio extraction failure is not fatal

    logger.info(f"Preprocessing complete for {video_name}")
    return result


def get_frame_paths(frames_dir: str) -> list:
    """
    Get sorted list of frame paths.

    Args:
        frames_dir: Directory containing extracted frames

    Returns:
        List of frame file paths, sorted by frame number
    """
    if not os.path.exists(frames_dir):
        return []

    frames = [
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".jpg")
    ]
    return sorted(frames)


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or 0 if failed
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to get video duration: {e}")
        return 0.0


if __name__ == "__main__":
    # Test usage
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess poker video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")

    args = parser.parse_args()

    result = preprocess_video(args.video, args.output, args.fps)
    print(f"Result: {result}")

