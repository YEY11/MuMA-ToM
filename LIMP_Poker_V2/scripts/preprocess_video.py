import os
import argparse
import subprocess
from loguru import logger
from LIMP_Poker_V2.config import config


def preprocess_video(video_path: str, output_dir: str, fps: int = None):
    """
    Extracts frames and audio from a video file.

    Args:
        video_path: Path to the input MP4 file.
        output_dir: Directory to save extracted data (will be created).
        fps: Frames per second for extraction. If None, uses config default.
    """
    if fps is None:
        fps = config.FPS
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, "audio.wav")

    logger.info(f"Processing {video_name} -> {output_dir}")

    # 1. Extract Frames
    # frame_%04d.jpg pattern
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
        "error",  # Quiet
    ]

    try:
        subprocess.run(cmd_frames, check=True)
        logger.info("Frames extracted successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract frames: {e}")
        return

    # 2. Extract Audio
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
        logger.info("Audio extracted successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e}")
        return

    logger.info(f"Preprocessing complete for {video_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess poker video for MuMA-ToM pipeline"
    )
    parser.add_argument("--video", required=True, help="Path to input video file (mp4)")
    parser.add_argument(
        "--output", required=True, help="Output directory for this episode"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second extraction rate"
    )

    args = parser.parse_args()

    preprocess_video(args.video, args.output, args.fps)
