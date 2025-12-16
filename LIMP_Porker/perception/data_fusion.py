import os
import json
import argparse
from loguru import logger
# from PIL import Image # No longer needed for on-the-fly crops
from agents.phase_segmentation_agent import PhaseSegmentationAgent
from agents.board_parsing_agent import BoardParsingAgent
from agents.audio_analysis_agent import AudioAnalysisAgent
from agents.face_emotion_agent import FaceEmotionAgent

def run_perception_pipeline(episode_dir: str, output_path: str):
    logger.info(f"Starting perception pipeline for {episode_dir}")
    
    frames_dir = os.path.join(episode_dir, "frames")
    audio_path = os.path.join(episode_dir, "audio.wav")
    
    # 1. Phase Segmentation
    phase_agent = PhaseSegmentationAgent()
    phases = phase_agent.detect_phases(frames_dir)
    logger.info(f"Phases detected: {phases}")
    
    # 2. Per-Phase Board & Face Analysis
    board_agent = BoardParsingAgent()
    face_agent = FaceEmotionAgent()
    
    # 预先提取一次 Pre-flop 的状态以获取玩家名字 (用于 Audio Entity Resolution)
    known_players = []
    first_phase_name = list(phases.keys())[0] if phases else None
    if first_phase_name:
        first_phase_data = phases[first_phase_name]
        mid_frame_idx = (first_phase_data["start"] + first_phase_data["end"]) // 2
        mid_frame_path = os.path.join(frames_dir, f"frame_{mid_frame_idx:04d}.jpg")
        if os.path.exists(mid_frame_path):
            logger.info("Extracting player names for entity resolution...")
            initial_state = board_agent.parse_game_state(mid_frame_path)
            known_players = [p.get("name", "Unknown") for p in initial_state.get("players", [])]
            logger.info(f"Identified players from UI: {known_players}")

    # 3. Audio Analysis (with known players)
    audio_agent = AudioAnalysisAgent()
    full_transcript = audio_agent.transcribe(audio_path)
    audio_insights = audio_agent.analyze_commentary(full_transcript, known_players=known_players)
    logger.info("Audio analysis complete")
    
    unified_state = {
        "meta": {"episode": os.path.basename(episode_dir)},
        "phases": phases,
        "transcript": full_transcript,
        "commentary_insights": audio_insights,
        "timeline": []
    }
    
    # 4. Detailed Per-Phase Analysis
    for phase_name, times in phases.items():
        # 取中间帧作为代表
        mid_frame_idx = (times["start"] + times["end"]) // 2
        mid_frame_name = f"frame_{mid_frame_idx:04d}.jpg"
        mid_frame_path = os.path.join(frames_dir, mid_frame_name)
        
        phase_data = {
            "phase": phase_name,
            "frame_idx": mid_frame_idx,
            "game_state": {},
            "scene_emotions": []
        }
        
        if os.path.exists(mid_frame_path):
            # A. 提取完整游戏状态 (Board, Stacks, Blinds, etc.)
            logger.info(f"Parsing game state for {phase_name}...")
            phase_data["game_state"] = board_agent.parse_game_state(mid_frame_path)
            
            # B. 全图表情分析 (Dynamic Scene Analysis)
            logger.info(f"Analyzing scene emotions for {phase_name}...")
            scene_result = face_agent.analyze_scene(mid_frame_path)
            phase_data["scene_emotions"] = scene_result.get("players", [])
        
        unified_state["timeline"].append(phase_data)
        logger.info(f"Processed phase {phase_name}")

    # Save
    with open(output_path, "w") as f:
        json.dump(unified_state, f, indent=2, ensure_ascii=False)
    logger.info(f"Pipeline finished. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", required=True, help="Path to episode directory (containing frames/ and audio.wav)")
    parser.add_argument("--output", default="unified_perception.json", help="Output JSON path")
    args = parser.parse_args()
    
    run_perception_pipeline(args.episode_dir, args.output)
