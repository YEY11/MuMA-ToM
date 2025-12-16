import os
import sys
import json
import subprocess
import shutil
import argparse
from pathlib import Path

# --- 配置区域 ---
# 视频抽帧率 (Frames Per Second)
FPS = 1

# 区域裁剪配置 (x, y, w, h) 
# TODO: 请根据你的视频 game1.mp4 的实际分辨率和 UI 布局修改这些坐标
# 可以使用画图工具打开一帧截图来测量坐标
REGIONS = {
    # 公共牌区域 (示例坐标，需修改)
    "board": (600, 300, 720, 200),  
    # 玩家1区域 (示例坐标，需修改)
    "p1_area": (100, 600, 300, 200), 
    # 玩家2区域 (示例坐标，需修改)
    "p2_area": (1500, 600, 300, 200) 
}

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)

def extract_media(video_path, output_dir):
    """提取视频帧和音频"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    audio_path = output_dir / "audio.wav"
    
    if frames_dir.exists():
        # 如果已存在，询问是否覆盖或直接覆盖
        # 这里为了安全起见，先清空
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    
    print(f"[Media] Extracting frames to {frames_dir}...")
    # 提取帧: -q:v 2 表示高质量 JPEG
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={FPS}",
        "-q:v", "2",
        str(frames_dir / "frame_%04d.jpg"),
        "-loglevel", "error"
    ], check=True)
    
    print(f"[Media] Extracting audio to {audio_path}...")
    # 提取音频: 单声道 16kHz (适合 ASR)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
        "-loglevel", "error"
    ], check=True)
    
    return frames_dir

def extract_crops(frames_dir, output_dir):
    """(可选) 提取特定区域的裁剪图"""
    crops_dir = Path(output_dir) / "crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir()
    
    print(f"[Crops] Extracting regions to {crops_dir}...")
    
    try:
        from PIL import Image
    except ImportError:
        print("Warning: PIL (Pillow) not found. Skipping crop generation.")
        print("Tip: pip install Pillow")
        return

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        return

    # 遍历所有帧进行裁剪
    for f in frame_files:
        # 为了演示速度，这里每帧都裁。如果太慢可以加 if i % 5 == 0
        try:
            with Image.open(f) as img:
                for name, (x, y, w, h) in REGIONS.items():
                    # 边界检查
                    if x + w > img.width or y + h > img.height:
                        # 仅警告一次
                        if f == frame_files[0]:
                            print(f"Warning: Region {name} out of bounds. Skipping.")
                        continue
                        
                    crop = img.crop((x, y, x+w, y+h))
                    
                    # 保持目录结构: crops/board/frame_0000.jpg
                    region_dir = crops_dir / name
                    region_dir.mkdir(exist_ok=True)
                    crop.save(region_dir / f.name)
        except Exception as e:
            print(f"Error cropping {f.name}: {e}")

def generate_template_files(output_dir, video_name):
    """生成需要人工标注的 JSON 模板"""
    output_dir = Path(output_dir)
    
    # 1. Meta.json 模板 - 纯 Ground Truth
    meta_template = {
        "game_id": video_name,
        "video_info": {
            "fps": FPS,
            "original_file": f"{video_name}.mp4"
        },
        "players": [
            {"seat": 1, "name": "P1", "is_hero": False}, 
            {"seat": 2, "name": "P2", "is_hero": False}
        ],
        # 盲注信息
        "blinds": {
            "SB_seat": "MANUAL_FILL (1 or 2)",
            "BB_seat": "MANUAL_FILL (1 or 2)",
            "SB_amount": 0.5,
            "BB_amount": 1.0
        },
        # 真实底牌 (Ground Truth)，用于最后验证
        "ground_truth_cards": {
            "P1": ["MANUAL_FILL (e.g. Ah, Kd)", "MANUAL_FILL"],
            "P2": ["MANUAL_FILL", "MANUAL_FILL"],
            "winner_seat": "MANUAL_FILL"
        }
    }
    
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_template, f, indent=2, ensure_ascii=False)
        
    # 2. Labels.json 模板 - 用于 ToM 评估的标签
    labels_template = {
        "events": [
            # 记录关键事件发生的时间帧，用于评估 Agent 是否识别到了状态变化
            {
                "stage": "Preflop",
                "start_frame": 0,
                "end_frame": "MANUAL_FILL",
                "description": "Preflop action"
            },
            {
                "stage": "Flop",
                "start_frame": "MANUAL_FILL",
                "board_cards": ["MANUAL_FILL", "MANUAL_FILL", "MANUAL_FILL"],
                "description": "Flop dealt"
            },
            {
                "stage": "Turn",
                "start_frame": "MANUAL_FILL",
                "board_cards": ["MANUAL_FILL"],
                "description": "Turn dealt"
            },
            {
                "stage": "River",
                "start_frame": "MANUAL_FILL",
                "board_cards": ["MANUAL_FILL"],
                "description": "River dealt"
            }
        ],
        # ToM 核心问题：在特定时刻，Agent 对意图的判断是否准确
        "tom_questions": [
            {
                "id": "Q1",
                "frame_id": "MANUAL_FILL (关键决策帧, e.g., frame_0050.jpg)",
                "question": "P1 此时的下注行为意图是什么？",
                "options": ["诈唬 (Bluff)", "价值 (Value)"],
                "ground_truth": "MANUAL_FILL (Bluff/Value)",
                "reasoning": "MANUAL_FILL (简述理由，例如：底牌弱但下注大)"
            }
        ]
    }
    
    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(labels_template, f, indent=2, ensure_ascii=False)
        
    print(f"[Template] Generated meta.json and ground_truth.json in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="构建扑克推理数据集 (仅原始数据 + GT模板)")
    parser.add_argument("--video", required=True, help="原始视频路径")
    parser.add_argument("--output", required=True, help="输出根目录")
    parser.add_argument("--crop", action="store_true", help="是否开启区域裁剪 (需确保代码中 REGIONS 坐标正确)")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found.")
        sys.exit(1)
        
    check_ffmpeg()
    
    game_name = video_path.stem
    # 这里的 output 是根目录，我们会在下面创建 game_name 子目录
    # 但如果用户指定的 output 结尾已经是 game_name，则直接使用
    if Path(args.output).name == game_name:
        target_dir = Path(args.output)
    else:
        target_dir = Path(args.output) / game_name
    
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        
    print(f"Processing {game_name} -> {target_dir}")
    
    # 1. 提取原始素材
    frames_dir = extract_media(video_path, target_dir)
    
    # 2. (可选) 提取裁剪区域
    # 逻辑：命令行 --crop 参数 OR 环境变量 POKER_CROP_ENABLED=true
    env_crop = os.getenv("POKER_CROP_ENABLED", "false").lower() == "true"
    
    if args.crop or env_crop:
        extract_crops(frames_dir, target_dir)
    else:
        print("[Crops] Skipped. (Enable via --crop or POKER_CROP_ENABLED=true)")
        
    # 3. 生成标注模板
    generate_template_files(target_dir, game_name)
    
    print("\nDone! 数据集构建完成。")
    print(f"1. 请查看 {target_dir}/frames 确认帧提取质量。")
    print(f"2. 请人工填写 {target_dir}/meta.json 和 {target_dir}/ground_truth.json 作为标准答案。")

if __name__ == "__main__":
    main()
