#!/bin/bash

# 1. 检查并安装 ffmpeg (如果需要)
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg 未找到，尝试通过 conda 安装..."
    conda install -y ffmpeg
    # 或者使用: sudo apt-get install -y ffmpeg
fi

# 2. 安装 Python 依赖
echo "正在安装 Python 依赖..."
# 只需要基础库和 Pillow (可选，用于裁剪)
pip install Pillow

# 3. 运行转换脚本
# 示例：将 episode_example/game1.mp4 转换为数据集
VIDEO_PATH="/data/nvme0/yy/vt/MuMA-ToM/datasets/poker/episode_example/game1.mp4"
OUTPUT_DIR="/data/nvme0/yy/vt/MuMA-ToM/datasets/poker/episode_example/"

echo "开始处理视频: $VIDEO_PATH"
echo "输出目录: $OUTPUT_DIR"

# 默认不开启 crop，如需开启可添加 --crop 或设置环境变量 POKER_CROP_ENABLED=true
# export POKER_CROP_ENABLED=true
python build_dataset.py --video "$VIDEO_PATH" --output "$OUTPUT_DIR"
