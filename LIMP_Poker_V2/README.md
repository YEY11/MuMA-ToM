# MuMA-ToM 德州扑克多模态管线 (V2)

这是一个升级版的多模态多智能体感知与推理管线，专为德州扑克场景分析设计。

## 核心特性
- **操作级分析 (Action-Centric)**：突破了基于阶段快照的粗粒度限制，支持精细化的动作检测（下注、过牌、弃牌）及思考时长分析。
- **严格解耦 (Strict Decoupling)**：
    - **感知层 (Perception Layer)**：仅使用视觉信息（帧 -> VLM -> 游戏状态），确保推理的公平性。
    - **标注层 (Annotation Layer)**：利用音频解说生成 Ground Truth（包含上帝视角的底牌信息），仅用于评估。
- **触发式情感分析 (Triggered Emotion Analysis)**：仅在关键决策时刻触发微表情分析，捕捉稍纵即逝的心理信号。
- **全配置化管理**：通过 `.env` 和 `config.py` 统一管理所有参数。

## 目录结构
```
LIMP_Poker_V2/
├── config.py                 # 配置管理器
├── run_pipeline.py           # 程序主入口
├── core/                     # 数据结构定义 (Pydantic Schema)
├── perception/               # 感知模块 (仅视觉)
│   ├── pipeline.py           # 流程编排器
│   └── agents/
│       ├── phase_agent.py    # 阶段分割 (Pre-flop/Flop...)
│       ├── action_agent.py   # 动作检测逻辑
│       ├── board_agent.py    # VLM 牌面解析
│       └── face_agent.py     # 触发式表情分析
├── annotation/               # Ground Truth 生成模块
│   └── audio_gt_agent.py     # 音频解说分析
└── scripts/                  # 工具脚本
    └── preprocess_video.py   # 视频预处理 (ffmpeg)
```

## 环境设置
1. 确保项目根目录下已有配置好的 `.env` 文件。
2. 安装依赖：`ffmpeg` (系统级), `openai`, `loguru`, `pydantic`。

## 使用方法

### 一键运行全流程
```bash
# 确保在项目根目录 MuMA-ToM 下运行，以便正确加载模块
export PYTHONPATH=$PYTHONPATH:$(pwd)
python LIMP_Poker_V2/run_pipeline.py --video /path/to/game.mp4 --output_root ./my_dataset
```

### 输出产物
- `frames/`: 提取的帧序列 (30fps)
- `audio.wav`: 提取的音频文件
- `perception_output.json`: 统一的结构化感知结果，包含：
    - 游戏阶段时间轴 (Timeline)
    - 每个阶段内的动作序列 (Actions)
    - 关键动作的情感上下文 (Emotional Context)
    - (可选) 基于解说的 Ground Truth
