# 数据集构建工具

本工具用于将扑克比赛视频转换为 MuMA-ToM 评测所需的标准数据集格式。
它专注于提取原始多模态数据（视频帧、音频），并生成用于人工标注的 Ground Truth 模板。

## 依赖
* Python 3.8+
* FFmpeg (必须安装并在 PATH 中)
* Pillow (可选，用于裁剪区域): `pip install Pillow`

## 使用方法

### 1. 快速开始 (自动安装依赖并运行)
```bash
bash setup_and_run.sh
```

### 2. 手动运行
```bash
# 基础用法 (仅提取全图和音频)
python build_dataset.py --video /path/to/game1.mp4 --output /path/to/dataset_root/
```

## 输出结构

脚本运行后，会在输出目录下生成：

*   `frames/`: 原始视频帧（1秒1帧），这是 Agent 的视觉输入。
*   `audio.wav`: 原始音频，这是 Agent 的听觉输入。
*   `crops/`: (如果开启 --crop) 关键区域的裁剪图，可辅助 Agent 聚焦。
*   `meta.json`: **[需人工填写]** 包含底牌、盲注位置等上帝视角信息。
*   `ground_truth.json`: **[需人工填写]** 包含关键事件（翻牌、转牌）的时间点和 ToM 问题标签。

## 为什么不自动提取 JSON？

为了保证评测的公正性，Agent 应当直接面对原始的图像和音频数据，自行完成感知（Perception）和推理（Reasoning）。数据集提供的 JSON 仅作为 Ground Truth（标准答案）用于计算 Agent 的准确率，而不应作为输入喂给 Agent。
