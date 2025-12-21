# LIMP_Poker_V3

多模态多Agent心智理论(ToM)推理框架 - 德州扑克场景

## 📁 项目结构

```
LIMP_Poker_V3/
├── __init__.py
├── config.py                     # 配置管理（含Agent开关）
├── main.py                       # 主入口
│
├── core/                         # 核心基础设施
│   ├── schema.py                 # Pydantic数据结构
│   └── registry.py               # Agent注册表（可插拔）
│
├── preprocessing/                # 预处理模块
│   └── video_preprocessor.py     # 视频抽帧+音频提取
│
├── perception/                   # 感知层
│   ├── pipeline.py               # 感知流程编排
│   └── agents/
│       ├── base.py               # Agent基类
│       ├── board_agent.py        # VLM盘面解析
│       └── action_detector.py    # 动作检测
│
├── annotation/                   # 标注模块
│   └── audio_gt_agent.py         # 音频GT提取（仅用于标注）
│
├── dataset/                      # 数据集生成
│   ├── qa_generator.py           # QA自动生成
│   └── templates/
│       ├── action_level.py       # 操作级问题模板
│       └── phase_level.py        # 阶段级问题模板
│
├── reasoning/                    # 推理层
│   ├── pipeline.py               # 推理流程编排
│   └── agents/
│       ├── base.py               # Agent基类
│       ├── posture_agent.py      # 微姿态分析
│       ├── equity_agent.py       # 胜率计算
│       ├── tom_belief_agent.py   # ToM信念推理
│       └── tom_social_agent.py   # ToM社会目标推理
│
├── evaluation/                   # 评估模块
│   └── metrics.py                # 评估指标
│
├── prompts/                      # 提示词模板
│   └── board_parsing.txt
│
└── scripts/                      # 工具脚本
    ├── batch_process.py          # 批量处理
    └── run_ablation.py           # 消融实验
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env，填入你的API Key
vim .env
```

### 2. 运行完整流程

```bash
# 处理单个视频
python -m LIMP_Poker_V3.main \
    --video datasets/poker_v2/game1.mp4 \
    --output datasets/processed_v3

# 仅运行推理（跳过预处理和感知）
python -m LIMP_Poker_V3.main \
    --video datasets/poker_v2/game1.mp4 \
    --output datasets/processed_v3 \
    --skip-preprocess --skip-perception
```

### 3. 批量处理

```bash
python -m LIMP_Poker_V3.scripts.batch_process \
    --video-dir datasets/poker_videos \
    --output datasets/processed_v3
```

### 4. 消融实验

```bash
# 列出可用配置
python -m LIMP_Poker_V3.scripts.run_ablation --list-configs

# 运行消融实验
python -m LIMP_Poker_V3.scripts.run_ablation \
    --episode-dir datasets/processed_v3/game1 \
    --output ablation_results.json
```

## ⚙️ 配置说明

### 视角模式

通过 `PROTOCOL_MODE` 环境变量切换：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `audience` | 双方底牌可见 | 观众视角，完备信息博弈 |
| `player` | 对手底牌不可见 | 玩家视角，不完备信息博弈 |

### Agent配置

可通过环境变量控制每个Agent的启用状态：

```bash
# 禁用姿态分析Agent（用于消融实验）
AGENT_POSTURE=False
```

## 📊 数据格式

### QA数据集格式

```json
{
  "episode_id": "game1",
  "protocol": "audience",
  "questions": [
    {
      "id": "game1_act_001",
      "level": "action",
      "question_type": "intent",
      "question": "Hellmuth 的这次 raise $50,000 最可能的意图是什么？",
      "options": [
        {"key": "A", "text": "Bluff（诈唬）..."},
        {"key": "B", "text": "Value（价值）..."},
        {"key": "C", "text": "Control（控池）..."}
      ],
      "answer": "A",
      "tom_labels": {
        "social_goal": "bluff"
      }
    }
  ]
}
```

## 🔧 扩展

### 添加新的推理Agent

1. 在 `reasoning/agents/` 创建新文件
2. 继承 `BaseReasoningAgent`
3. 使用 `@AgentRegistry.register_reasoning("agent_name")` 装饰器注册
4. 在 `config.py` 的 `AGENT_CONFIG` 中添加开关

```python
from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.reasoning.agents.base import BaseReasoningAgent

@AgentRegistry.register_reasoning("my_agent")
class MyAgent(BaseReasoningAgent):
    def analyze(self, question, perception_data, **kwargs):
        # 你的推理逻辑
        pass
```

## 📝 许可证

MIT License

