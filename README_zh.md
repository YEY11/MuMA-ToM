# MuMA-ToM：多模态多智能体心智理论  <br> <sub> 🚀 AAAI 2025 口头报告 🎤 </sub>
### [论文](https://arxiv.org/abs/2408.12574) | [项目页面](https://scai.cs.jhu.edu/projects/MuMA-ToM/) | [数据集](https://huggingface.co/datasets/SCAI-JHU/MUMA-TOM-BENCHMARK)<br><br>
![intro](figures/question_types.png)

本仓库包含论文 [**MuMA-ToM：多模态多智能体心智理论**](https://arxiv.org/abs/2408.12574) 的代码。

包含内容：
* 使用 MuMA-ToM 基准的说明
* LIMP 模型的实现与使用指南
* 数据的程序化生成代码

## 基于语言模型的逆向多智能体规划（LIMP）
我们提出了基于语言模型的逆向多智能体规划（LIMP），这是一种解决多模态和多智能体心智理论推理的新方法。

要在 MuMA-ToM 基准上运行 LIMP，请在相关文件中填入你的 GPT API 密钥。我们在所有任务中使用 GPT-4o。

视觉信息提取方面，我们在 Google AI Studio 中使用 Gemini 1.5 Pro 的网页版，因为它比 API 版本更强大。

对于视觉动作提取，请将每个视频上传到 Google AI Studio。“Files” 文件夹下的 “actions_extracted.json” 包含我们对每个（按 id）片段使用的提示词。将对应视频上传到 Google AI Studio，并把输出填入该 json 文件中每条目下的 “actions” 字段。

随后，直接运行 LIMP.py。

## MuMA-ToM 基准
MuMA-ToM 基准托管在 Hugging Face。链接在[这里](https://huggingface.co/datasets/SCAI-JHU/MUMA-TOM-BENCHMARK/tree/main)。

在数据集中，“questions.json” 和 “texts.json” 包含了我们基准的提问文本与多模态文本输入。“Videos” 文件夹包含所有 RGB 视频。“full episode descriptions” 文件夹包含由 GPT 生成的交互场景描述以及对应的真实动作和话语。

我们还生成了一个包含一千段视频的训练集，覆盖家庭场景中的多智能体交互。训练集存放于 “training_set” 文件夹，并附有智能体动作标注。

如果你需要实例分割和深度图像以进行进一步实验，请联系我们。用于通过实例分割生成场景图的视觉分析结果，存储在 “visual data” 文件夹中。

## 引用
如果你觉得这个工作有趣/有用，请引用论文并为本仓库点星标，谢谢！

```bibtex
@article{shi2024muma,
  title={MuMA-ToM: Multi-modal Multi-Agent Theory of Mind},
  author={Shi, Haojun and Ye, Suyu and Fang, Xinyu and Jin, Chuanyang and Isik, Leyla and Kuo, Yen-Ling and Shu, Tianmin},
  journal={arXiv preprint arXiv:2408.12574},
  year={2024}
}
```