# Agentic R1 — 基于 GRPO 强化学习的智能转账 Agent

使用**单张 4090D GPU**，用 **GRPO + LoRA** 微调 Qwen2.5-3B-Instruct，让模型学会在 `<think>` 里推理（汇率换算、金额核对），再输出 `<tool_call>` JSON 执行转账。

## 核心能力

- **R1 格式**：`<think>...</think><tool_call>...</tool_call>`
- **拼音转换**：收款人中文名 → 拼音全拼
- **金额规范化**：统一 `¥xxx.xx` 格式输出
- **美元换算**：1 USD = 7 CNY，自动折算
- **负样本**：余额查询、转账记录等闲聊不调用工具
- **消歧**：处理同姓不同人（张伟 vs 张三）

## 项目结构

```
.
├── config.py      # 全局配置（提示词、姓名映射、奖励权重、训练/推理参数）
├── dataset.py     # 训练数据生成（多模板 × 多人名 × 多金额格式）
├── train.py       # GRPO 训练主脚本（含四种奖励函数 + 周期性评估）
├── export.py      # LoRA 权重合并导出（16-bit 模型）
├── test.py        # 模型评测（基座 vs 微调对比报告）
└── utils.py       # 工具函数（解析、提取、指标计算）
```

## 环境要求

- Python 3.10+
- CUDA GPU（推荐 24GB+ 显存）
- [unsloth](https://github.com/unslothai/unsloth)
- [trl](https://github.com/huggingface/trl)（含 GRPOTrainer）
- transformers、torch、wandb（可选）

## 安装

```bash
git clone <repo-url>
cd agentic_r1

pip install -r requirements.txt

# 下载基座模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ./Qwen2.5-3B-Instruct-Local
```

## 运行方式

### 1. 训练

```bash
python train.py
```

训练输出在 `results/Agentic_R1_Lora_<timestamp>/` 下：
- `checkpoints/best_model/` — 评估集上最优的 LoRA 权重
- `checkpoints/final_model/` — 训练结束时的权重
- `training_metrics.csv` — 每步 reward / KL / loss / grad_norm
- `eval_metrics.jsonl` — 周期性评估记录
- `completions_samples/` — 模型输出采样

可调的超参（`config.py` → `GRPO_TRAINING_CONFIG`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | 1500 | 训练步数 |
| `learning_rate` | 5e-6 | 学习率 |
| `per_device_train_batch_size` | 8 | 单卡 batch size |
| `num_generations` | 8 | 每组生成样本数 |
| `beta` | 0.06 | KL 惩罚系数 |
| `max_completion_length` | 768 | 最大回答长度 |

### 2. 导出模型

```bash
# 导出最优 checkpoint
python export.py best

# 导出最终 checkpoint
python export.py final

# 指定训练输出目录
python export.py Agentic_R1_Lora_20250101_120000 best
```

合并后的模型默认保存为 16-bit 格式，可直接用于推理。

### 3. 评测

```bash
# 自动使用最新训练结果，对比基座 vs 微调模型
python test.py

# 指定模型目录
python test.py Agentic_R1_Lora_20250101_120000
```

测试集覆盖全部人名、CNY/USD 双币种、整数/小数金额，输出按人统计的通过率。

## 奖励函数设计

四种加权奖励：

| 奖励 | 权重 | 逻辑 |
|------|------|------|
| **格式** | 0.15 | 正样本无 `<think>` + `<tool_call>` 则为 0；负样本输出 tool_call 则为 0 |
| **拼音** | 0.35 | 编辑距离相似度，完全匹配 1.0，偏差大则衰减 |
| **金额** | 0.25 | 相对误差 <1% 满分，<10% 线性衰减，超过即 0 |
| **推理** | 0.25 | think 分档评分（无/有/算错/缺汇率/全对），CNY 误 ×7 封顶 0.3 |

## 主要结果

评测在含 10+ 个人名、CNY/USD 双币种的测试集上进行。微调后的模型相比基座有明显提升：整体通过率从35.4%（base）提升至67.1%（+31.7%）。其中格式正确率达99.6%，货币兑换金额正确率达90%。

