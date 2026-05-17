import os
import csv
import json
import torch
import re
import logging
from datetime import datetime

# 抑制 transformers generation/config 模块中 logger.warning_once 的无害告警
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from transformers import is_wandb_available
from transformers.trainer_callback import ProgressCallback
from dataset import generate_training_data
from config import (
    PINYIN_MAP,
    REWARD_WEIGHTS,
    REGEX_CONFIG,
    TRAIN_MODEL_CONFIG,
    LORA_CONFIG,
    GRPO_TRAINING_CONFIG,
    TEST_CONFIG,
    USD_TO_CNY_RATE,
    RESULTS_BASE_DIR,
    SYSTEM_PROMPT,
)
from test import compute_metrics
from utils import (
    extract_tool_data,
    normalize_cny_amount,
    extract_name_from_query,
    extract_query_amount,
    has_valid_r1_format,
    levenshtein_ratio,
    is_usd_query,
    save_code_snapshot,
    KLLogger,
    get_checkpoint_dir,
    get_best_ckpt_path,
    get_final_ckpt_path,
    save_peft_checkpoint,
    extract_user_query,
    is_negative_prompt,
    format_cny_value,
    has_calc_signal,
    extract_amount_numbers,
    expected_answer,
    amounts_equal,
    iter_prompt_completion,
)

if is_wandb_available():
    import wandb

# ==========================================
# 生成训练数据集
# ==========================================
FORMAT_REWARD_WEIGHT = REWARD_WEIGHTS["format"]
PINYIN_REWARD_WEIGHT = REWARD_WEIGHTS["pinyin"]
AMOUNT_REWARD_WEIGHT = REWARD_WEIGHTS["amount"]
SELF_CORRECTION_REWARD_WEIGHT = REWARD_WEIGHTS["self_correction"]


# =========================================================
# 奖励函数 1：R1 格式校验（保持二值，权重降低）
# =========================================================
def format_reward_func(prompts, completions, **kwargs):
    """判断每个 completion 是否符合 R1 格式要求。

    正样本：要求 <think> + <tool_call> JSON。
    负样本（无转账意图）：正确行为是不输出 tool_call，得满分。
    """
    rewards = []
    logger = logging.getLogger(__name__)

    for prompt, comp in iter_prompt_completion(prompts, completions, logger):
        try:
            if is_negative_prompt(prompt):
                rewards.append(0.0 if has_valid_r1_format(comp) else 1.0)
            else:
                rewards.append(1.0 if has_valid_r1_format(comp) else 0.0)
        except Exception as e:
            logger.warning(f"format_reward_func error: {e}")
            rewards.append(0.0)

    return rewards


# =========================================================
# 奖励函数 2：拼音部分匹配奖励（平滑梯度）
# =========================================================
def pinyin_reward_func(prompts, completions, **kwargs):
    """对拼音进行编辑距离归一化的部分匹配奖励。

    消除原先"全有或全无"的二值信号，提供平滑梯度。
    负样本：正确行为是不输出 tool_call，得满分 1.0。
    """
    rewards = []
    logger = logging.getLogger(__name__)

    for prompt, comp in iter_prompt_completion(prompts, completions, logger):
        try:
            if is_negative_prompt(prompt):
                rewards.append(0.0 if has_valid_r1_format(comp) else 1.0)
                continue

            if not has_valid_r1_format(comp):
                rewards.append(0.0)
                continue

            user_query = extract_user_query(prompt)
            if not user_query:
                rewards.append(0.0)
                continue

            user_name = extract_name_from_query(user_query, PINYIN_MAP)
            if not user_name:
                rewards.append(0.0)
                continue

            expected_pinyin = PINYIN_MAP[user_name].lower()
            params = extract_tool_data(comp)
            if params is None:
                rewards.append(0.0)
                continue

            actual_pinyin = (params.get("target_account") or "").strip().lower()
            if not actual_pinyin:
                rewards.append(0.0)
                continue

            # 编辑距离归一化相似度
            sim = levenshtein_ratio(actual_pinyin, expected_pinyin)
            # 完全匹配直接满分；部分匹配按相似度给分但设下限 0.1
            if sim >= 0.999:
                rewards.append(1.0)
            elif sim >= 0.5:
                rewards.append(sim)
            else:
                rewards.append(0.1 * sim)  # 太低不给有效信号

        except Exception as e:
            logger.warning(f"pinyin_reward_func error: {e}")
            rewards.append(0.0)

    return rewards


# =========================================================
# 奖励函数 3：金额近似匹配奖励（平滑梯度）
# =========================================================
def amount_reward_func(prompts, completions, **kwargs):
    """对最终金额进行相对误差归一化的部分匹配奖励。

    误差 < 1% → 满分 1.0，< 10% → 线性衰减，>= 10% → 0。
    负样本：正确行为是不输出 tool_call，得满分 1.0。
    """
    rewards = []
    logger = logging.getLogger(__name__)

    for prompt, comp in iter_prompt_completion(prompts, completions, logger):
        try:
            if is_negative_prompt(prompt):
                rewards.append(0.0 if has_valid_r1_format(comp) else 1.0)
                continue

            if not has_valid_r1_format(comp):
                rewards.append(0.0)
                continue

            user_query = extract_user_query(prompt)
            if not user_query:
                rewards.append(0.0)
                continue

            user_amount = extract_query_amount(user_query)
            if user_amount is None:
                rewards.append(0.0)
                continue

            is_usd = is_usd_query(user_query)
            expected_value = round(user_amount * USD_TO_CNY_RATE, 2) if is_usd else round(user_amount, 2)
            expected_str = format_cny_value(expected_value)

            params = extract_tool_data(comp)
            if params is None:
                rewards.append(0.0)
                continue

            expected_normalized = normalize_cny_amount(expected_str)
            actual_normalized = normalize_cny_amount(params.get("amount"))

            if not expected_normalized or not actual_normalized:
                rewards.append(0.0)
                continue

            # 从归一化字符串中提取数值
            exp_match = re.search(r"(\d+(?:\.\d+)?)", expected_normalized)
            act_match = re.search(r"(\d+(?:\.\d+)?)", actual_normalized)
            if not exp_match or not act_match:
                rewards.append(0.0)
                continue

            exp_val = float(exp_match.group(1))
            act_val = float(act_match.group(1))
            if exp_val <= 0:
                rewards.append(0.0)
                continue

            error = abs(act_val - exp_val) / exp_val
            if error < 0.01:
                rewards.append(1.0)
            elif error < 0.10:
                rewards.append(1.0 - error / 0.10)  # 线性衰减
            else:
                rewards.append(0.0)

        except Exception as e:
            logger.warning(f"amount_reward_func error: {e}")
            rewards.append(0.0)

    return rewards


# =========================================================
# 奖励函数 4：思路过程校验（阶梯评分 + CNY 过度换算惩罚）
# =========================================================
def self_correction_reward_func(prompts, completions, **kwargs):
    """阶梯式评估 <think> 区块质量，并对 CNY 的过度换算施加惩罚。

    评分阶梯（正样本）：
      - 无 think → 0.0
      - 有 think 无计算信号 → 0.2
      - 有计算信号但结果错误 → 0.4
      - 有计算信号且结果正确（USD 缺汇率提示）→ 0.7
      - 完整且正确 → 1.0

    CNY 过度换算惩罚：若 CNY 查询的 think 中出现汇率/USD 标记，
    结果正确也降为最高 0.3（模型学会了"见到金额就 ×7"的偏差）。
    """
    rewards = []
    logger = logging.getLogger(__name__)

    for prompt, comp in iter_prompt_completion(prompts, completions, logger):
        try:
            if is_negative_prompt(prompt):
                rewards.append(0.0 if has_valid_r1_format(comp) else 1.0)
                continue

            if not has_valid_r1_format(comp):
                rewards.append(0.0)
                continue

            user_query = extract_user_query(prompt)
            if not user_query:
                rewards.append(0.0)
                continue

            user_amount = extract_query_amount(user_query)
            if user_amount is None:
                rewards.append(0.0)
                continue

            is_usd = is_usd_query(user_query)
            expected_value = user_amount * USD_TO_CNY_RATE if is_usd else user_amount

            if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[-1], dict):
                comp_str = comp[-1].get("content", "")
            else:
                comp_str = str(comp)

            think_match = re.search(REGEX_CONFIG["think_block_pattern"], comp_str, re.DOTALL)
            if not think_match:
                rewards.append(0.0)
                continue

            thought = think_match.group(1)
            amount_nums = extract_amount_numbers(thought)
            if not amount_nums:
                amount_nums = re.findall(r"(\d+(?:\.\d+)?)", thought)

            has_calc = has_calc_signal(thought)
            has_result = any(abs(float(num) - float(expected_value)) <= 0.05 for num in amount_nums)
            has_source_amount = any(abs(float(num) - float(user_amount)) <= 0.05 for num in amount_nums)
            has_exchange_hint = bool(re.search(REGEX_CONFIG["exchange_hint_pattern"], thought))

            # ── CNY 过度换算检测 ──
            # 核心信号：工具调用中的金额是否是输入的约 7 倍（只有实际输出错误才惩罚）
            cny_over_conversion = False
            if not is_usd:
                params = extract_tool_data(comp)
                if params and user_amount > 0:
                    actual_norm = normalize_cny_amount(params.get("amount"))
                    if actual_norm:
                        act_match = re.search(r"(\d+(?:\.\d+)?)", actual_norm)
                        if act_match:
                            actual_val = float(act_match.group(1))
                            if abs(actual_val / user_amount - USD_TO_CNY_RATE) < 0.3:
                                cny_over_conversion = True
                # 辅助信号：think 中出现汇率/USD 标记，且非否定语境
                # 按标记逐条检查邻近否定词，避免全局否定误杀
                if not cny_over_conversion:
                    for m in re.finditer(
                        r"美元|美金|美刀|[$]|usd|USD|汇率|汇兑|转换|乘以\s*7|×\s*7|[*]\s*7|7\s*[*×]",
                        thought
                    ):
                        prefix = thought[max(0, m.start() - 10):m.start()]
                        if not re.search(r"不(?:需要|用|必|是|能|会|该|应该)|无需|别(?:用|要|换算|算|转)", prefix):
                            cny_over_conversion = True
                            break

            # ── 阶梯评分 ──
            if not has_calc:
                score = 0.2  # 有 think 但无计算
            elif not has_result:
                if has_source_amount and has_calc:
                    score = 0.4  # 有计算但结果全错
                else:
                    score = 0.3  # 有信号但既无源金额也无结果
            elif is_usd and not has_exchange_hint:
                score = 0.7  # 结果正确但缺汇率说明
            else:
                score = 1.0  # 完整正确

            if cny_over_conversion:
                score = min(score, 0.3)  # CNY 过度换算：封顶 0.3

            rewards.append(score)

        except Exception as e:
            logger.warning(f"self_correction_reward_func error: {e}")
            rewards.append(0.0)

    return rewards


# ── 加权包装 ──
def weighted_format_reward(prompts, completions, **kwargs):
    base = format_reward_func(prompts, completions, **kwargs)
    return [r * FORMAT_REWARD_WEIGHT for r in base]


def weighted_pinyin_reward(prompts, completions, **kwargs):
    base = pinyin_reward_func(prompts, completions, **kwargs)
    return [r * PINYIN_REWARD_WEIGHT for r in base]


def weighted_amount_reward(prompts, completions, **kwargs):
    base = amount_reward_func(prompts, completions, **kwargs)
    return [r * AMOUNT_REWARD_WEIGHT for r in base]


def weighted_self_correction_reward(prompts, completions, **kwargs):
    base = self_correction_reward_func(prompts, completions, **kwargs)
    return [r * SELF_CORRECTION_REWARD_WEIGHT for r in base]


# =========================================================
# 本地指标 CSV 回调：每步保存核心监测指标到本地文件
# =========================================================
_METRIC_FIELDS = [
    "step",
    "reward",
    "reward_std",                 # 族内标准差（GRPO 组内）
    "frac_reward_zero_std",       # 组内零方差比例
    "kl",                         # KL 散度
    "entropy",                    # 策略熵
    "completions/mean_length",    # 平均回答长度
    "completions/max_length",     # 最大回答长度
    "completions/min_length",     # 最小回答长度
    "completions/clipped_ratio",  # 被截断比例
    "clip_ratio/region_mean",     # clip 区域比例
    "clip_ratio/low_mean",        # 低 clip 比例
    "clip_ratio/high_mean",       # 高 clip 比例
    "grad_norm",                  # 梯度范数
    "loss",                       # 训练损失
]


class LocalMetricsCallback(TrainerCallback):
    """每步将核心训练指标追加到本地 CSV，方便离线查看。

    trl 的 GRPOTrainer 已内置计算以下指标：
      - reward / reward_std / frac_reward_zero_std （含族内方差）
      - completions/mean(min/max)_length（回答长度统计）
      - kl / entropy / clip_ratio_* / loss / grad_norm
    本回调只做"从 logs 中提取并写入 CSV"这一件事。
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._header = None  # 首次写时确定

    def _init_header(self, logs_keys):
        available = [f for f in _METRIC_FIELDS if f in logs_keys]
        extra = sorted(k for k in logs_keys if k.startswith("rewards/") and k not in _METRIC_FIELDS)
        self._header = available + extra
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._header)
            writer.writeheader()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs = dict(logs)
        logs.setdefault("step", state.global_step)
        if self._header is None:
            self._init_header(list(logs.keys()))
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._header)
            row = {k: logs.get(k) for k in self._header}
            writer.writerow(row)


# =========================================================
# 本地 Completion 样本回调：每 N 步保存一次（不上传 wandb）
# =========================================================
class LocalCompletionsCallback(TrainerCallback):
    """周期性将 completion 样本保存为本地 JSONL 文件。

    取代原先的 wandb.Table，用户不想在 wandb 看到大表格。
    """

    def __init__(self, output_dir: str, log_steps: int = 50):
        self.output_dir = output_dir
        self.log_steps = log_steps
        self._trainer = None
        self._sample_count = 0

    def bind_trainer(self, trainer):
        self._trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._trainer is None or not hasattr(self._trainer, "_logs"):
            return
        if state.global_step % self.log_steps != 0:
            return

        _logs = self._trainer._logs
        if len(_logs["prompt"]) == 0:
            return

        samples = []
        for i, (prompt, comp) in enumerate(zip(_logs["prompt"], _logs["completion"])):
            entry = {
                "step": state.global_step,
                "idx": i,
                "prompt": prompt,
                "completion": comp,
            }
            for name, rewards in _logs["rewards"].items():
                if i < len(rewards):
                    entry[f"reward_{name}"] = rewards[i]
            if i < len(_logs["advantages"]):
                entry["advantage"] = _logs["advantages"][i]
            samples.append(entry)

        # 每 N 步写一个独立文件，避免无限增长
        path = os.path.join(self.output_dir, f"completions_step{state.global_step:05d}.jsonl")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


# =========================================================
# 周期性评估回调：轻量级监测微调效果
# =========================================================
class PeriodicEvalCallback(TrainerCallback):
    """每隔 eval_steps 步用固定测试集做一次贪婪解码评估。

    结果写入 wandb（eval/* 指标）和本地 JSONL 文件，不在终端输出。
    仅评估 20 条查询 × 1 次推理，对训练速度影响 < 1%。

    同时负责：
      - 跟踪并覆盖保存最优 checkpoint（pass_rate 最高时触发）
      - 将 KL 散度超标告警写入本地文件（不输出终端）
    """

    def __init__(self, model, tokenizer, eval_queries, output_dir, eval_steps=50,
                 kl_warning_threshold=0.08):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_queries = eval_queries
        self.output_dir = output_dir
        self.eval_steps = eval_steps
        self.results_path = os.path.join(output_dir, "eval_metrics.jsonl")
        self._device = None

        # ── 最优 checkpoint 追踪 ──
        self._best_pass_rate = -1.0
        self._best_step = -1
        self._best_ckpt_path = get_best_ckpt_path(output_dir)

        # ── KL 日志器（只写文件，不输出终端） ──
        kl_log_path = os.path.join(output_dir, "kl_warnings.log")
        self._kl_logger = KLLogger(kl_log_path, threshold=kl_warning_threshold)
        self._trainer = None

    def bind_trainer(self, trainer):
        self._trainer = trainer

    def _get_device(self):
        if self._device is not None:
            return self._device
        for param in self.model.parameters():
            self._device = param.device
            return self._device
        return torch.device("cpu")

    def _eval_one(self, query):
        """贪婪解码单条查询，返回与 compute_metrics 兼容的记录。"""
        prompt = SYSTEM_PROMPT + f"\nUser: {query}\nAssistant: "
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self._get_device())

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=GRPO_TRAINING_CONFIG.get("eval_max_new_tokens", 256),
                use_cache=True,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        expected_pinyin, expected_amount = expected_answer(query)
        params = extract_tool_data(response)

        actual_pinyin = (params.get("target_account") or "").strip().lower() if params else None
        actual_amount = params.get("amount") if params else None

        pinyin_ok = actual_pinyin == expected_pinyin.lower() if expected_pinyin else False
        amount_ok = amounts_equal(expected_amount, actual_amount) if expected_amount else False
        format_ok = has_valid_r1_format(response)

        return {
            "query": query,
            "response": response,
            "format_ok": format_ok,
            "pinyin_ok": pinyin_ok,
            "amount_ok": amount_ok,
            "expected_pinyin": expected_pinyin,
            "expected_amount": expected_amount,
            "actual_pinyin": actual_pinyin,
            "actual_amount": actual_amount,
            "is_usd": is_usd_query(query),
            "name": extract_name_from_query(query, PINYIN_MAP),
            "passed": format_ok and pinyin_ok and amount_ok,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每步检查 KL 散度，超标则写入本地日志（不输出终端）。"""
        if logs is None:
            return
        kl = logs.get("kl")
        if kl is not None:
            self._kl_logger.check(state.global_step, kl)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return

        was_training = self.model.training
        self.model.eval()

        records = []
        for query in self.eval_queries:
            try:
                records.append(self._eval_one(query))
            except Exception:
                pass

        if was_training:
            self.model.train()

        if not records:
            return

        metrics = compute_metrics(records)
        pass_rate = metrics["overall"]["pass_rate"]

        # ── 记录到 wandb ──
        if is_wandb_available() and wandb.run is not None:
            wandb.log({
                "eval/pass_rate": pass_rate,
                "eval/format_rate": metrics["overall"]["format_rate"],
                "eval/pinyin_rate": metrics["overall"]["pinyin_rate"],
                "eval/amount_rate": metrics["overall"]["amount_rate"],
                "eval/cny_pass_rate": metrics["by_currency"]["cny"]["pass_rate"],
                "eval/usd_pass_rate": metrics["by_currency"]["usd"]["pass_rate"],
                "step": state.global_step,
            })

        # ── 本地保存完整 metrics ──
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.results_path, "a") as f:
            entry = {"step": state.global_step, "metrics": metrics}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # ── 最优 checkpoint：覆盖保存 ──
        if self._trainer is not None and pass_rate > self._best_pass_rate:
            self._best_pass_rate = pass_rate
            self._best_step = state.global_step
            save_peft_checkpoint(self._trainer, self._best_ckpt_path)
            # 将元信息写入旁边的小文件，方便 test.py 定位
            meta_path = self._best_ckpt_path + "_meta.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "step": self._best_step,
                    "pass_rate": self._best_pass_rate,
                    "metrics": metrics["overall"],
                }, f, ensure_ascii=False)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = GRPO_TRAINING_CONFIG["output_dir"]
    output_dir = os.path.join(RESULTS_BASE_DIR, f"{base_output_name}_{timestamp}")

    # 保存代码快照供事后分析
    save_code_snapshot(output_dir)

    # ==========================================
    # 生成训练数据集
    # ==========================================
    train_dataset = generate_training_data()

    # ==========================================
    # 加载 Unsloth 量化模型
    # ==========================================
    print("正在加载 Qwen2.5-3B 模型与 LoRA 适配器...")
    max_seq_length = TRAIN_MODEL_CONFIG["max_seq_length"]

    torch_dtype = getattr(torch, TRAIN_MODEL_CONFIG["torch_dtype"])

    # 使用配置中的精度与量化选项加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = TRAIN_MODEL_CONFIG["base_model_name"],
        max_seq_length = max_seq_length,
        load_in_4bit = TRAIN_MODEL_CONFIG["load_in_4bit"],
        torch_dtype = torch_dtype,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_CONFIG["r"],
        target_modules = LORA_CONFIG["target_modules"],
        lora_alpha = LORA_CONFIG["lora_alpha"],
        lora_dropout = LORA_CONFIG["lora_dropout"],
        bias = LORA_CONFIG["bias"],
        use_gradient_checkpointing = LORA_CONFIG["use_gradient_checkpointing"],
    )

    # ==========================================
    # 配置并启动 GRPO 训练
    # ==========================================
    training_args = GRPOConfig(
        learning_rate = GRPO_TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type = GRPO_TRAINING_CONFIG["lr_scheduler_type"],
        warmup_steps = GRPO_TRAINING_CONFIG.get("warmup_steps", 0),
        logging_steps = GRPO_TRAINING_CONFIG["logging_steps"],
        bf16 = GRPO_TRAINING_CONFIG["bf16"],
        per_device_train_batch_size = GRPO_TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps = GRPO_TRAINING_CONFIG["gradient_accumulation_steps"],
        num_generations = GRPO_TRAINING_CONFIG["num_generations"],
        generation_batch_size = GRPO_TRAINING_CONFIG["generation_batch_size"],
        max_prompt_length = GRPO_TRAINING_CONFIG["max_prompt_length"],
        max_completion_length = GRPO_TRAINING_CONFIG["max_completion_length"],
        max_steps = GRPO_TRAINING_CONFIG["max_steps"],
        output_dir = output_dir,
        report_to = GRPO_TRAINING_CONFIG["report_to"],
        beta = GRPO_TRAINING_CONFIG["beta"],
        log_completions = GRPO_TRAINING_CONFIG["log_completions"],
        wandb_log_unique_prompts = GRPO_TRAINING_CONFIG["wandb_log_unique_prompts"],
        generation_kwargs = GRPO_TRAINING_CONFIG["generation_kwargs"],
        log_level = GRPO_TRAINING_CONFIG.get("log_level", "passive"),
    )

    # trl 0.24.0 要求模型具备 warnings_issued 属性（transformers 5.5.0 尚未提供）
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    # 消除每步的 max_new_tokens vs max_length 冲突告警
    model.generation_config.max_length = None

    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [weighted_format_reward, weighted_pinyin_reward, weighted_amount_reward, weighted_self_correction_reward],
        args = training_args,
        train_dataset = train_dataset,
        processing_class = tokenizer,
    )

    # 禁用 ProgressCallback 中每步打印 metrics 的行为（保留进度条，wandb 仍正常记录）
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, ProgressCallback):
            cb.on_log = lambda *args, **kwargs: None
            break

    # 注册 completion 抽样回调 → 本地 JSONL 保存（不上传 wandb 表格）
    completions_dir = os.path.join(output_dir, "completions_samples")
    completion_callback = LocalCompletionsCallback(
        completions_dir, log_steps=GRPO_TRAINING_CONFIG.get("completions_log_steps", 50)
    )
    completion_callback.bind_trainer(trainer)
    trainer.add_callback(completion_callback)

    # 注册指标 CSV 回调 → 每步保存核心指标到本地
    metrics_csv = os.path.join(output_dir, "training_metrics.csv")
    metrics_callback = LocalMetricsCallback(metrics_csv)
    trainer.add_callback(metrics_callback)

    # 注册周期性评估回调 → 轻量级监测微调效果（wandb + 本地 + 最优 checkpoint）
    eval_queries = list(TEST_CONFIG["eval_queries"])
    eval_steps = GRPO_TRAINING_CONFIG.get("eval_steps", 50)
    kl_threshold = GRPO_TRAINING_CONFIG.get("kl_warning_threshold", 0.08)
    eval_callback = PeriodicEvalCallback(model, tokenizer, eval_queries, output_dir,
                                         eval_steps=eval_steps,
                                         kl_warning_threshold=kl_threshold)
    eval_callback.bind_trainer(trainer)
    trainer.add_callback(eval_callback)

    print("准备就绪，开始 GRPO 训练。")
    print(f"  本地指标 CSV → {metrics_csv}")
    print(f"  本地样本 JSONL → {completions_dir}/")
    print(f"  KL 超标日志 → {os.path.join(output_dir, 'kl_warnings.log')}")
    trainer.train()

    # ── 保存最终 checkpoint ──
    final_ckpt_path = get_final_ckpt_path(output_dir)
    save_peft_checkpoint(trainer, final_ckpt_path)
    # 记录元信息
    with open(final_ckpt_path + "_meta.json", "w") as f:
        json.dump({"step": GRPO_TRAINING_CONFIG["max_steps"]}, f)

    # 也保存一份到 output_dir 根目录（兼容 test.py 的自动查找）
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"训练完成，模型已保存至: {output_dir}")
    print(f"  最优 checkpoint → {get_best_ckpt_path(output_dir)} "
          f"(pass_rate={eval_callback._best_pass_rate:.4f} @ step {eval_callback._best_step})" if eval_callback._best_step > 0
          else f"  最优 checkpoint → 无（评估未触发或未改善）")
    print(f"  最终 checkpoint → {final_ckpt_path}")


if __name__ == "__main__":
    main()
