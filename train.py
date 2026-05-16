import os
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
from dataset import generate_training_data, is_usd_query
from config import (
    PINYIN_MAP,
    REWARD_WEIGHTS,
    REGEX_CONFIG,
    TRAIN_MODEL_CONFIG,
    LORA_CONFIG,
    GRPO_TRAINING_CONFIG,
    USD_TO_CNY_RATE,
    RESULTS_BASE_DIR,
)
from utils import (
    extract_tool_data,
    normalize_cny_amount,
    extract_name_from_query,
    extract_query_amount,
    has_valid_r1_format,
)

if is_wandb_available():
    import wandb

# ==========================================
# 生成训练数据集
# ==========================================
FORMAT_REWARD_WEIGHT = REWARD_WEIGHTS["format"]
CONSTRAINT_REWARD_WEIGHT = REWARD_WEIGHTS["constraint"]
SELF_CORRECTION_REWARD_WEIGHT = REWARD_WEIGHTS["self_correction"]


def _extract_user_query(prompt):
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[-1], dict):
        prompt = prompt[-1].get("content", "")
    prompt = str(prompt)
    user_line_match = re.search(REGEX_CONFIG["user_line_pattern"], prompt)
    if not user_line_match:
        return None
    return user_line_match.group(1).strip()


def _is_negative_prompt(prompt):
    """检测 prompt 是否不包含有效的转账请求（如负样本 / 闲聊）。"""
    user_query = _extract_user_query(prompt)
    if not user_query:
        return True
    name = extract_name_from_query(user_query, PINYIN_MAP)
    amount = extract_query_amount(user_query)
    return name is None or amount is None


def _format_cny_value(value):
    return f"¥{float(value):.2f}"


def _has_calc_signal(text):
    """判断思路中是否真的出现了计算或换算信号。"""
    return bool(re.search(r"[+\-*/×=]|乘|算|计算|换算|折合|得出|所以|因此", text))


# 匹配金额上下文中的数字：币种标记、计算表达式等
_AMOUNT_CTX_RE = re.compile(
    r"(?:[¥￥$]|人民币|CNY|USD|usd|元|块|美元|美金|美刀)\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*(?:[¥￥$]|人民币|CNY|元|块|美元|美金|美刀)"
    r"|(\d+(?:\.\d+)?)\s*[=＝]\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*[*×Xx]\s*(\d+(?:\.\d+)?)"
    r"|[=＝]\s*(\d+(?:\.\d+)?)"
)


def _extract_amount_numbers(text):
    """从 think 文本中提取出现在金额/计算上下文中的数字，避免误匹配无关数字（如日期）。"""
    nums = set()
    for m in _AMOUNT_CTX_RE.finditer(text):
        for g in m.groups():
            if g is not None:
                nums.add(g)
    return list(nums)


def _iter_prompt_completion(prompts, completions, logger):
    """遍历样本并在长度不一致时告警，避免 zip 静默截断。"""
    prompt_len = len(prompts)
    completion_len = len(completions)
    if prompt_len != completion_len:
        logger.warning(
            "prompts/completions length mismatch: prompts=%s, completions=%s",
            prompt_len,
            completion_len,
        )
    return zip(prompts, completions)


# =========================================================
# 奖励函数 1：R1 格式校验
# =========================================================
def format_reward_func(prompts, completions, **kwargs):
    """判断每个 completion 是否符合 R1 格式要求。

    正样本：要求 <think> + <tool_call> JSON。
    负样本（无转账意图）：正确行为是不输出 tool_call，得满分。
    """
    rewards = []
    logger = logging.getLogger(__name__)

    for prompt, comp in _iter_prompt_completion(prompts, completions, logger):
        try:
            if _is_negative_prompt(prompt):
                # 负样本：不输出 tool_call 才是正确的
                rewards.append(0.0 if has_valid_r1_format(comp) else 1.0)
            else:
                rewards.append(1.0 if has_valid_r1_format(comp) else 0.0)
        except Exception as e:
            logger.warning(f"format_reward_func error: {e}")
            rewards.append(0.0)

    return rewards


# =========================================================
# 奖励函数 2：业务约束校验
# =========================================================
def constraint_reward_func(prompts, completions, **kwargs):
    """校验工具调用中的姓名与金额是否与 prompt 中的用户输入一致。

    负样本：正确行为是不输出 tool_call，得满分 1.0。
    """
    rewards = []
    logger = logging.getLogger(__name__)

    for prompt, comp in _iter_prompt_completion(prompts, completions, logger):
        try:
            # 负样本：正确行为是不调用工具
            if _is_negative_prompt(prompt):
                rewards.append(0.0 if has_valid_r1_format(comp) else 1.0)
                continue

            if not has_valid_r1_format(comp):
                rewards.append(0.0)
                continue

            # 1. 提取用户输入行
            user_query = _extract_user_query(prompt)
            if not user_query:
                rewards.append(0.0)
                continue

            # 2. 提取姓名
            user_name = extract_name_from_query(user_query, PINYIN_MAP)

            # 3. 提取金额
            user_amount = extract_query_amount(user_query)

            if not user_name or user_amount is None:
                rewards.append(0.0)
                continue

            # 4. 计算参考答案
            expected_pinyin = PINYIN_MAP[user_name]
            is_usd = is_usd_query(user_query)
            if is_usd:
                expected_amount_str = _format_cny_value(round(user_amount * USD_TO_CNY_RATE, 2))
            else:
                expected_amount_str = _format_cny_value(round(user_amount, 2))

            params = extract_tool_data(comp)
            if params is None:
                rewards.append(0.0)
                continue

            score = 0.0
            actual_pinyin = (params.get("target_account") or "").strip().lower()
            if actual_pinyin == expected_pinyin.lower():
                score += 0.6

            expected_normalized = normalize_cny_amount(expected_amount_str)
            actual_normalized = normalize_cny_amount(params.get("amount"))

            if expected_normalized and actual_normalized and expected_normalized == actual_normalized:
                score += 0.4

            rewards.append(score)

        except Exception as e:
            logger.warning(f"constraint_reward_func error: {e}")
            rewards.append(0.0)

    return rewards


# =========================================================
# 奖励函数 3：思路过程校验
# =========================================================
def self_correction_reward_func(prompts, completions, **kwargs):
    """检查 <think> 区块中是否包含必要的计算过程或结果。

    负样本：正确行为是不输出 tool_call，得满分 1.0。
    """
    rewards = []
    logger = logging.getLogger(__name__)

    for prompt, comp in _iter_prompt_completion(prompts, completions, logger):
        try:
            # 负样本：正确行为是不调用工具
            if _is_negative_prompt(prompt):
                rewards.append(0.0 if has_valid_r1_format(comp) else 1.0)
                continue

            if not has_valid_r1_format(comp):
                rewards.append(0.0)
                continue

            # 提取用户输入与金额
            user_query = _extract_user_query(prompt)
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
            if think_match:
                thought = think_match.group(1)

                amount_nums = _extract_amount_numbers(thought)
                if not amount_nums:
                    amount_nums = re.findall(r"(\d+(?:\.\d+)?)", thought)
                has_calc_signal = _has_calc_signal(thought)
                has_result = any(abs(float(num) - float(expected_value)) <= 0.05 for num in amount_nums)
                has_source_amount = any(abs(float(num) - float(user_amount)) <= 0.05 for num in amount_nums)

                if is_usd:
                    has_exchange_hint = bool(re.search(REGEX_CONFIG["exchange_hint_pattern"], thought))
                    if has_exchange_hint and has_result and has_source_amount and has_calc_signal:
                        rewards.append(1.0)
                    elif has_result and has_source_amount and has_calc_signal:
                        rewards.append(0.4)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(1.0 if has_result and has_calc_signal else 0.0)
            else:
                rewards.append(0.0)

        except Exception as e:
            logger.warning(f"self_correction_reward_func error: {e}")
            rewards.append(0.0)

    return rewards


def weighted_format_reward(prompts, completions, **kwargs):
    base = format_reward_func(prompts, completions, **kwargs)
    return [r * FORMAT_REWARD_WEIGHT for r in base]


def weighted_constraint_reward(prompts, completions, **kwargs):
    base = constraint_reward_func(prompts, completions, **kwargs)
    return [r * CONSTRAINT_REWARD_WEIGHT for r in base]


def weighted_self_correction_reward(prompts, completions, **kwargs):
    base = self_correction_reward_func(prompts, completions, **kwargs)
    return [r * SELF_CORRECTION_REWARD_WEIGHT for r in base]


# =========================================================
# Wandb 抽样回调：每 N 步记录一次 completion 样本表
# =========================================================
class WandbCompletionCallback(TrainerCallback):
    """周期性将 completion 样本写入 wandb 表格，避免每步记录造成数据膨胀。"""

    def __init__(self, log_completion_steps: int = 50):
        self.log_completion_steps = log_completion_steps
        self._trainer = None

    def bind_trainer(self, trainer):
        self._trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_wandb_available() or wandb.run is None:
            return
        if self._trainer is None or not hasattr(self._trainer, "_logs"):
            return
        if state.global_step % self.log_completion_steps != 0:
            return

        _logs = self._trainer._logs
        if len(_logs["prompt"]) == 0:
            return

        import pandas as pd

        table = {
            "step": [str(state.global_step)] * len(_logs["prompt"]),
            "prompt": list(_logs["prompt"]),
            "completion": list(_logs["completion"]),
        }
        for name, rewards in _logs["rewards"].items():
            table[name] = list(rewards)
        table["advantage"] = list(_logs["advantages"])

        df = pd.DataFrame(table)
        if args.wandb_log_unique_prompts:
            df = df.drop_duplicates(subset=["prompt"])
        wandb.log({"completions": wandb.Table(dataframe=df)})


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = GRPO_TRAINING_CONFIG["output_dir"]
    output_dir = os.path.join(RESULTS_BASE_DIR, f"{base_output_name}_{timestamp}")

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
        reward_funcs = [weighted_format_reward, weighted_constraint_reward, weighted_self_correction_reward],
        args = training_args,
        train_dataset = train_dataset,
        processing_class = tokenizer,
    )

    # 禁用 ProgressCallback 中每步打印 metrics 的行为（保留进度条，wandb 仍正常记录）
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, ProgressCallback):
            cb.on_log = lambda *args, **kwargs: None
            break

    # 注册 completion 抽样回调（每 50 步在 wandb 中记录一次样本表）
    completion_callback = WandbCompletionCallback(log_completion_steps=50)
    completion_callback.bind_trainer(trainer)
    trainer.add_callback(completion_callback)

    print("准备就绪，开始 GRPO 训练。")
    trainer.train()

    # 保存LoRA权重
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"训练完成，模型已保存至: {output_dir}")


if __name__ == "__main__":
    main()
