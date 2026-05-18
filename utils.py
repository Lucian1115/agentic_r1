import os
import re
import json
import sys
import glob
import shutil
import logging
import subprocess
from config import (
    REGEX_CONFIG,
    USD_CURRENCY_MARKERS,
    PINYIN_MAP,
    USD_TO_CNY_RATE,
    NAME_PREFIXES,
    DATASET_CONFIG,
)


def is_usd_query(text):
    """检测文本是否包含美元货币标记。"""
    return any(marker in text for marker in USD_CURRENCY_MARKERS)


def levenshtein_ratio(s1, s2):
    """归一化编辑距离相似度：1.0 表示完全相同，0.0 表示完全不同。"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # 删除
                curr[j - 1] + 1,   # 插入
                prev[j - 1] + cost # 替换
            )
        prev, curr = curr, prev
    return 1.0 - prev[len2] / max(len1, len2)


def extract_tool_data(completion):
    """从模型输出中提取 <tool_call> 内的 JSON。"""
    if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
        completion = completion[-1].get("content", "")
    completion = str(completion)

    match = re.search(REGEX_CONFIG["tool_call_pattern"], completion, re.DOTALL)
    if not match:
        return None

    raw = match.group(1).strip()
    try:
        tool_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logging.getLogger(__name__).debug("direct json.loads failed, attempting brace extraction")
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = raw[start:end + 1]
        try:
            tool_data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None

    if not isinstance(tool_data, dict):
        return None

    tool_name = None
    for key in ("tool_name", "tool", "function", "name"):
        value = tool_data.get(key)
        if isinstance(value, str) and value.strip():
            tool_name = value.strip()
            break

    if tool_name is not None and tool_name != "transfer_funds":
        return None

    params = tool_data.get("params", tool_data)
    if not isinstance(params, dict):
        return None

    if "target_account" not in params or "amount" not in params:
        return None

    return params


def is_cny_amount_format(amount_str):
    """校验 amount 是否以 ¥/￥ 开头且不含美元标记。"""
    if amount_str is None:
        return False
    text = str(amount_str).strip()
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in ["$", "usd", "美元", "美金", "美刀"]):
        return False
    if text.startswith("¥") or text.startswith("￥"):
        return True
    return False


def normalize_cny_amount(amount_str):
    """将人民币金额规范化为 ¥xx.xx。"""
    if not is_cny_amount_format(amount_str):
        return None
    text = str(amount_str)
    text = re.sub(r"[¥￥]|人民币|CNY", "", text, flags=re.IGNORECASE)
    text = text.replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return f"¥{value:.2f}"


def surname_unique_count(surname, pinyin_map):
    """返回 PINYIN_MAP 中以某个姓氏开头的名字数量。"""
    return sum(1 for n in pinyin_map if n[0] == surname)


def extract_name_from_query(user_query, pinyin_map):
    """从用户输入中提取姓名。

    按长度降序精确匹配，支持称呼前缀（老/小+姓，仅姓氏唯一时生效）。
    """
    for name in sorted(pinyin_map.keys(), key=len, reverse=True):
        if name in user_query:
            return name

    for prefix in NAME_PREFIXES:
        for name in sorted(pinyin_map.keys(), key=len, reverse=True):
            surname = name[0]
            if surname_unique_count(surname, pinyin_map) == 1:
                if prefix + surname in user_query:
                    return name

    return None


_CHINESE_NUMERAL_REVERSE = {}


def _init_chinese_numeral_map():
    """延迟加载中文口语金额映射。"""
    global _CHINESE_NUMERAL_REVERSE
    if _CHINESE_NUMERAL_REVERSE:
        return
    raw = DATASET_CONFIG.get("chinese_numeral_map", {})
    _CHINESE_NUMERAL_REVERSE = {v: float(k) for k, v in raw.items()}


def extract_query_amount(user_query):
    """从用户语句中提取转账金额，支持数字和中文口语（如"一百块"）。"""
    amount_match = re.search(REGEX_CONFIG["query_amount_pattern"], user_query)
    if amount_match:
        try:
            amount = float(amount_match.group(1))
            if amount > 0:
                return amount
        except (ValueError, TypeError):
            pass

    _init_chinese_numeral_map()
    # 按长度降序匹配，避免"五十块"被"十块"子串误命中
    for chinese_str, value in sorted(
        _CHINESE_NUMERAL_REVERSE.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if chinese_str in user_query:
            return value

    return None


def has_valid_r1_format(completion):
    """检查 completion 是否符合 R1 格式：<think>...</think> + 可解析的 <tool_call> JSON。"""
    if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
        completion = completion[-1].get("content", "")
    text = str(completion)
    stripped = text.lstrip()

    think_start_idx = stripped.find("<think>")
    if think_start_idx == -1:
        return False

    think_end_idx = stripped.find("</think>", think_start_idx)
    if think_end_idx == -1:
        return False

    tool_start_idx = stripped.find("<tool_call>", think_end_idx)
    if tool_start_idx == -1:
        return False

    return extract_tool_data(stripped) is not None


def find_latest_lora_result(results_base_dir="results"):
    """返回 results/ 下最新的 Agentic_R1_Lora_* 目录。"""
    if not os.path.isdir(results_base_dir):
        return None
    candidates = [
        d for d in os.listdir(results_base_dir)
        if d.startswith("Agentic_R1_Lora_") and os.path.isdir(os.path.join(results_base_dir, d))
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return os.path.join(results_base_dir, candidates[0])


def resolve_model_path(user_input, results_base_dir="results"):
    """将用户输入解析为模型路径：支持完整路径或 results/ 子目录名。"""
    if user_input is None:
        return None
    if os.path.isdir(user_input):
        return user_input
    inside_results = os.path.join(results_base_dir, user_input)
    if os.path.isdir(inside_results):
        return inside_results
    print(f"错误: 找不到模型目录 '{user_input}'（或 '{inside_results}'）")
    sys.exit(1)


def save_code_snapshot(output_dir, source_dir="."):
    """保存项目 .py 文件和 git commit hash 到训练输出目录。"""
    snapshot_dir = os.path.join(output_dir, "code_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    for py_file in glob.glob(os.path.join(source_dir, "*.py")):
        if os.path.isfile(py_file):
            shutil.copy2(py_file, snapshot_dir)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=source_dir, timeout=5,
        )
        if result.returncode == 0:
            with open(os.path.join(snapshot_dir, "git_commit.txt"), "w") as f:
                f.write(result.stdout.strip() + "\n")
    except Exception:
        pass


class KLLogger:
    """KL 散度超标时写入本地日志文件，不在终端输出。"""

    def __init__(self, log_path: str, threshold: float = 0.08):
        self.log_path = log_path
        self.threshold = threshold
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._logger = logging.getLogger(f"kl_monitor_{id(self)}")
        self._logger.propagate = False
        self._logger.setLevel(logging.WARNING)
        if not self._logger.handlers:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            self._logger.addHandler(handler)

    def check(self, step: int, kl: float):
        if kl > self.threshold:
            self._logger.warning(
                "step=%s KL=%.4f > threshold=%.3f (policy may be drifting too far)",
                step, kl, self.threshold,
            )


def get_checkpoint_dir(output_dir: str):
    return os.path.join(output_dir, "checkpoints")


def get_best_ckpt_path(output_dir: str):
    return os.path.join(get_checkpoint_dir(output_dir), "best_model")


def get_final_ckpt_path(output_dir: str):
    return os.path.join(get_checkpoint_dir(output_dir), "final_model")


def save_peft_checkpoint(trainer, path: str):
    """保存 LoRA 权重与 tokenizer，覆盖已有。"""
    import io, contextlib
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.model.save_pretrained(path)
        trainer.processing_class.save_pretrained(path)


def validate_pinyin_map():
    if not isinstance(PINYIN_MAP, dict):
        raise TypeError("PINYIN_MAP 必须是字典类型")
    for name, pinyin in PINYIN_MAP.items():
        if not isinstance(name, str) or not isinstance(pinyin, str):
            raise ValueError(f"映射表条目必须为字符串对: {name} -> {pinyin}")
        if len(name) == 0 or len(pinyin) == 0:
            raise ValueError(f"映射表中不应包含空字符串: {name} -> {pinyin}")


def extract_user_query(prompt):
    """从 prompt 中提取 User: 行的查询文本。"""
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[-1], dict):
        prompt = prompt[-1].get("content", "")
    prompt = str(prompt)
    user_line_match = re.search(REGEX_CONFIG["user_line_pattern"], prompt)
    if not user_line_match:
        return None
    return user_line_match.group(1).strip()


def is_negative_prompt(prompt):
    """检测 prompt 是否缺少有效转账请求（闲聊/查询类负样本）。"""
    user_query = extract_user_query(prompt)
    if not user_query:
        return True
    name = extract_name_from_query(user_query, PINYIN_MAP)
    amount = extract_query_amount(user_query)
    return name is None or amount is None


def format_cny_value(value):
    return f"¥{float(value):.2f}"


def amounts_equal(expected, actual):
    exp_norm = normalize_cny_amount(expected)
    act_norm = normalize_cny_amount(actual)
    return bool(exp_norm and act_norm and exp_norm == act_norm)


def has_calc_signal(text):
    """检查 think 文本中是否包含计算或换算信号。"""
    return bool(re.search(r"[+\-*/×=]|乘|算|计算|换算|折合|得出|所以|因此", text))


_AMOUNT_CTX_RE = re.compile(
    r"(?:[¥￥$]|人民币|CNY|USD|usd|元|块|美元|美金|美刀)\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*(?:[¥￥$]|人民币|CNY|元|块|美元|美金|美刀)"
    r"|(\d+(?:\.\d+)?)\s*[=＝]\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*[*×Xx]\s*(\d+(?:\.\d+)?)"
    r"|[=＝]\s*(\d+(?:\.\d+)?)"
)


def extract_amount_numbers(text):
    """从 think 文本中提取金额/计算上下文中的数字，过滤日期等无关数字。"""
    nums = set()
    for m in _AMOUNT_CTX_RE.finditer(text):
        for g in m.groups():
            if g is not None:
                nums.add(g)
    return list(nums)


def expected_answer(query):
    name = extract_name_from_query(query, PINYIN_MAP)
    amount = extract_query_amount(query)
    if name is None or amount is None:
        return None, None
    pinyin = PINYIN_MAP[name]
    if is_usd_query(query):
        final_value = round(amount * USD_TO_CNY_RATE, 2)
    else:
        final_value = round(amount, 2)
    final_amount = f"¥{final_value:.2f}"
    return pinyin, final_amount


def iter_prompt_completion(prompts, completions, logger):
    """zip(prompts, completions)，长度不一致时告警，避免静默截断。"""
    prompt_len = len(prompts)
    completion_len = len(completions)
    if prompt_len != completion_len:
        logger.warning(
            "prompts/completions length mismatch: prompts=%s, completions=%s",
            prompt_len,
            completion_len,
        )
    return zip(prompts, completions)
