import re
import json
import logging
from config import REGEX_CONFIG


def extract_tool_data(completion):
    """从模型输出中提取 <tool_call> 内的 JSON 工具调用数据。"""
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
    """校验 amount 是否为人民币格式（必须以 ¥ 或 ￥ 开头，且不含美元标记）。"""
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
    """仅接受人民币金额格式，并规范化为统一的 ¥xx.xx 形式。"""
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


NAME_PREFIXES = ("老", "小", "大", "阿")


def _surname_unique_count(surname, pinyin_map):
    """返回 PINYIN_MAP 中以某个姓氏开头的名字数量。"""
    return sum(1 for n in pinyin_map if n[0] == surname)


def extract_name_from_query(user_query, pinyin_map):
    """按名字长度降序匹配用户输入中的姓名。

    支持：
    - 精确匹配（如 "张三"）
    - 称呼前缀匹配（如 "老张" → 仅在姓氏唯一时匹配，避免歧义）
    - 括号补充（如 "老王（王五）" → 括号内的名字直接命中）
    """
    # 1) 精确匹配
    for name in sorted(pinyin_map.keys(), key=len, reverse=True):
        if name in user_query:
            return name

    # 2) 称呼前缀匹配：老/小/大/阿 + 姓氏（仅当该姓氏在映射中唯一）
    for prefix in NAME_PREFIXES:
        for name in sorted(pinyin_map.keys(), key=len, reverse=True):
            surname = name[0]
            if _surname_unique_count(surname, pinyin_map) == 1:
                if prefix + surname in user_query:
                    return name

    return None


# 中文口语金额 → 数值的映射（在模块加载时从 config 构建）
_CHINESE_NUMERAL_REVERSE = {}


def _init_chinese_numeral_map():
    """延迟加载中文金额映射，避免循环导入。"""
    global _CHINESE_NUMERAL_REVERSE
    if _CHINESE_NUMERAL_REVERSE:
        return
    from config import DATASET_CONFIG
    raw = DATASET_CONFIG.get("chinese_numeral_map", {})
    _CHINESE_NUMERAL_REVERSE = {v: float(k) for k, v in raw.items()}


def extract_query_amount(user_query):
    """从用户语句中提取转账金额（支持数字和中文口语金额如"一百块"）。"""
    # 优先尝试正则匹配数字金额
    amount_match = re.search(REGEX_CONFIG["query_amount_pattern"], user_query)
    if amount_match:
        try:
            amount = float(amount_match.group(1))
            if amount > 0:
                return amount
        except (ValueError, TypeError):
            pass

    # 回退：查找中文口语金额（按长度降序，避免"五十块"被"十块"子串匹配）
    _init_chinese_numeral_map()
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
