import random
from datasets import Dataset
from config import (
    SYSTEM_PROMPT,
    PINYIN_MAP,
    USD_CURRENCY_MARKERS,
    CURRENCY_CNY,
    DATASET_CONFIG,
)

random.seed(42)


def is_usd_query(text):
    return any(marker in text for marker in USD_CURRENCY_MARKERS)


def validate_pinyin_map():
    if not isinstance(PINYIN_MAP, dict):
        raise TypeError("PINYIN_MAP 必须是字典类型")
    for name, pinyin in PINYIN_MAP.items():
        if not isinstance(name, str) or not isinstance(pinyin, str):
            raise ValueError(f"映射表条目必须为字符串对: {name} -> {pinyin}")
        if len(name) == 0 or len(pinyin) == 0:
            raise ValueError(f"映射表中不应包含空字符串: {name} -> {pinyin}")


# ── 金额格式化 ──────────────────────────────────────────────
def _format_amount_str(amount, currency):
    """将数值金额与货币组合成多样化的表达。

    返回字符串填入模板的 {amount_str} 占位。可能产生：
      - "100 块钱" / "100 元"（普通 CNY）
      - "¥100"（CNY 前缀）
      - "100 美元"（普通 USD）
      - "$100"（USD 前缀）
      - "一百块"（中文口语金额）
    """
    cfg = DATASET_CONFIG

    # 中文口语金额：从映射表中随机取匹配范围内的键
    if currency in CURRENCY_CNY and random.random() < cfg.get("chinese_amount_probability", 0):
        chinese_map = cfg.get("chinese_numeral_map", {})
        valid_keys = [k for k in chinese_map if cfg["amount_min"] <= k <= cfg["amount_max"]]
        if valid_keys:
            key = random.choice(valid_keys)
            return chinese_map[key]

    # ¥ / $ 前缀格式
    if random.random() < cfg.get("amount_prefix_probability", 0):
        if currency in CURRENCY_CNY:
            return f"¥{amount:g}"  # :g 去尾随零
        else:
            return f"${amount:g}"

    # 默认：数值 + 货币
    return f"{amount:g} {currency}"


# ── 人名格式化 ──────────────────────────────────────────────
def _surname_unique_count(surname):
    return sum(1 for n in PINYIN_MAP if n[0] == surname)


def _format_name(name):
    """可选地在人名前添加称呼前缀（仅当姓氏在映射中唯一时，避免歧义）。"""
    if random.random() < DATASET_CONFIG.get("name_prefix_probability", 0):
        surname = name[0]
        if _surname_unique_count(surname) == 1:
            prefix = random.choice(["老", "小"])
            return prefix + surname
    return name


# ── 金额生成 ──────────────────────────────────────────────
def _random_amount():
    cfg = DATASET_CONFIG
    if random.random() < cfg.get("decimal_amount_probability", 0):
        base = random.uniform(cfg["amount_min"], cfg["amount_max"])
        return round(base, 2)
    return float(random.randint(cfg["amount_min"], cfg["amount_max"]))


# ── 主生成函数 ──────────────────────────────────────────────
def generate_training_data():
    """生成多样化的训练数据集。

    涵盖：20 种查询模板、18 个人名、多种金额格式、称呼前缀、负样本。
    """
    print("生成多样化训练数据...")
    validate_pinyin_map()

    cfg = DATASET_CONFIG
    names = list(PINYIN_MAP.keys())
    templates = list(cfg["query_templates"])
    negative_templates = list(cfg.get("negative_templates", []))
    raw_data = []

    for _ in range(cfg["num_samples"]):
        # 负样本：无转账意图的查询
        if negative_templates and random.random() < cfg.get("negative_sample_probability", 0):
            tpl = random.choice(negative_templates)
            name = random.choice(names) if "{name}" in tpl else ""
            query = tpl.format(name=name) if name else tpl
            prompt = SYSTEM_PROMPT + f"\nUser: {query}\nAssistant: "
            raw_data.append({"prompt": prompt, "is_negative": True})
            continue

        # 正样本
        name = random.choice(names)
        amount = _random_amount()
        display_name = _format_name(name)

        if random.random() < cfg["usd_probability"]:
            currency = random.choice(USD_CURRENCY_MARKERS)
        else:
            currency = random.choice(CURRENCY_CNY)

        amount_str = _format_amount_str(amount, currency)
        tpl = random.choice(templates)
        query = tpl.format(name=display_name, amount_str=amount_str)

        prompt = SYSTEM_PROMPT + f"\nUser: {query}\nAssistant: "
        raw_data.append({"prompt": prompt, "is_negative": False})

    train_dataset = Dataset.from_list(raw_data)
    pos_count = sum(1 for r in raw_data if not r["is_negative"])
    neg_count = sum(1 for r in raw_data if r["is_negative"])
    print(f"已生成 {len(train_dataset)} 条训练数据（正例 {pos_count}，负例 {neg_count}）")

    if len(train_dataset) == 0:
        raise ValueError("生成的训练数据集为空。")

    return train_dataset
