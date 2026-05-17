import random
from datasets import Dataset
from config import (
    SYSTEM_PROMPT,
    PINYIN_MAP,
    USD_CURRENCY_MARKERS,
    CURRENCY_CNY,
    DATASET_CONFIG,
    NAME_PREFIXES,
)
from utils import surname_unique_count, validate_pinyin_map

random.seed(DATASET_CONFIG["random_seed"])


# ── 金额格式化 ──────────────────────────────────────────────
def _format_amount_str(amount, currency):
    """将数值金额与货币组合成多样化的表达。

    返回字符串填入模板的 {amount_str} 占位。可能产生：
      - "100 块钱" / "100 元"（普通 CNY）
      - "¥100"（CNY 前缀）
      - "100 美元"（普通 USD）
      - "$100"（USD 前缀）
    """
    cfg = DATASET_CONFIG

    # ¥ / $ 前缀格式
    if random.random() < cfg.get("amount_prefix_probability", 0):
        if currency in CURRENCY_CNY:
            return f"¥{amount:g}"  # :g 去尾随零
        else:
            return f"${amount:g}"

    # 默认：数值 + 货币
    return f"{amount:g} {currency}"


# ── 人名格式化 ──────────────────────────────────────────────
def _format_name(name):
    """可选地在人名前添加称呼前缀（仅当姓氏在映射中唯一时，避免歧义）。"""
    if random.random() < DATASET_CONFIG.get("name_prefix_probability", 0):
        surname = name[0]
        if surname_unique_count(surname, PINYIN_MAP) == 1:
            gen_prefixes = DATASET_CONFIG.get("name_prefixes", ("老", "小"))
            prefix = random.choice(gen_prefixes)
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

    涵盖：23 种查询模板、18 个人名、多种金额格式、称呼前缀、负样本、
    名字消歧样本、困难名字过采样。
    """
    print("生成多样化训练数据...")
    validate_pinyin_map()

    cfg = DATASET_CONFIG
    names = list(PINYIN_MAP.keys())
    templates = list(cfg["query_templates"])
    negative_templates = list(cfg.get("negative_templates", []))
    disambiguation_templates = list(cfg.get("disambiguation_templates", []))
    confusable_pairs = cfg.get("confusable_name_pairs", {})
    hard_names = set(cfg.get("hard_names", []))
    oversample_ratio = cfg.get("hard_name_oversample_ratio", 1.0)
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
        # ── 困难名字过采样 ──
        if hard_names and random.random() < cfg.get("hard_name_sample_probability", 0.3):
            name_pool = [n for n in names if n in hard_names]
            name_weights = [oversample_ratio if n in hard_names else 1.0 for n in name_pool]
            name = random.choices(name_pool, weights=name_weights, k=1)[0]
        else:
            name = random.choice(names)

        display_name = _format_name(name)

        # ── 货币与金额生成 ──
        if random.random() < cfg["usd_probability"]:
            currency = random.choice(USD_CURRENCY_MARKERS)
        else:
            currency = random.choice(CURRENCY_CNY)

        # 中文口语金额：直接从映射表选键作为实际金额，使 5% 概率真正生效
        use_chinese = False
        chinese_map = cfg.get("chinese_numeral_map", {})
        if currency in CURRENCY_CNY and random.random() < cfg.get("chinese_amount_probability", 0):
            valid_keys = [k for k in chinese_map if cfg["amount_min"] <= k <= cfg["amount_max"]]
            if valid_keys:
                amount = float(random.choice(valid_keys))
                amount_str = chinese_map[int(amount)]
                use_chinese = True

        if not use_chinese:
            amount = _random_amount()
            amount_str = _format_amount_str(amount, currency)

        # ── 名字消歧模板 ──
        if disambiguation_templates and random.random() < cfg.get("disambiguation_probability", 0):
            confusable = confusable_pairs.get(name)
            if confusable:
                tpl = random.choice(disambiguation_templates)
                query = tpl.format(name=display_name, confusable=confusable, amount_str=amount_str)
            else:
                tpl = random.choice(templates)
                query = tpl.format(name=display_name, amount_str=amount_str)
        else:
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
