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


def _format_amount_str(amount, currency):
    """格式化金额+货币为模板用的 amount_str，如 "¥100"、"100 块钱"、"$50"。"""
    cfg = DATASET_CONFIG

    if random.random() < cfg.get("amount_prefix_probability", 0):
        if currency in CURRENCY_CNY:
            return f"¥{amount:g}"
        else:
            return f"${amount:g}"

    return f"{amount:g} {currency}"


def _format_name(name):
    """人名前随机加称呼前缀（老/小+姓），仅姓氏唯一时生效。"""
    if random.random() < DATASET_CONFIG.get("name_prefix_probability", 0):
        surname = name[0]
        if surname_unique_count(surname, PINYIN_MAP) == 1:
            gen_prefixes = DATASET_CONFIG.get("name_prefixes", ("老", "小"))
            prefix = random.choice(gen_prefixes)
            return prefix + surname
    return name


def _random_amount():
    cfg = DATASET_CONFIG
    if random.random() < cfg.get("decimal_amount_probability", 0):
        base = random.uniform(cfg["amount_min"], cfg["amount_max"])
        return round(base, 2)
    return float(random.randint(cfg["amount_min"], cfg["amount_max"]))


def generate_training_data():
    """生成训练数据集：多模板 × 多人名 × 多种金额格式，含负样本和消歧样本。"""
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
        if negative_templates and random.random() < cfg.get("negative_sample_probability", 0):
            tpl = random.choice(negative_templates)
            name = random.choice(names) if "{name}" in tpl else ""
            query = tpl.format(name=name) if name else tpl
            prompt = SYSTEM_PROMPT + f"\nUser: {query}\nAssistant: "
            raw_data.append({"prompt": prompt, "is_negative": True})
            continue

        if hard_names and random.random() < cfg.get("hard_name_sample_probability", 0.3):
            name_pool = [n for n in names if n in hard_names]
            name_weights = [oversample_ratio if n in hard_names else 1.0 for n in name_pool]
            name = random.choices(name_pool, weights=name_weights, k=1)[0]
        else:
            name = random.choice(names)

        display_name = _format_name(name)

        if random.random() < cfg["usd_probability"]:
            currency = random.choice(USD_CURRENCY_MARKERS)
        else:
            currency = random.choice(CURRENCY_CNY)

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
