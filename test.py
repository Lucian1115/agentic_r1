import os
import sys
import json
import random
import torch
from datetime import datetime
import tqdm
from unsloth import FastLanguageModel
from dataset import is_usd_query
from config import (
    SYSTEM_PROMPT,
    PINYIN_MAP,
    USD_CURRENCY_MARKERS,
    CURRENCY_CNY,
    INFERENCE_CONFIG,
    TRAIN_MODEL_CONFIG,
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


# ── 测试集生成 ──────────────────────────────────────────────
def generate_test_queries(seed=123):
    """用固定种子生成覆盖全面的测试集，与训练数据生成逻辑解耦。"""
    random.seed(seed)
    names = sorted(PINYIN_MAP.keys())
    templates = [
        "帮我给{name}转账 {amount} {currency}。",
        "请给{name}转 {amount} {currency}。",
        "给{name}打 {amount} {currency}。",
        "麻烦给{name}转账 {amount} {currency}。",
    ]
    test_amounts = [
        10, 25, 50, 100, 200, 500, 999, 123.45, 88.8, 0.5, 10000, 76543.21
    ]

    queries = []

    # 每个名字 × 每个金额档位 × 每种货币 → 系统性覆盖
    for name in names:
        for amount in test_amounts[:6]:  # CNY
            currency = random.choice(CURRENCY_CNY)
            tpl = random.choice(templates)
            queries.append(tpl.format(name=name, amount=amount, currency=currency))
        for amount in test_amounts[:6]:  # USD
            currency = random.choice(USD_CURRENCY_MARKERS)
            tpl = random.choice(templates)
            queries.append(tpl.format(name=name, amount=amount, currency=currency))

    # 额外补充边界与干扰用例
    extra = [
        # 原有边界用例
        "给赵六打 ¥100。",
        "帮我给王五转账 20 美元。",
        "给孙七转账¥88.8",
        "请给周八转 999 人民币。",
        "给老王（王五）转个 666 元吧",
        "帮我给张三打 98765.43 美元！十万火急！",
        "因为今天1号，给李四转 10 美元",
        "我已经转了2次了，这次再给赵六打 50 人民币",
        "帮吴九转 0.01 美元谢谢",
        "转账给郑十 1500 块钱，别搞错了",
        "马上给张三打$250 过去",
        "周八，帮我转 3.14 美元",
        # 新增：覆盖训练中的多样化模板格式
        "我要给刘洋汇 300 块钱。",
        "请打 150 美元 给赵敏，就这笔。",
        "上次说好要给马超转 88.88 元 的。",
        "麻烦转 250 块钱 给林志远，谢谢。",
        "现在立刻给陈小明转账 ¥500。",
        "扣除手续费后，给何秀英转 120 美元。",
        "帮我付黄丽 $75。",
        "帮我把钱转出给李娜，¥66.66，快点。",
        "对了，顺便给孙七转 ¥300 吧。",
    ]
    queries.extend(extra)

    random.shuffle(queries)
    return queries


# ── 指标计算 ──────────────────────────────────────────────
def amounts_equal(expected, actual):
    exp_norm = normalize_cny_amount(expected)
    act_norm = normalize_cny_amount(actual)
    return bool(exp_norm and act_norm and exp_norm == act_norm)


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


def evaluate_single(query, model, tokenizer, device):
    prompt = SYSTEM_PROMPT + f"\nUser: {query}\nAssistant: "
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=INFERENCE_CONFIG["max_new_tokens"],
        use_cache=True,
        do_sample=INFERENCE_CONFIG["do_sample"],
        temperature=INFERENCE_CONFIG["temperature"],
        top_p=INFERENCE_CONFIG["top_p"],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    expected_pinyin, expected_amount = expected_answer(query)
    params = extract_tool_data(response)

    actual_pinyin = (params.get("target_account") or "").strip().lower() if params else None
    actual_amount = params.get("amount") if params else None

    return {
        "query": query,
        "response": response,
        "format_ok": has_valid_r1_format(response),
        "pinyin_ok": actual_pinyin == expected_pinyin.lower() if expected_pinyin else False,
        "amount_ok": amounts_equal(expected_amount, actual_amount) if expected_amount else False,
        "expected_pinyin": expected_pinyin,
        "expected_amount": expected_amount,
        "actual_pinyin": actual_pinyin,
        "actual_amount": actual_amount,
        "is_usd": is_usd_query(query),
        "name": extract_name_from_query(query, PINYIN_MAP),
    }


# ── 模型加载与测试 ──────────────────────────────────────────
def load_model(model_name):
    torch_dtype = getattr(torch, INFERENCE_CONFIG["torch_dtype"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=INFERENCE_CONFIG["max_seq_length"],
        load_in_4bit=INFERENCE_CONFIG["load_in_4bit"],
        torch_dtype=torch_dtype,
    )
    FastLanguageModel.for_inference(model)
    model.generation_config.max_length = None  # 避免与 max_new_tokens 冲突的告警
    return model, tokenizer


def run_evaluation(model, tokenizer, device, queries, num_runs=3):
    total = len(queries)
    results = []

    for qi, query in enumerate(tqdm(queries), desc="Evaluating", unit="query"):
        run_records = []
        for _ in range(num_runs):
            rec = evaluate_single(query, model, tokenizer, device)
            run_records.append(rec)

        # 多数投票决定该样本是否通过
        format_ok = sum(r["format_ok"] for r in run_records) >= (num_runs // 2 + 1)
        pinyin_ok = sum(r["pinyin_ok"] for r in run_records) >= (num_runs // 2 + 1)
        amount_ok = sum(r["amount_ok"] for r in run_records) >= (num_runs // 2 + 1)

        results.append({
            "query": query,
            "name": run_records[0]["name"],
            "is_usd": run_records[0]["is_usd"],
            "expected_pinyin": run_records[0]["expected_pinyin"],
            "expected_amount": run_records[0]["expected_amount"],
            "format_ok": format_ok,
            "pinyin_ok": pinyin_ok,
            "amount_ok": amount_ok,
            "passed": format_ok and pinyin_ok and amount_ok,
            "run_details": [
                {
                    "actual_pinyin": r["actual_pinyin"],
                    "actual_amount": r["actual_amount"],
                    "format_ok": r["format_ok"],
                    "pinyin_ok": r["pinyin_ok"],
                    "amount_ok": r["amount_ok"],
                }
                for r in run_records
            ],
        })

    return results


def compute_metrics(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    format_ok = sum(1 for r in results if r["format_ok"])
    pinyin_ok = sum(1 for r in results if r["pinyin_ok"])
    amount_ok = sum(1 for r in results if r["amount_ok"])

    # 按货币类型分类
    cny_results = [r for r in results if not r["is_usd"]]
    usd_results = [r for r in results if r["is_usd"]]

    # 按名字分类
    by_name = {}
    for name in sorted(PINYIN_MAP.keys()):
        name_results = [r for r in results if r["name"] == name]
        if name_results:
            by_name[name] = {
                "total": len(name_results),
                "passed": sum(1 for r in name_results if r["passed"]),
                "format_ok": sum(1 for r in name_results if r["format_ok"]),
                "pinyin_ok": sum(1 for r in name_results if r["pinyin_ok"]),
                "amount_ok": sum(1 for r in name_results if r["amount_ok"]),
            }

    failed = [r for r in results if not r["passed"]]

    return {
        "overall": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0,
            "format_ok": format_ok,
            "format_rate": format_ok / total if total else 0,
            "pinyin_ok": pinyin_ok,
            "pinyin_rate": pinyin_ok / total if total else 0,
            "amount_ok": amount_ok,
            "amount_rate": amount_ok / total if total else 0,
        },
        "by_currency": {
            "cny": {
                "total": len(cny_results),
                "passed": sum(1 for r in cny_results if r["passed"]),
                "pass_rate": sum(1 for r in cny_results if r["passed"]) / len(cny_results) if cny_results else 0,
            },
            "usd": {
                "total": len(usd_results),
                "passed": sum(1 for r in usd_results if r["passed"]),
                "pass_rate": sum(1 for r in usd_results if r["passed"]) / len(usd_results) if usd_results else 0,
            },
        },
        "by_name": by_name,
        "failed_cases": [
            {
                "query": r["query"],
                "name": r["name"],
                "is_usd": r["is_usd"],
                "expected_pinyin": r["expected_pinyin"],
                "expected_amount": r["expected_amount"],
                "format_ok": r["format_ok"],
                "pinyin_ok": r["pinyin_ok"],
                "amount_ok": r["amount_ok"],
            }
            for r in failed[:20]  # 最多保留 20 个失败样本
        ],
    }


def print_report(base_metrics, ft_metrics, num_queries, num_runs):
    print("\n" + "=" * 60)
    print(f"  模型对比测试报告 ({num_queries} 条 × {num_runs} 轮)")
    print("=" * 60)

    def print_model(name, m):
        o = m["overall"]
        print(f"\n[{name}]")
        print(f"  综合通过率:  {o['passed']}/{o['total']} ({o['pass_rate']:.1%})")
        print(f"  R1 格式:     {o['format_ok']}/{o['total']} ({o['format_rate']:.1%})")
        print(f"  拼音提取:    {o['pinyin_ok']}/{o['total']} ({o['pinyin_rate']:.1%})")
        print(f"  金额计算:    {o['amount_ok']}/{o['total']} ({o['amount_rate']:.1%})")
        cny = m["by_currency"]["cny"]
        usd = m["by_currency"]["usd"]
        print(f"  CNY 用例:    {cny['passed']}/{cny['total']} ({cny['pass_rate']:.1%})")
        print(f"  USD 用例:    {usd['passed']}/{usd['total']} ({usd['pass_rate']:.1%})")
        print(f"  按人统计:")
        for name, stats in sorted(m["by_name"].items()):
            print(f"    {name}: {stats['passed']}/{stats['total']} "
                  f"(拼音: {stats['pinyin_ok']}/{stats['total']}, "
                  f"金额: {stats['amount_ok']}/{stats['total']})")
        if m["failed_cases"]:
            print(f"  失败样本 (前 5):")
            for fc in m["failed_cases"][:5]:
                print(f"    -> {fc['query']}")

    print_model("基座模型", base_metrics)
    print_model("微调模型", ft_metrics)

    # 对比摘要
    delta = ft_metrics["overall"]["pass_rate"] - base_metrics["overall"]["pass_rate"]
    print(f"\n  提升幅度: {delta:+.1%}")
    print("=" * 60)


def save_results(base_metrics, ft_metrics, base_name, ft_name, num_queries, num_runs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "num_queries": num_queries,
        "num_runs_per_query": num_runs,
        "base_model": base_name,
        "finetuned_model": ft_name,
        "base": base_metrics,
        "finetuned": ft_metrics,
        "delta_pass_rate": ft_metrics["overall"]["pass_rate"] - base_metrics["overall"]["pass_rate"],
    }
    save_dir = ft_name  # 放入对应的训练结果文件夹
    filename = os.path.join(save_dir, f"test_results_{timestamp}.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {filename}")


# ── 主流程 ──────────────────────────────────────────────────
def _find_latest_result():
    """在 results/ 中找到最新的 Agentic_R1_Lora_* 目录。"""
    if not os.path.isdir(RESULTS_BASE_DIR):
        return None
    candidates = [
        d for d in os.listdir(RESULTS_BASE_DIR)
        if d.startswith("Agentic_R1_Lora_") and os.path.isdir(os.path.join(RESULTS_BASE_DIR, d))
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return os.path.join(RESULTS_BASE_DIR, candidates[0])


def _resolve_model_path(user_input):
    """解析用户输入为实际模型路径。

    支持：完整的相对/绝对路径，或 results/ 下的子目录名。
    """
    if user_input is None:
        return None
    if os.path.isdir(user_input):
        return user_input
    inside_results = os.path.join(RESULTS_BASE_DIR, user_input)
    if os.path.isdir(inside_results):
        return inside_results
    print(f"错误: 找不到模型目录 '{user_input}'（或 '{inside_results}'）")
    sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="评估 Agentic R1 微调模型")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="微调模型路径：results/ 下的子目录名、相对路径、或绝对路径。不指定则使用最新结果。",
    )
    args = parser.parse_args()

    if args.model_path:
        ft_model_path = _resolve_model_path(args.model_path)
    else:
        ft_model_path = _find_latest_result()
        if ft_model_path is None:
            print(f"错误: 未找到微调模型。请在 {RESULTS_BASE_DIR}/ 下放置模型，或通过命令行参数指定路径。")
            sys.exit(1)
        print(f"自动选择最新模型: {ft_model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_runs = 3

    # 生成测试集
    test_queries = generate_test_queries(seed=123)
    print(f"测试集: {len(test_queries)} 条，每条件推理 {num_runs} 次\n")

    # ── 基座模型 ──
    print(f"加载基座模型: {TRAIN_MODEL_CONFIG['base_model_name']}")
    base_model, base_tokenizer = load_model(TRAIN_MODEL_CONFIG["base_model_name"])
    base_model.to(device)
    print("开始测试基座模型...")
    base_results = run_evaluation(base_model, base_tokenizer, device, test_queries, num_runs)
    base_metrics = compute_metrics(base_results)
    del base_model, base_tokenizer
    torch.cuda.empty_cache()

    # ── 微调模型 ──
    print(f"\n加载微调模型: {ft_model_path}")
    ft_model, ft_tokenizer = load_model(ft_model_path)
    ft_model.to(device)
    print("开始测试微调模型...")
    ft_results = run_evaluation(ft_model, ft_tokenizer, device, test_queries, num_runs)
    ft_metrics = compute_metrics(ft_results)
    del ft_model, ft_tokenizer
    torch.cuda.empty_cache()

    # ── 输出 ──
    print_report(base_metrics, ft_metrics, len(test_queries), num_runs)
    save_results(
        base_metrics, ft_metrics,
        TRAIN_MODEL_CONFIG["base_model_name"],
        ft_model_path,
        len(test_queries), num_runs,
    )


if __name__ == "__main__":
    main()
