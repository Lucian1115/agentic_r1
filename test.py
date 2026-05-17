import os
import sys
import json
import random
import torch
from datetime import datetime
from tqdm import tqdm
from unsloth import FastLanguageModel
from config import (
    SYSTEM_PROMPT,
    PINYIN_MAP,
    USD_CURRENCY_MARKERS,
    CURRENCY_CNY,
    INFERENCE_CONFIG,
    TRAIN_MODEL_CONFIG,
    TEST_CONFIG,
    RESULTS_BASE_DIR,
)
from utils import (
    extract_tool_data,
    normalize_cny_amount,
    extract_name_from_query,
    extract_query_amount,
    has_valid_r1_format,
    is_usd_query,
    find_latest_lora_result,
    resolve_model_path,
    get_checkpoint_dir,
    get_best_ckpt_path,
    get_final_ckpt_path,
    amounts_equal,
    expected_answer,
)


# ── 测试集生成 ──────────────────────────────────────────────
def generate_test_queries(seed=None):
    """用固定种子生成覆盖全面的测试集，与训练数据生成逻辑解耦。"""
    cfg = TEST_CONFIG
    random.seed(seed if seed is not None else cfg["seed"])
    names = sorted(PINYIN_MAP.keys())
    templates = list(cfg["query_templates"])
    test_amounts = list(cfg["test_amounts"])
    num_amounts = cfg["num_amounts_per_currency"]

    queries = []

    # 每个名字 × 每个金额档位 × 每种货币 → 系统性覆盖
    for name in names:
        for amount in test_amounts[:num_amounts]:  # CNY
            currency = random.choice(CURRENCY_CNY)
            tpl = random.choice(templates)
            queries.append(tpl.format(name=name, amount=amount, currency=currency))
        for amount in test_amounts[:num_amounts]:  # USD
            currency = random.choice(USD_CURRENCY_MARKERS)
            tpl = random.choice(templates)
            queries.append(tpl.format(name=name, amount=amount, currency=currency))

    queries.extend(list(cfg["extra_queries"]))

    random.shuffle(queries)
    return queries


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
    model.generation_config.max_length = None
    return model, tokenizer


def run_evaluation(model, tokenizer, device, queries):
    results = []

    for query in tqdm(queries, desc="Evaluating", unit="query"):
        rec = evaluate_single(query, model, tokenizer, device)
        results.append({
            "query": query,
            "name": rec["name"],
            "is_usd": rec["is_usd"],
            "expected_pinyin": rec["expected_pinyin"],
            "expected_amount": rec["expected_amount"],
            "format_ok": rec["format_ok"],
            "pinyin_ok": rec["pinyin_ok"],
            "amount_ok": rec["amount_ok"],
            "passed": rec["format_ok"] and rec["pinyin_ok"] and rec["amount_ok"],
            "actual_pinyin": rec["actual_pinyin"],
            "actual_amount": rec["actual_amount"],
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
            for r in failed[:20]
        ],
    }


def print_report(model_results, num_queries):
    """打印多模型对比报告。

    model_results: list of (label, metrics) tuples，第一个为基座模型。
    """
    print("\n" + "=" * 60)
    model_labels = [label for label, _ in model_results]
    print(f"  模型对比测试报告 ({num_queries} 条, greedy decoding)")
    if len(model_labels) == 2:
        print(f"  对比: 基座 vs 微调")
    else:
        print(f"  对比: 基座 vs 最优 vs 最后")
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
        for n, stats in sorted(m["by_name"].items()):
            print(f"    {n}: {stats['passed']}/{stats['total']} "
                  f"(拼音: {stats['pinyin_ok']}/{stats['total']}, "
                  f"金额: {stats['amount_ok']}/{stats['total']})")
        if m["failed_cases"]:
            print(f"  失败样本 (前 5):")
            for fc in m["failed_cases"][:5]:
                print(f"    -> {fc['query']}")

    for label, metrics in model_results:
        print_model(label, metrics)

    # 对比摘要
    base_metrics = model_results[0][1]
    base_pass = base_metrics["overall"]["pass_rate"]
    for label, metrics in model_results[1:]:
        delta = metrics["overall"]["pass_rate"] - base_pass
        print(f"\n  [{label}] 较基座提升: {delta:+.1%}")
    print("=" * 60)


def save_results(base_metrics, ft_variants, base_name, output_dir, num_queries):
    """保存多模型对比结果。

    ft_variants: dict mapping label → (model_path, metrics)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "num_queries": num_queries,
        "base_model": base_name,
        "base": base_metrics,
        "finetuned_variants": {
            label: {
                "model_path": path,
                "metrics": metrics,
                "delta_pass_rate": metrics["overall"]["pass_rate"] - base_metrics["overall"]["pass_rate"],
            }
            for label, (path, metrics) in ft_variants.items()
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"test_results_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {filename}")


def _load_and_eval(model_path, device, test_queries, label):
    """加载模型、评估、卸载，返回 metrics。"""
    print(f"\n加载{label}: {model_path}")
    model, tokenizer = load_model(model_path)
    model.to(device)
    print(f"开始测试{label}...")
    results = run_evaluation(model, tokenizer, device, test_queries)
    metrics = compute_metrics(results)
    del model, tokenizer
    torch.cuda.empty_cache()
    return metrics


# ── 主流程 ──────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="评估 Agentic R1 微调模型（基座 vs 最优 vs 最后）")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="微调模型路径：results/ 下的子目录名、相对路径、或绝对路径。不指定则使用最新结果。",
    )
    args = parser.parse_args()

    if args.model_path:
        ft_output_dir = resolve_model_path(args.model_path, RESULTS_BASE_DIR)
    else:
        ft_output_dir = find_latest_lora_result(RESULTS_BASE_DIR)
        if ft_output_dir is None:
            print(f"错误: 未找到微调模型。请在 {RESULTS_BASE_DIR}/ 下放置模型，或通过命令行参数指定路径。")
            sys.exit(1)
        print(f"自动选择最新模型: {ft_output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 生成测试集
    test_queries = generate_test_queries()
    print(f"测试集: {len(test_queries)} 条\n")

    # ── 基座模型 ──
    base_name = TRAIN_MODEL_CONFIG["base_model_name"]
    base_metrics = _load_and_eval(base_name, device, test_queries, "基座模型")

    # ── 微调变体: 最优 checkpoint + 最后 checkpoint ──
    best_path = get_best_ckpt_path(ft_output_dir)
    final_path = get_final_ckpt_path(ft_output_dir)

    best_exists = os.path.isdir(best_path)
    final_exists = os.path.isdir(final_path)

    ft_variants = {}  # label → (path, metrics)
    model_results = [("基座模型", base_metrics)]  # for print_report

    if final_exists:
        metrics = _load_and_eval(final_path, device, test_queries, "最后模型")
        ft_variants["finetuned_final"] = (final_path, metrics)
        model_results.append(("最后模型 (final)", metrics))

    if best_exists:
        # 判断 best 是否与 final 相同（通过比较路径或直接覆盖 label）
        if final_exists and os.path.samefile(best_path, final_path):
            # best == final，不重复评估
            ft_variants["finetuned_best"] = (best_path, ft_variants["finetuned_final"][1])
        else:
            metrics = _load_and_eval(best_path, device, test_queries, "最优模型")
            ft_variants["finetuned_best"] = (best_path, metrics)
            # 如果 best != final，把 best 也插入 model_results（final 已在）
            if final_exists:
                model_results.insert(1, ("最优模型 (best)", metrics))
            else:
                model_results.append(("最优模型 (best)", metrics))

    if not ft_variants:
        print("警告: 未找到任何微调 checkpoint (best_model / final_model)，仅测试基座模型。")

    # ── 输出 ──
    print_report(model_results, len(test_queries))
    save_results(base_metrics, ft_variants, base_name, ft_output_dir, len(test_queries))


if __name__ == "__main__":
    main()
