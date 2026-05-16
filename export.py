import os
import sys
import torch
from datetime import datetime
from unsloth import FastLanguageModel
from config import EXPORT_CONFIG, RESULTS_BASE_DIR


def _find_latest_lora():
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


def _resolve_source(user_input):
    """解析用户输入为实际模型路径。"""
    if os.path.isdir(user_input):
        return user_input
    inside_results = os.path.join(RESULTS_BASE_DIR, user_input)
    if os.path.isdir(inside_results):
        return inside_results
    print(f"错误: 找不到模型目录 '{user_input}'（或 '{inside_results}'）")
    sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="合并 LoRA 权重并导出 16-bit 模型")
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="LoRA 模型路径：results/ 下的子目录名、相对路径、或绝对路径。不指定则使用最新结果。",
    )
    args = parser.parse_args()

    if args.source:
        source_path = _resolve_source(args.source)
    else:
        source_path = _find_latest_lora()
        if source_path is None:
            print(f"错误: 未找到 LoRA 模型。请在 {RESULTS_BASE_DIR}/ 下放置模型，或通过命令行参数指定路径。")
            sys.exit(1)
        print(f"自动选择最新模型: {source_path}")

    # 导出目标：results/Agentic_R1_Merged_16bit_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_dir = os.path.join(RESULTS_BASE_DIR, f"Agentic_R1_Merged_16bit_{timestamp}")

    print(f"加载模型: {source_path}")
    torch_dtype = getattr(torch, EXPORT_CONFIG["torch_dtype"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_path,
        max_seq_length=EXPORT_CONFIG["max_seq_length"],
        load_in_4bit=EXPORT_CONFIG["load_in_4bit"],
        torch_dtype=torch_dtype,
    )

    print(f"合并权重并导出为 16-bit 模型 -> {merged_dir}")
    try:
        from unsloth import unsloth_save_model_merged_16bit
        unsloth_save_model_merged_16bit(model, tokenizer, merged_dir)
    except (ImportError, AttributeError):
        print("使用标准的 save_pretrained 方法导出模型...")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    print(f"模型合并完成，已导出到: {merged_dir}")


if __name__ == "__main__":
    main()
