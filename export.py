import os
import sys
import torch
from datetime import datetime
from unsloth import FastLanguageModel
from config import EXPORT_CONFIG, RESULTS_BASE_DIR
from utils import (
    find_latest_lora_result,
    resolve_model_path,
    get_best_ckpt_path,
    get_final_ckpt_path,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="合并 LoRA 权重并导出 16-bit 模型")
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="训练输出目录：results/ 下的子目录名、相对路径、或绝对路径。不指定则使用最新结果。",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        choices=["best", "final"],
        help="要导出的 checkpoint: best（最优）或 final（最终）",
    )
    args = parser.parse_args()

    # 允许省略 source 直接传 checkpoint，如: python export.py best
    if args.source in ("best", "final") and args.checkpoint is None:
        args.checkpoint = args.source
        args.source = None
    elif args.checkpoint is None:
        parser.error("必须指定 checkpoint: best 或 final")

    if args.source:
        base_dir = resolve_model_path(args.source, RESULTS_BASE_DIR)
    else:
        base_dir = find_latest_lora_result(RESULTS_BASE_DIR)
        if base_dir is None:
            print(f"错误: 未找到训练输出目录。请在 {RESULTS_BASE_DIR}/ 下放置模型，或通过命令行参数指定路径。")
            sys.exit(1)
        print(f"自动选择最新训练输出: {base_dir}")

    source_path = get_best_ckpt_path(base_dir) if args.checkpoint == "best" else get_final_ckpt_path(base_dir)
    label = args.checkpoint

    if not os.path.isdir(source_path):
        print(f"错误: checkpoint 目录不存在: {source_path}")
        sys.exit(1)

    # 导出目标：results/Agentic_R1_Merged_16bit_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_dir = os.path.join(RESULTS_BASE_DIR, f"Agentic_R1_Merged_{label}_{timestamp}")

    print(f"加载{label}模型: {source_path}")
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

    print(f"{label}模型合并完成，已导出到: {merged_dir}")


if __name__ == "__main__":
    main()
