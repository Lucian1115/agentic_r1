import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# === 配置中文字体 ===
font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
font_family = prop.get_name()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [font_family, "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 12,
})

# === 加载数据 ===
json_path = "results/Agentic_R1_Lora_20260517_121347/test_results_20260518_062511.json"
with open(json_path, "r") as f:
    data = json.load(f)

base = data["base"]["overall"]
final = data["finetuned_variants"]["finetuned_final"]["metrics"]["overall"]

metrics_labels = ["总通过率", "格式正确率", "拼音正确率", "金额正确率"]
metrics_keys = ["pass_rate", "format_rate", "pinyin_rate", "amount_rate"]

base_vals = [base[k] * 100 for k in metrics_keys]
final_vals = [final[k] * 100 for k in metrics_keys]

# === 绘图 ===
fig, ax = plt.subplots(figsize=(8, 5.5))

x = np.arange(len(metrics_labels))
width = 0.32

bars1 = ax.bar(x - width / 2, base_vals, width, label="Qwen2.5-3B-Instruct",
               color="#7f7f7f", edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x + width / 2, final_vals, width, label="GRPO 微调后",
               color="#4472C4", edgecolor="black", linewidth=0.6)

for bar, val in zip(bars1, base_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
for bar, val in zip(bars2, final_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

for i, (bv, fv) in enumerate(zip(base_vals, final_vals)):
    delta = fv - bv
    ax.annotate(f"+{delta:.1f}%", xy=(x[i] + width / 2, fv + 6.5), fontsize=9,
                ha="center", va="bottom", color="#c0392b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#fdecea",
                          edgecolor="#e74c3c", linewidth=0.5))

ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=13)
ax.set_ylabel("通过率 (%)", fontsize=13)
ax.set_ylim(0, 112)
ax.legend(loc="upper left", fontsize=11, frameon=True, edgecolor="black", fancybox=False)

# 装饰
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = "results/Agentic_R1_Lora_20260517_121347/model_comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"图片已保存至: {out_path}")
