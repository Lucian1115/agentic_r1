import os

RESULTS_BASE_DIR = "results"

SYSTEM_PROMPT = """你是一个严谨的API智能体。请仔细阅读用户的请求，判断是否需要调用工具。
规则：
1. 如果用户的请求不是转账操作，直接礼貌地回复说明即可，不要调用工具。
2. 如果是转账操作，必须先写在 <think> ... </think> 标签内进行推理，且必须包含明确的计算或换算过程，不能只复述数字。
3. 最终的工具调用必须以 JSON 格式输出在 <tool_call> ... </tool_call> 标签内。

【可用工具：transfer_funds】
参数约束：
- target_account (string): 收款人拼音全拼。必须把中文名转为拼音全拼。
- amount (string): 金额。最终输出必须显式以 "¥" 开头，不接受纯数字、"人民币" 或 "CNY" 作为最终格式。如果用户输入包含美元，必须按 1 美元 = 7 人民币换算后再输出。
"""

PINYIN_MAP = {
    # 原有人名
    "张三": "zhangsan",
    "李四": "lisi",
    "王五": "wangwu",
    "赵六": "zhaoliu",
    "孙七": "sunqi",
    "周八": "zhouba",
    "吴九": "wujiu",
    "郑十": "zhengshi",
    # 新增：常见真实姓名，覆盖不同姓氏和名字结构
    "张伟": "zhangwei",
    "李娜": "lina",
    "王建国": "wangjianguo",
    "赵敏": "zhaomin",
    "陈小明": "chenxiaoming",
    "刘洋": "liuyang",
    "黄丽": "huangli",
    "林志远": "linzhiyuan",
    "马超": "machao",
    "何秀英": "hexiuying",
}

# 人名称呼前缀（用于匹配和生成）
NAME_PREFIXES = ("老", "小", "大", "阿")

USD_CURRENCY_MARKERS = ("美元", "美金", "美刀", "$", "usd", "USD")
CURRENCY_CNY = ("块钱", "元", "人民币")

# 美元到人民币汇率
USD_TO_CNY_RATE = 7

REGEX_CONFIG = {
    # 匹配示例中实际包含的换行与空白字符（修正了原来字符串中双重转义的问题）
    "user_line_pattern": r"User:\s*(.*?)(?:\n|$)",
    # 支持整数与小数金额（如 20、¥20、20.5、20元、20美元）
    # 匹配动词后的金额，动词覆盖：转账、转、打、汇、付、转出
    # .*? 允许动词与金额之间有其他文字（如 "付张三 100"）
    "query_amount_pattern": r"(?:转账|转出|打给|打|汇|付给|付|转给|转).*?([0-9]+(?:\.[0-9]+)?)\s*(?:[¥￥$]|人民币|元|块钱|美元|美金|美刀)?",
    "tool_call_pattern": r"<tool_call>(.*?)</tool_call>",
    "think_block_pattern": r"<think>(.*?)</think>",
    # 识别汇率转换的提示，需要与汇率相关的词组配合以避免误匹配
    # 更通用地匹配汇率提示（允许浮点数、等号形式，以及中英混合写法）
    # 例如: "汇率 1.0", "1美元=7元", "1 USD = 7 CNY", "汇率为7" 等
    "exchange_hint_pattern": r"(?:汇率|汇兑|汇换|转换|×|乘)\s*(?:是|为|=|：|:)?\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:USD|usd|美元)\s*[=:]\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:CNY|cny|人民币)\s*[=:]\s*\d+(?:\.\d+)?",
}

DATASET_CONFIG = {
    "random_seed": 42,
    "num_samples": 1200,               # 增加样本量：配合更多训练步数
    "amount_min": 10,
    "amount_max": 500,
    "usd_probability": 0.40,           # 从 0.50 降至 0.40：增加 CNY 占比，减少模型的"换算惯性"
    # 小数金额概率（如 123.45）
    "decimal_amount_probability": 0.2,
    # 使用 ¥/$ 前缀的概率（如 ¥100、$50）
    "amount_prefix_probability": 0.15,
    # 使用中文大写/口语金额的概率（如 "一百块"）
    "chinese_amount_probability": 0.05,
    # 名字带称呼前缀的概率（如 "老王"、"小张"）。仅当姓氏在 PINYIN_MAP 中唯一时生效
    "name_prefix_probability": 0.1,
    # 称呼前缀候选列表
    "name_prefixes": ("老", "小"),
    # 负样本概率：无转账意图的查询，防止盲目输出 tool_call
    "negative_sample_probability": 0.08,  # 从 0.02 增至 0.08
    # 名字消歧样本概率（如 "注意不是李四"）
    "disambiguation_probability": 0.08,
    # 困难名字过采样：30% 的采样从困难名字池中均匀抽取
    "hard_name_sample_probability": 0.3,
    # 困难名字过采样倍率（拼音通过率 < 50% 的名字采样权重翻倍）
    "hard_name_oversample_ratio": 2.0,
    "hard_names": ["李娜", "李四", "王建国", "何秀英", "赵六", "马超"],
    "query_templates": (
        # 标准转账表达
        "帮我给{name}转账 {amount_str}。",
        "请给{name}转 {amount_str}。",
        "给{name}打 {amount_str}。",
        "麻烦给{name}转账 {amount_str}。",
        # 口语化表达
        "转 {amount_str} 给{name}。",
        "{name}，给你转 {amount_str} 过去。",
        "帮我打 {amount_str} 到{name}账户。",
        "我要给{name}汇 {amount_str}。",
        "帮我付{name} {amount_str}。",
        "麻烦转 {amount_str} 给{name}，谢谢。",
        # 带额外语境的表达
        "对了，顺便给{name}转 {amount_str} 吧。",
        "请打 {amount_str} 给{name}，就这笔。",
        "上次说好要给{name}转 {amount_str} 的。",
        "现在立刻给{name}转账 {amount_str}。",
        "帮我把钱转出给{name}，{amount_str}，快点。",
        # 带干扰信息的表达
        "刚才那笔不算，再给{name}转 {amount_str} 过去。",
        "本来要转上回的，这次先给{name}打 {amount_str}。",
        "重要提醒：给{name}转账 {amount_str}，别忘了。",
        "扣除手续费后，给{name}转 {amount_str}。",
        "已收到通知，给{name}转账 {amount_str}。",
        # 显式 CNY 无需换算的表达
        "这是人民币，不用换算，直接给{name}转 {amount_str}。",
        "对方只收人民币，给{name}打 {amount_str} 就行。",
        "国内转账，给{name}转 {amount_str}，不需要汇率换算。",
    ),
    # 无转账意图的查询模板（用于负例）
    "negative_templates": (
        "我这个月已经转了多少钱了？",
        "查看{name}的账户余额。",
        "帮我查一下给{name}的转账记录。",
        "转账的最大限额是多少？",
        "最近的汇率是多少？",
        "帮我看看{name}有没有收到钱。",
        # 新增：带数字但非金额的干扰句
        "今天{name}第3次问我了。",
        "上次那两笔给{name}的转账确认了没？",
        "{name}的账号是6222开头的那张卡。",
        "汇率今天到7.2了，但我只是问问。",
        "给{name}发个消息确认一下。",
    ),
    # 名字消歧模板
    "disambiguation_templates": (
        "给{name}转 {amount_str}，注意不是{confusable}。",
        "给{name}（不是{confusable}）转账 {amount_str}。",
        "这次给{name}转 {amount_str}，别和{confusable}搞混了。",
    ),
    # 中文口语金额映射
    "chinese_numeral_map": {
        10: "十块", 20: "二十块", 50: "五十块", 100: "一百块",
        200: "两百块", 300: "三百块", 500: "五百块",
        1000: "一千块", 5000: "五千块", 10000: "一万块",
    },
    # 共享姓氏的名字对（用于消歧模板）
    "confusable_name_pairs": {
        "张三": "张伟", "张伟": "张三",
        "李四": "李娜", "李娜": "李四",
        "王五": "王建国", "王建国": "王五",
        "赵六": "赵敏", "赵敏": "赵六",
    },
}

REWARD_WEIGHTS = {
    "format": 0.15,         # 降低：格式在早期就会饱和，减少死梯度比例
    "pinyin": 0.35,         # 新增：拼音部分匹配奖励（最难问题，给最大权重）
    "amount": 0.25,         # 新增：金额接近匹配奖励
    "self_correction": 0.25, # 提高：整个训练周期都有用
}

TRAIN_MODEL_CONFIG = {
    "base_model_name": "./Qwen2.5-3B-Instruct-Local",
    "max_seq_length": 1536,
    "load_in_4bit": False,
    "torch_dtype": "bfloat16",
}

LORA_CONFIG = {
    "r": 16,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": False,
}

GRPO_TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",     # 单次余弦衰减：平滑，避免 cosine_with_restarts 在 step 775 处的 LR 突跳导致 KL 飙升
    "warmup_steps": 50,                # 线性预热：0 → 5e-6，避免初期随机策略产生过大梯度
    "logging_steps": 1,
    "bf16": True,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_generations": 8,
    "generation_batch_size": 8,
    "max_prompt_length": 640,
    "max_completion_length": 768,
    "max_steps": 1500,
    "output_dir": "Agentic_R1_Lora",
    "report_to": "wandb",
    # GRPO-specific: KL 正则化与日志
    "beta": 0.06,
    "log_completions": False,
    "wandb_log_unique_prompts": False,
    "generation_kwargs": {"max_length": None},
    "log_level": "warning",
    # ── 评估与 checkpoint ──
    "eval_steps": 50,                    # 轻量级评估间隔
    "eval_max_new_tokens": 256,          # 贪婪解码最大 token 数（评测用）
    "kl_warning_threshold": 0.08,        # KL 超此值记入本地日志（策略偏离过大）
    # ── 本地日志采样间隔 ──
    "completions_log_steps": 50,         # completion 样本保存间隔
}

INFERENCE_CONFIG = {
    "model_name": "./Agentic_R1_Lora",
    "max_seq_length": 1536,
    "load_in_4bit": False,
    "torch_dtype": "bfloat16",
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 0.1,
    "top_p": 0.9,
}

EXPORT_CONFIG = {
    "source_model_name": "./Agentic_R1_Lora",
    "max_seq_length": 1536,
    "load_in_4bit": False,
    "torch_dtype": "bfloat16",
    "merged_output_dir": "Agentic_R1_Merged_16bit",
}

TEST_CONFIG = {
    "seed": 42,
    "num_runs": 1,
    "query_templates": (
        "帮我给{name}转账 {amount} {currency}。",
        "请给{name}转 {amount} {currency}。",
        "给{name}打 {amount} {currency}。",
        "麻烦给{name}转账 {amount} {currency}。",
    ),
    "test_amounts": [
        10, 25, 50, 100, 200, 500, 999, 123.45, 88.8, 0.5, 10000, 76543.21
    ],
    "num_amounts_per_currency": 6,
    # 补充边界与干扰用例
    "extra_queries": [
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
        "我要给刘洋汇 300 块钱。",
        "请打 150 美元 给赵敏，就这笔。",
        "上次说好要给马超转 88.88 元 的。",
        "麻烦转 250 块钱 给林志远，谢谢。",
        "现在立刻给陈小明转账 ¥500。",
        "扣除手续费后，给何秀英转 120 美元。",
        "帮我付黄丽 $75。",
        "帮我把钱转出给李娜，¥66.66，快点。",
        "对了，顺便给孙七转 ¥300 吧。",
    ],
    # 训练中周期性评估的轻量测试集（20条，覆盖关键场景）
    "eval_queries": [
        # Easy names, CNY
        "帮我给张三转账 100 块钱。",
        "请给李四转 50 元。",
        # Hard names, CNY
        "给李娜打 200 人民币。",
        "麻烦给王建国转账 150 块钱。",
        "转 300 元 给何秀英。",
        "帮我付马超 120 块钱。",
        # USD conversion
        "给张三转账 20 美元。",
        "帮李四打 100 美元。",
        "转 15 美元 给李娜。",
        "给王建国打 30 美元。",
        # Decimal amounts
        "给赵六转账 ¥123.45。",
        "帮孙七转 88.8 元。",
        # Prefix names
        "给老张转 300 块钱。",
        "帮小刘转 50 美元。",
        # Edge cases
        "给黄丽打 $50。",
        "麻烦转 250 块钱 给林志远。",
        # Explicit CNY (no conversion needed)
        "这是人民币，不用换算，直接给孙七转 200 块钱。",
        "对方只收人民币，给周八打 175 元 就行。",
        # Negative samples (should not output tool_call)
        "查看张三的账户余额。",
        "帮我查一下给李四的转账记录。",
    ],
}