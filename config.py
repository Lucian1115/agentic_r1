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
    "num_samples": 1000,
    "amount_min": 10,
    "amount_max": 500,
    "usd_probability": 0.5,
    # 小数金额概率（如 123.45）
    "decimal_amount_probability": 0.2,
    # 使用 ¥/$ 前缀的概率（如 ¥100、$50）
    "amount_prefix_probability": 0.15,
    # 使用中文大写/口语金额的概率（如 "一百块"）
    "chinese_amount_probability": 0.05,
    # 名字带称呼前缀的概率（如 "老王"、"小张"）
    "name_prefix_probability": 0.1,
    # 负样本概率：无转账意图的查询，防止盲目输出 tool_call
    "negative_sample_probability": 0.02,
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
    ),
    # 无转账意图的查询模板（用于负例）
    "negative_templates": (
        "我这个月已经转了多少钱了？",
        "查看{name}的账户余额。",
        "帮我查一下给{name}的转账记录。",
        "转账的最大限额是多少？",
        "最近的汇率是多少？",
        "帮我看看{name}有没有收到钱。",
    ),
    # 中文口语金额映射
    "chinese_numeral_map": {
        10: "十块", 20: "二十块", 50: "五十块", 100: "一百块",
        200: "两百块", 300: "三百块", 500: "五百块",
        1000: "一千块", 5000: "五千块", 10000: "一万块",
    },
}

REWARD_WEIGHTS = {
    "format": 0.2,
    "constraint": 0.6,
    "self_correction": 0.2,
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
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "bf16": True,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_generations": 16,
    "generation_batch_size": 16,
    "max_prompt_length": 640,
    "max_completion_length": 768,
    "max_steps": 500,
    "output_dir": "Agentic_R1_Lora",  # 基名，train.py 中会追加时间戳并放入 results/
    "report_to": "wandb",
    # GRPO-specific: KL regularization & logging
    "beta": 0.04,
    "log_completions": False,  # 关闭每步的 completion 表格，改用 callback 抽样记录
    "wandb_log_unique_prompts": True,
    # 消除每步的 max_new_tokens vs max_length 冲突告警
    "generation_kwargs": {"max_length": None},
    "log_level": "warning",
}

INFERENCE_CONFIG = {
    "model_name": "./Agentic_R1_Lora",
    "max_seq_length": 1536,
    "load_in_4bit": False,
    "torch_dtype": "bfloat16",
    "max_new_tokens": 256,
    "do_sample": True,
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