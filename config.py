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
    "张三": "zhangsan",
    "李四": "lisi",
    "王五": "wangwu",
    "赵六": "zhaoliu",
    "孙七": "sunqi",
    "周八": "zhouba",
    "吴九": "wujiu",
    "郑十": "zhengshi",
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

NAME_PREFIXES = ("老", "小", "大", "阿")

USD_CURRENCY_MARKERS = ("美元", "美金", "美刀", "$", "usd", "USD")
CURRENCY_CNY = ("块钱", "元", "人民币")

# 美元到人民币汇率
# 1 USD = 7 CNY
USD_TO_CNY_RATE = 7

REGEX_CONFIG = {
    "user_line_pattern": r"User:\s*(.*?)(?:\n|$)",
    # 动词后匹配金额数字，.*? 允许中间插入其他文字（如"付张三 100"）
    "query_amount_pattern": r"(?:转账|转出|打给|打|汇|付给|付|转给|转).*?([0-9]+(?:\.[0-9]+)?)\s*(?:[¥￥$]|人民币|元|块钱|美元|美金|美刀)?",
    "tool_call_pattern": r"<tool_call>(.*?)</tool_call>",
    "think_block_pattern": r"<think>(.*?)</think>",
    # 匹配汇率提示：中英混合、等号形式、浮点数等
    "exchange_hint_pattern": r"(?:汇率|汇兑|汇换|转换|×|乘)\s*(?:是|为|=|：|:)?\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:USD|usd|美元)\s*[=:]\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:CNY|cny|人民币)\s*[=:]\s*\d+(?:\.\d+)?",
}

DATASET_CONFIG = {
    "random_seed": 42,
    "num_samples": 1200,
    "amount_min": 10,
    "amount_max": 500,
    "usd_probability": 0.40,
    "decimal_amount_probability": 0.2,
    "amount_prefix_probability": 0.15,
    "chinese_amount_probability": 0.05,
    # 名字加称呼前缀的概率（仅姓氏唯一时生效）
    "name_prefix_probability": 0.1,
    "name_prefixes": ("老", "小"),
    # 负样本/消歧样本概率
    "negative_sample_probability": 0.08,
    "disambiguation_probability": 0.08,
    # 困难名字过采样（拼音通过率低的样本加权）
    "hard_name_sample_probability": 0.3,
    "hard_name_oversample_ratio": 2.0,
    "hard_names": ["李娜", "李四", "王建国", "何秀英", "赵六", "马超"],
    "query_templates": (
        "帮我给{name}转账 {amount_str}。",
        "请给{name}转 {amount_str}。",
        "给{name}打 {amount_str}。",
        "麻烦给{name}转账 {amount_str}。",
        "转 {amount_str} 给{name}。",
        "{name}，给你转 {amount_str} 过去。",
        "帮我打 {amount_str} 到{name}账户。",
        "我要给{name}汇 {amount_str}。",
        "帮我付{name} {amount_str}。",
        "麻烦转 {amount_str} 给{name}，谢谢。",
        "对了，顺便给{name}转 {amount_str} 吧。",
        "请打 {amount_str} 给{name}，就这笔。",
        "上次说好要给{name}转 {amount_str} 的。",
        "现在立刻给{name}转账 {amount_str}。",
        "帮我把钱转出给{name}，{amount_str}，快点。",
        # 以下带干扰信息或显式声明无需换算
        "刚才那笔不算，再给{name}转 {amount_str} 过去。",
        "本来要转上回的，这次先给{name}打 {amount_str}。",
        "重要提醒：给{name}转账 {amount_str}，别忘了。",
        "扣除手续费后，给{name}转 {amount_str}。",
        "已收到通知，给{name}转账 {amount_str}。",
        "这是人民币，不用换算，直接给{name}转 {amount_str}。",
        "对方只收人民币，给{name}打 {amount_str} 就行。",
        "国内转账，给{name}转 {amount_str}，不需要汇率换算。",
    ),
    "negative_templates": (
        "我这个月已经转了多少钱了？",
        "查看{name}的账户余额。",
        "帮我查一下给{name}的转账记录。",
        "转账的最大限额是多少？",
        "最近的汇率是多少？",
        "帮我看看{name}有没有收到钱。",
        "今天{name}第3次问我了。",
        "上次那两笔给{name}的转账确认了没？",
        "{name}的账号是6222开头的那张卡。",
        "汇率今天到7.2了，但我只是问问。",
        "给{name}发个消息确认一下。",
    ),
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
    "format": 0.15,
    "pinyin": 0.35,
    "amount": 0.25,
    "self_correction": 0.25,
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
    "lr_scheduler_type": "cosine",  # 避免 cosine_with_restarts 在 step 775 LR 突跳导致 KL 飙升
    "warmup_steps": 50,
    "logging_steps": 1,
    "bf16": True,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_generations": 8,
    "generation_batch_size": 8,
    "max_prompt_length": 640,
    "max_completion_length": 768,
    "max_steps": 1000, # 1500
    "output_dir": "Agentic_R1_Lora",
    "report_to": "wandb",
    "beta": 0.06,
    "log_completions": False,
    "wandb_log_unique_prompts": False,
    "generation_kwargs": {"max_length": None},
    "log_level": "warning",
    "eval_steps": 50,
    "eval_max_new_tokens": 256,
    "kl_warning_threshold": 0.08,
    "completions_log_steps": 50,
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
    "eval_queries": [
        "帮我给张三转账 100 块钱。",
        "请给李四转 50 元。",
        "给李娜打 200 人民币。",
        "麻烦给王建国转账 150 块钱。",
        "转 300 元 给何秀英。",
        "帮我付马超 120 块钱。",
        "给张三转账 20 美元。",
        "帮李四打 100 美元。",
        "转 15 美元 给李娜。",
        "给王建国打 30 美元。",
        "给赵六转账 ¥123.45。",
        "帮孙七转 88.8 元。",
        "给老张转 300 块钱。",
        "帮小刘转 50 美元。",
        "给黄丽打 $50。",
        "麻烦转 250 块钱 给林志远。",
        "这是人民币，不用换算，直接给孙七转 200 块钱。",
        "对方只收人民币，给周八打 175 元 就行。",
        # 负样本
        "查看张三的账户余额。",
        "帮我查一下给李四的转账记录。",
    ],
}
