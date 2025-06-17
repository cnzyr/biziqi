import json
import re

# 读取 JSON 文件
with open(r"D:\10.0.79.35\t_bridge_cot-sft\t-bridge-cot-sft-0528_003.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 遍历数据
for item in data:
    # 获取 conversations[1]["value"]
    value = item["conversations"][1]["value"]

    # 提取 <answer> 内容
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, value)
    if answer_match:
        answer_ = answer_match.group(1)
        answer_ = " ".join(answer_.split())  # 清理多余空格
    else:
        print(f"Warning: No <answer> tag found in item {item['id']}")
        answer_ = ""  # 设置默认值或根据需求处理

    # 提取 <think> 内容
    think_pattern = r"<think>\s*(.*?)\s*</think>"
    think_match = re.search(think_pattern, value, re.DOTALL | re.MULTILINE)
    if think_match:
        think_ = think_match.group(1)
    else:
        print(f"Warning: No <think> tag found in item {item['id']}")
        think_ = ""  # 设置默认值或根据需求处理

    # 更新 conversations[1]["value"]
    item["conversations"][1]["value"] = f"<think>{think_}</think><answer>{answer_}</answer>"

# 保存修改后的 JSON 文件
with open(r"D:\10.0.79.35\t_bridge_cot-sft\t-bridge-cot-sft-0528_004.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)