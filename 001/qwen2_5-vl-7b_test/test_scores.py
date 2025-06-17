import json
import re

json_file = r"D:\10.0.79.35\qwen2_5-vl-7b_test\output_002_1.json"

with open(json_file, "r", encoding='utf-8') as f:
    data = json.load(f)

count = 0
for item in data:
    answer = item["answer"]
    model_think = item["think"][0]

    if "<answer>" in model_think:
        answer_pattern = r"<answer>(.*?)</answer>"
        model_answer = re.search(answer_pattern, model_think)
        if model_answer:
            answer_ = model_answer.group(1)
            print(answer_)
        else:
            answer_ = model_think
    else:
        answer_ = model_think

    if answer in answer_:
        count += 1

print(count)
accuracy = (count / len(data)) * 100
print(f"\nAccuracy: {accuracy:.2f}%")