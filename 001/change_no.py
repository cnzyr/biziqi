import json
import random
import re

json_file = r"D:\10.0.79.35\t_bridge_cot-sft\t-bridge-cot-sft-0519_005.json"

with open(json_file, "r", encoding='utf-8') as f:
    data = json.load(f)

for d in data:
    value = d["conversations"][1]["value"]
    text_boxes = re.findall(r"{\"bbox_2d\": \[(\d+), (\d+), (\d+), (\d+)\], \"text_content\": \"([^\"]+)\"}",value)
    answer = re.search(r"<answer>(.*?)</answer>", d["conversations"][1]["value"]).group(1)
    if len(text_boxes) >= 2:
        if text_boxes[0][4] == answer:
            i = random.choice([1,2,3,4,5])
            while i >= len(text_boxes):
                i = i - 1
            text_boxes[0], text_boxes[i] = text_boxes[i], text_boxes[0]
    print(text_boxes, answer)