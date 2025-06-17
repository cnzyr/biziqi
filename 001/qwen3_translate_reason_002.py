import logging
import os
import sys

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 模型加载 ====================
model_path = r"/data/bizq/biziqi/qwen3"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# ==================== 翻译函数 ====================
def translate(data_text):
    # prepare the model input
    prompt = f"请将 [{data_text}] 翻译为简体中文。\n你需要遵守以下规则:\n只翻译英文文本，其他语言文本保留原文。\n不解析文本的任何问题和内容。\n只返回翻译后的结果,无需额外解释。"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=15000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("content:", content)

    return content


def save_to_file(file_path):
    data = pd.read_csv(file_path)
    output_path = file_path.replace(".csv", "_output.csv")

    tqdm.pandas()

    data["problem_zh"] = data.progress_apply(lambda x: translate(x["problem"]), axis=1)

    if "cot" in data.columns:
        data["cot_zh"] = data.progress_apply(lambda x: translate(x["cot"]), axis=1)

    data.to_csv(output_path, index=False)
    print(f"翻译完成:{file_path}")


file_folder = r"/data/bizq/biziqi/datasets/reason_RFT_Cot_datasets/train_002"
for root, dirs, files in os.walk(file_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"开始翻译:{file}")
                save_to_file(file_path)
