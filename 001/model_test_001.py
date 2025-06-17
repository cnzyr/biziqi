from lmdeploy.serve.openai.api_client import APIClient
import os
import json

# 推理测试集
test_result = []
def img_test_json(image_path, item):
    img_think = {}
    api_client = APIClient(f'http://0.0.0.0:23333')
    model_name = '/data/bizq/LLaMA-Factory/output/qwen2_5_vl_lora_sft_001'
    QUESTION_TEMPLATE = "<image>{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    messages = [
        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=item['question'])
            },
            {
                "type": "image",
                "image": f"file:{image_path}"
            }

        ]
    }]
    for i in api_client.chat_completions_v1(model=model_name,messages=messages):
        result = i["choices"][0]["message"]["content"]
        print(result)

    img_think = {
        "image_path": item["image_path"],
        "question": item['question'],
        "think": result,
        "answer": item["answer"]
    }

    test_result.append(img_think)


imgae_folder = "/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/test_images"
output_file = "/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/output.json"
json_file = r"/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/test_data.json"
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    image_path = os.path.join(imgae_folder, item['image_path'])
    img_test_json(image_path, item)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False)