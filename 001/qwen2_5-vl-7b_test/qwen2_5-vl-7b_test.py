import json
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
MODEL_PATH = "/data/bizq/LLaMA-Factory/output/qwen2_5_vl_lora_sft_002"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map='auto',
    use_cache=False
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

test_result = []
def input_model(image_path, item):
    QUESTION_TEMPLATE_ZH = "{Question} ，直接输出最终答案。"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_path}",
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE_ZH.format(Question=item['question'])
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

    img_think = {
        "image_path": item["image_path"],
        "question": item['question'],
        "think": output_text,
        "answer": item["answer"]
    }

    test_result.append(img_think)

test_json = r"/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/test_data.json"
img_folder = r"/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/test_images"
output_json = r"/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/output_002.json"

with open(test_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    img_path = os.path.join(img_folder, item["image_path"])
    input_model(img_path, item)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(test_result, f, ensure_ascii=False, indent=4)
