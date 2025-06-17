from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
from pprint import pprint
import random
import pandas as pd

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:true'


MODEL_PATH="/data/bizq/LLaMA-Factory/output/qwen2_5_vl_lora_sft_001"
OUTPUT_PATH="./logs/rec_results_{}.json".format(os.path.basename(MODEL_PATH))
BSZ=8

device='cuda:4'



# TEST_DATASETS = ['refcoco_val', 'refcocop_val', 'refcocog_val']
# IMAGE_ROOT = "/data/shz/dataset/coco"


test_json_path = '/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/test_data.json'
IMAGE_ROOT = "/data/bizq/biziqi_002/test_model/qwen2_5-vl-7b_test/test_images"

# test_json_path = '/data03/sunwn/data/rlhf/AIOT_stage/test_data/output.json'
# IMAGE_ROOT = '/data03/sunwn/data/rlhf/AIOT_stage/test_data/images'

random.seed(42)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map='auto',
    use_cache=False
)

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="cuda:7"
# )

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)


# 关键修复：从处理器中获取分词器并设置 padding_side
tokenizer = processor.tokenizer
tokenizer.padding_side = "left"  # 设置填充方向为左侧

def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            x1, y1, x2, y2 = bbox
            return bbox, False
    return [0, 0, 0, 0], False


def extract_cls_answer(content):
    # print(content)
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if not content_answer_match:
        return None
    content_answer = content_answer_match.group(1).strip()
    content_answer=content_answer.replace('\n','')
    # print(content_answer_match)
    # print(content_answer)
    # 定义正则表达式，匹配整数和浮点数
    number_pattern = r"-?\d+\.?\d*"  # 匹配整数或浮点数，支持负数
    number_pattern = r"*(-?\d+\.?\d)*"
    # 使用 re.findall() 提取所有匹配的数值
    # number_answer_match = re.search(number_pattern, content_answer, re.DOTALL)
    # print(' number_answer_match',  number_answer_match)
    # number_answer = re.findall(r'\d+', content_answer)
    number_answer = re.findall(r'(-?\d+\.?\d*%?)', content_answer)
    if number_answer:
        number_answer=number_answer[0]

    if not number_answer:
        pattern = r'[A-Za-z%]*%[A-Za-z%]*'
        matches = re.findall(pattern, content_answer)
        if matches:
            number_answer=matches[0]

    if not number_answer:
        number_answer=content_answer

    # if number_answer_match:
    #     number_answer = number_answer_match.group(1).strip()
    # print('number_answer',number_answer)
# else:
    #     number_answer=None
    return number_answer


def extract_json_answer(content):
    # print(content)
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if not content_answer_match:
        return None
    content_answer = content_answer_match.group(1).strip()
    pattern = r'\{.*?\}'

    # Parse predicted JSON to get bbox list
    pred_bbox_data = {}
    json_match = re.findall(pattern, content_answer, re.DOTALL)
    if json_match:
        try:
            pred_bbox_data = json.loads(json_match[-1].strip())
        except:
            # Return empty list if JSON parsing fails
            pred_bbox_data = {}
    print('pred_bbox_data', pred_bbox_data)
    number_answer=None
    if 'text_content' in pred_bbox_data:
        number_answer=pred_bbox_data['text_content']

    return number_answer

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union



def extract_choice(content):
    if 'A' in content:
        return 'A'
    elif 'B' in content:
        return 'B'
    elif 'C' in content:
        return 'C'
    elif 'D' in content:
        return 'D'
    elif 'E' in content:
        return 'E'


ds_path = test_json_path
data = json.load(open(ds_path, "r"))
random.shuffle(data)
QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
output_format='请按照json格式返回，格式要求如下{"bbox_2d":[x1, y1, x2, y2], "text_content":text_content}'
data = data
messages = []

for x in data:
    image_path = os.path.join(IMAGE_ROOT, x['image_path'])
    message = [
        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"file://{image_path}"
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(Question=x['question'])
            }
        ]
    }]
    messages.append(message)

all_outputs = []  # List to store all answers

# Process data
for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i + BSZ]

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    all_outputs.extend(batch_output_text)
    # print(f"Processed batch {i//BSZ + 1}/{(len(messages) + BSZ - 1)//BSZ}")

final_output = []
correct_number = 0

for input_example, model_output in zip(data, all_outputs):
    original_output = model_output
    image_name=input_example['image_path']
    ground_truth = input_example['answer']
    # ground_truth_normalized = input_example['normalized_solution']
    # model_answer, normalized = extract_bbox_answer(original_output)
    model_answer = extract_cls_answer(original_output)
    # model_answer = extract_json_answer(original_output)
    print('model_answer',model_answer)
    print('ground_truth', ground_truth)
    #Count correct answers
    correct = 0
    # model_answer= extract_choice(original_output)

    if "°" in  ground_truth:
        ground_truth=ground_truth.replace('°','')
    if model_answer==ground_truth:
        correct=1

    correct_number += correct

    choice=extract_choice(original_output)

    # Create a result dictionary for this example
    result = {
        'image_name':image_name,
        'question': input_example['question'],
        'ground_truth': ground_truth,
        'model_output': original_output,
        'extracted_answer': model_answer,
        'correct': correct
    }
    print(result)
    final_output.append(result)

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_path, "w",encoding='utf-8') as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f,ensure_ascii=False,  indent=2)

print(f"Results saved to {output_path}")
print("-"*100)





