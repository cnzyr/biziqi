import base64
import json
import os
import time
import requests

all_results = []
total_time = 0  # 记录创建任务的总时间
api_calls = 0  # 记录API调用次数


# API请求
def dino_API(img_name, img_path):
    # 请求头
    global item, total_time, api_calls
    headers = {
        "Content-Type": "application/json",
        "Token":
    }

    # 请求参数
    json_data = {
        "model": "GroundingDino-1.6-Pro",
        "image": f"data:image/png;base64,{img_to_base64(img_path)}",
        "prompt": {
            "type": "text",
            "text": "cave",
        },
        "targets": ["bbox"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8
    }

    # 记录创建任务开始时间
    start_time = time.time()

    # 创建任务接口
    resp = requests.post(
        url="https://api.deepdataspace.com/v2/task/grounding_dino/detection",
        headers=headers,
        json=json_data
    )

    # 记录创建任务结束时间
    end_time = time.time()
    duration = end_time - start_time
    total_time += duration
    api_calls += 1

    json_resp = resp.json()
    task_uuid = json_resp["data"]["task_uuid"]
    print(json_resp)

    # 轮询任务状态（这部分时间不计入统计）
    while True:
        resp = requests.get(
            url=f"https://api.deepdataspace.com/v2/task_status/{task_uuid}",
            headers=headers
        )
        json_resp = resp.json()
        if json_resp["data"]["status"] not in ["waiting", "running"]:
            break

    if json_resp["data"]["status"] == "failed":
        print(json_resp)
    elif json_resp["data"]["status"] == "success":
        print(json_resp)

        item = {
            "img_name": img_name,
            "img_result": json_resp["data"]["result"]
        }

        all_results.append(item)


# 图片转base64
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


if __name__ == '__main__':
    img_folder = r"D:\10.0.79.35\DINO_model\13-水平洞口\test_dataset_2\images"
    for img_name in os.listdir(img_folder):
        file_path = os.path.join(img_folder, img_name)
        dino_API(img_name, file_path)

    with open(r"D:\10.0.79.35\DINO_model\13-水平洞口\grounding_1_6_pro.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # 计算并打印平均时间
    if api_calls > 0:
        avg_time = total_time / api_calls
        print(f"\nAPI调用统计（仅计算创建任务时间）:")
        print(f"总调用次数: {api_calls}")
        print(f"创建任务总耗时: {total_time:.4f}秒")
        print(f"平均每次创建任务耗时: {avg_time:.4f}秒")
    else:
        print("没有进行API调用")