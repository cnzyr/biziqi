import base64
import json
import os
import time
import requests


all_results = []
# API请求
def dino_API(img_name, img_path):
    # 请求头
    global item
    headers = {
        "Content-Type": "application/json",
        "Token" :
    }

    # 请求参数
    json_data = {
        "model": "DINO-X-1.0",
        "image": f"data:image/png;base64,{img_to_base64(img_path)}",
        "prompt": {
            "type":"text",
            "text":"head.helmet"
        },
        "targets": ["bbox"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8
    }

    # 创建任务接口
    resp = requests.post(
        url = "https://api.deepdataspace.com/v2/task/dinox/detection",
        headers=headers,
        json = json_data
    )

    json_resp = resp.json()
    task_uuid = json_resp["data"]["task_uuid"]
    print(json_resp)

    # 轮询任务状态
    while True:
        resp = requests.get(
            url = f"https://api.deepdataspace.com/v2/task_status/{task_uuid}",
            headers=headers
        )
        json_resp = resp.json()
        if json_resp["data"]["status"] not in ["waiting", "running"]:
            break
        time.sleep(1)

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
    img_folder = r"D:\10.0.79.35\DINO_model\安全帽\img"
    for img_name in os.listdir(img_folder):
        file_path = os.path.join(img_folder, img_name)
        dino_API(img_name, file_path)

    with open(r"D:\10.0.79.35\DINO_model\DINO-X_helmet.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)