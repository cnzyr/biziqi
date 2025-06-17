import json
import os
import random
import re
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56,
                 max_pixels: int = 16384 * 28 * 28):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def convert_bbox_to_resized(bbox, orig_height, orig_width, new_height, new_width):
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    # 确保坐标不越界
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))

    return [x1_new, y1_new, x2_new, y2_new]


def read_image_with_cv2(path):
    with open(path, 'rb') as f:
        img_bytes = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)


def draw_boxes(image_path, output_path, data):
    model_output = data["conversations"][1]["value"]
    image_name = os.path.basename(data["images"])

    # 提取坐标
    no_text_box = re.findall(r"{\"bbox_2d\": \[(\d+), (\d+), (\d+), (\d+)], \"label\": \"([^\"]+)\"}", model_output)
    text_boxes = re.findall(r"{\"bbox_2d\": \[(\d+), (\d+), (\d+), (\d+)], \"text_content\": \"([^\"]+)\"}",model_output)

    cv2_img = read_image_with_cv2(image_path)
    orig_height, orig_width = cv2_img.shape[:2]

    new_width, new_height = smart_resize(orig_width, orig_height)

    boxes = {
        "box": [convert_bbox_to_resized([int(x) for x in box[:4]], orig_height, orig_width, new_height, new_width)
                for box in no_text_box],
        "texts": [{
            "bbox": convert_bbox_to_resized([int(x) for x in box[:4]], orig_height, orig_width, new_height, new_width),
            "text": box[4]
        } for box in text_boxes]
    }

    # 加载图像并调整尺寸
    image = Image.open(image_path).convert("RGB")
    image = image.resize((new_width, new_height))
    image_width, image_height = image.size
    print(image_width, image_height)
    draw = ImageDraw.Draw(image)
    text_color = ["blue"]
    colors = ["red", "darkgreen", "mediumvioletred", "indigo" , "navy", "crimson"]

    try:
        font_size = max(10, min(new_width, new_height) // 20)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    i = 0
    # 绘制结构框
    for box in boxes["box"]:
        box_color = colors[i]
        x1, y1, x2, y2 = box
        # 裁剪坐标到图片范围内
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))
        if y2 - y1 < 10:
            draw.rectangle((x1, y1, x2, y2), outline=box_color, width=8)

        draw.rectangle((x1, y1, x2, y2), outline=box_color, width=2)
        i+=1

    # 绘制文字框
    for text_box in boxes["texts"]:
        box_color = text_color[0]
        x1, y1, x2, y2 = text_box["bbox"]
        draw.rectangle((x1, y1, x2, y2), outline=box_color, width=2)
        draw.text((x1, y1 - 20), text_box["text"], fill=box_color, font=font)

    # 保存结果（确保输出目录存在）
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, image_name)
    image.save(output_file)
    print(f"Processed: {image_name} -> {output_file}")

def main():
    json_file = r"D:\10.0.79.35\t_bridge_cot-sft\t-bridge-cot-sft-0519_005.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        for item in data:
            images_folder = r"D:\10.0.79.35\t_bridge_cot-sft\t-bridge-cot-images-0519"
            image_name = os.path.basename(item["images"])
            image_path = os.path.join(images_folder, image_name)
            output_path = r"D:\10.0.79.35\t_bridge_cot-sft\output_img"
            for img_name in os.listdir(images_folder):
                if img_name == image_name:
                    draw_boxes(image_path, output_path, item)


if __name__ == "__main__":
    main()