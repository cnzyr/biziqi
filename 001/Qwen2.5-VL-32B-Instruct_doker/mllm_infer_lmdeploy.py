import cv2
import base64
import numpy as np
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
import json
import time
import re
import os
import torch
from PIL import Image
# from transformers import AutoProcessor, AutoModelForVision2Seq, GenerationConfig



class Mllm_Response:
    def __init__(self, args):
        torch.cuda.empty_cache()
        self.gen_config = GenerationConfig(temperature=0)
        self.pipe = pipeline(args.vlm_model_path,
                             backend_config=TurbomindEngineConfig(
                                 session_len=8192, tp=1, cache_max_entry_count=0.1
                             ))

        self.img_size = 1280

    # 将VLM的文本输出解析为JSON格式，处理异常情况
    def prediction2json(self, model_result):
        if model_result == "" or model_result == "[]" or model_result == [] or model_result == None:
            model_result = "[{\"设计压力\": \"\", \"口径\": \"\", \"防腐类型\": \"\", \"隔热类型\": \"\", \"保温厚度\": \"\"}]"
        st = model_result.find('[')
        ed = model_result.rfind(']')
        model_result = model_result[st:ed + 1]

        json_data = json.loads(model_result)
        # 格式化输出 JSON
        formatted_json = json.dumps(json_data, ensure_ascii=False, indent=2)
        return formatted_json

    # 生成标准化的Prompt结构（用户或助手角色）
    def response_prompt_stencil(self, question, base64_encoded_data):
        prompt = []
        result = re.split('<image>', question)
        i = 0
        for data in result:
            if data == "<image>":
                prompt.append({'type': 'image_url','image_url': {'url': (base64_encoded_data[i])}})
                i+=1
            else:
                prompt.append({'type': 'text', 'text': data})
        return prompt

    # 执行VLM推理，获取模型输出
    def chat_and_pe_response(self,text, test_img_base64):
        image_tag_count = text.count("<image>")
        image_count = len(test_img_base64) if test_img_base64 else 0
        if image_tag_count < image_count:
            text += "<image>" * (image_count - image_tag_count)
        model_input = self.response_prompt_stencil(text, test_img_base64)
        start = time.time()
        result = self.pipe(model_input, gen_config=self.gen_config).text
        print(f'mllm time: {time.time() - start}')
        return result

    # 高级接口，处理图像、调用VLM推理并格式化输出为JSON
    # TODO:api输出
    def chat_and_pe_response_pro(self, text,test_img_base64):
        test_img_base64 = self.process_image(test_img_base64)
        model_result = self.chat_and_pe_response(text ,test_img_base64)
        result = self.prediction2json(model_result)
        # print(result)
        return result

    # 处理输入图像（解码、缩放、编码）
    def process_image(self, img_base64):
        img_data = []
        for img in img_base64:
            img_cv = base64_to_cv2(img)
            if img_cv is not None:
                img_cv = resize_image_by_long_side(img_cv, self.img_size)
                img_base64 = img_to_base64(img_cv)
            else:
                img_base64 = None
            img_data.append(img_base64)
            return img_data


# 将OpenCV图像编码为Base64字符串
def img_to_base64(img_array):
    encode_image = cv2.imencode(".png", img_array)[1]
    byte_data = encode_image.tobytes()
    base64_str = base64.b64encode(byte_data).decode("ascii")
    return base64_str

# 将Base64编码的图像字符串转换为OpenCV图像
def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

# 按长边等比例缩放图像，确保宽高为 max_stride（128）的倍数
def resize_image_by_long_side(image, size=1024):
    h, w, _ = image.shape
    resize_w = w
    resize_h = h
    # Fix the longer side
    if resize_h > resize_w:
        ratio = float(size) / resize_h
    else:
        ratio = float(size) / resize_w
    ratio = min(1, ratio)
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    max_stride = 128
    resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
    resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
    image = cv2.resize(image, (int(resize_w), int(resize_h)))

    return image