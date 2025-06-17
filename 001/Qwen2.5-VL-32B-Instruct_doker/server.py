import base64
import json
import logging
import os
import random
import sys
import urllib
import cv2
import numpy as np
from flask import Flask, request, Response

from mllm_infer_lmdeploy import Mllm_Response
from utils import parse_args

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, './')))

random.seed(42)
args = parse_args()
mllm_response = Mllm_Response(args)


def download_img_cv(img_url):
    try:
        req = urllib.request.Request(
            img_url,
            data=None,
            headers={

            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image_as_cvimage = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if image_as_cvimage is None:
                raise ValueError("Failed to decode image")
            return image_as_cvimage
    except Exception as e:
        raise ValueError(f"Failed to download or decode image from {img_url}: {str(e)}")

def img_to_base64(img_array):
    encode_image = cv2.imencode(".png", img_array)[1]
    byte_data = encode_image.tobytes()
    base64_str = base64.b64encode(byte_data).decode("ascii")
    return base64_str


app = Flask(__name__)


@app.route('/vlm_model_qwen/qwen2_5', methods=['POST'])
def vlm_model_qwen():
    request_json = 'failed to get request json'
    try:
        request_json = request.get_json(force=True)
        if "images" in request_json:
            images_base64 = request_json["images"]
            if images_base64 == []:
                if "urls" in request_json:
                    urls = request_json["urls"]
                    if urls != []:
                        images_base64 = [img_to_base64(download_img_cv(url)) for url in urls]
                    else:
                        return Response(status=400,
                                        response=json.dumps(
                                            {"code": "400", "message": "No base64 and url", "data": ""}),
                                        content_type="application/json")
        elif 'urls' in request_json:
            urls = request_json["urls"]
            if urls != []:
                images_base64 = [img_to_base64(download_img_cv(url)) for url in urls]
            else:
                return Response(status=400,
                                response=json.dumps(
                                    {"code": "400", "message": "No base64 and url", "data": ""}),
                                content_type="application/json")
        else:
            return Response(status=400,
                            response=json.dumps(
                                {"code": "400", "message": "No base64 and url", "data": ""}),
                            content_type="application/json")

        data = []
        text = request_json["text"]
        try:
            result = mllm_response.chat_and_pe_response_pro(text ,images_base64)
            data.append(result)
        except Exception as e:
            data.append({"error": f"Failed to process image: {str(e)}"})
        res_dict = {
            'code': '200',
            'message': 'success',
            'data': data
        }
        return Response(status=200, response=json.dumps(res_dict), content_type='application/json')

    except Exception as e:
        logger = logging.getLogger('exception_logger')
        logger.setLevel(logging.ERROR)
        log_file = './log/exception.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.exception(e)
        logger.exception(f'request body: {request_json}')
        return Response(status=500, response=json.dumps({'code': '500', 'message': 'internal error', 'data': ''}),
                        content_type='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=False, processes=1)
