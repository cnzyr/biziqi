import os

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    text_detection_model_name="PP-OCRv5_server_det",
    device="gpu:0",
)
# ocr = PaddleOCR(lang="en") # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数来使用 PP-OCR 其他版本
# ocr = PaddleOCR(device="gpu") # 通过 device 参数使得在模型推理时使用 GPU
img_folder = r"/data/bizq/biziqi_002/test_model/paddleOCR/data"
for img_name in os.listdir(img_folder):
    print("开始识别:",img_name)
    img_path = os.path.join(img_folder, img_name)
    result = ocr.predict(img_path)
    for res in result:
        res.print()
        res.save_to_img("output_v5")
        res.save_to_json("output_v5")