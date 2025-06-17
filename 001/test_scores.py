import json
import os
from typing import List, Dict

from numpy.ma.extras import average

def calculate_key_accuracy(standard_entries: List[Dict], ocr_entries: List[Dict]) -> float:
    """计算所有字典的字段级（Key）平均精度"""
    total_accuracy = 0.0
    total_entries = 0

    for std_dict, ocr_dict in zip(standard_entries, ocr_entries):
        std_keys = set(std_dict.keys())
        ocr_keys = set(ocr_dict.keys())

        correct_keys = std_keys & ocr_keys
        total_keys = std_keys | ocr_keys

        if total_keys:
            total_accuracy += len(correct_keys) / len(total_keys)
            total_entries += 1

    return total_accuracy / max(total_entries, 1)  # 避免除零


def calculate_precision_recall(standard_entries: List[Dict], ocr_entries: List[Dict]) -> float:
    truth_ocrs = {(key, value) for d in standard_entries for key, value in d.items()}
    pred_ocrs = {(key, value) for d in ocr_entries for key, value in d.items()}
    # print("初步处理结果:","\n",truth_ocrs,"\n",pred_ocrs)
    truth_ocr = {(k, v) for k, v in truth_ocrs if v != ""}
    pred_ocr = {(k, v) for k, v in pred_ocrs if v != ""}
    # print("去除空字段:",truth_ocr, pred_ocr)
    if len(truth_ocr) == 0 and len(pred_ocr) == 0:
        precision = 1.000
        recall = 1.000


    else:
        tp = len(truth_ocr & pred_ocr)
        fp = len(pred_ocr - truth_ocr)
        fn = len(truth_ocr - pred_ocr)
        print("tp:",tp, "fp:",fp, "fn:",fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0


    return precision, recall

def evaluate_ocr_accuracy(standard_json_str: str, ocr_json_str: str) -> Dict[str, float]:
    try:
        standard_data = json.loads(standard_json_str)
        ocr_data = json.loads(ocr_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")

    # 打印结果
    print(f"标准JSON: {standard_data}")
    print(f"OCR结果: {ocr_data}")
    # print("=== 处理过程 ===")
    # 计算指标
    key_accuracy = calculate_key_accuracy(standard_data, ocr_data)

    # 精度 召回率
    precision, recall= calculate_precision_recall(standard_data, ocr_data)

    return key_accuracy, precision, recall


# 示例测试
if __name__ == "__main__":
    test_img_output = r"D:\10.0.79.35\model_test\next_2\test_img_output"
    key_accuracys = []
    precisions = []
    recalls = []
    total_key = 0
    warn_key = 0
    for filename in os.listdir(test_img_output):
        with open(os.path.join(test_img_output, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(filename)

        if data == "" or data == "[]":
            data = "[{\"key\": \"value\"}]"
        ocr_json_str = data.replace("```json","").replace("```","").replace("\n","")

        with open(r"D:\10.0.79.35\model_test\001_data_json_test.json", 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        for jsondata in data_json:
            if os.path.splitext(jsondata["image"])[0] == os.path.splitext(filename)[0]:
                standard_json_str = jsondata["conversations"][1]["value"]

                key_accuracy, precision, recall= evaluate_ocr_accuracy(standard_json_str, ocr_json_str)
                # print("=== 评估结果 ===")
                print(f"1. 字段级精度（Key Accuracy）: {key_accuracy:.2%}")
                print(f"2. 精度: {precision:.3f}; 召回率: {recall:.3f}\n")
                key_accuracys.append(key_accuracy)
                precisions.append(precision)
                recalls.append(recall)


    # print(f"总体字段级精度:{average(key_accuracys):.2f}")
    print(f"精度: {average(precisions):.3f}; 召回率: {average(recalls):.3f}")

