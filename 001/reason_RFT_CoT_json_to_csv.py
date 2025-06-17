import csv
import json
import os
from math import nan


def json_to_csv(json_file):
    file_path = json_file
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # 更安全的文件名提取方式


    csv_path = os.path.join(r"D:\10.0.79.35\model\reason_RFT_Cot_datasets\test", f"{file_name}.csv")

    # 定义CSV字段顺序
    # train
    # fieldnames = ["id", "image", "ood_image", "problem", "cot", "answer"]
    # test
    fieldnames = ["id", "image", "problem", "answer"]

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        print(f"开始处理文件: {file_name}.json")

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)  # 写入标题行

            for item in data:
                # train
                # row = [
                #     item.get("id", nan),
                #     item.get("image", nan),
                #     item.get("ood_image", nan),
                #     item.get("problem", nan),
                #     item.get("cot", nan),
                #     item.get("answer", nan)
                # ]
                # test
                row = [
                    item.get("id", nan),
                    item.get("image", nan),
                    item.get("problem", nan),
                    item.get("answer", nan)
                ]
                writer.writerow(row)

        print(f"成功转换并保存为: {file_name}.csv")

    except Exception as e:
        print(f"处理文件 {file_name}.json 时出错: {str(e)}")


# 处理文件夹中的所有JSON文件
file_folder = r"D:\10.0.79.35\datasets\Reason-RFT-CoT-Dataset\test_jsons"
for file in os.listdir(file_folder):
    if file.endswith(".json"):
        file_path = os.path.join(file_folder, file)
        json_to_csv(file_path)

print("所有文件处理完成!")