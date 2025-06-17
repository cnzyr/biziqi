import json
import numpy as np

# 文件路径
pred_file = r"D:\10.0.79.35\DINO_model\13-水平洞口\grounding_1_6_pro.json"
real_file = r"D:\10.0.79.35\DINO_model\13-水平洞口\test_dataset_2\_annotations.coco.json"

# 读取 JSON 文件
with open(pred_file, 'r') as f:
    pred_data = json.load(f)
with open(real_file, 'r') as f:
    real_data = json.load(f)


# 转换 COCO 真实框格式
def convert_coco_box(box):
    x_min, y_min, width, height = box
    return [x_min, y_min, x_min + width, y_min + height]


# 计算 IoU
def calculate_iou(pred_box, gt_box):
    x1_p, y1_p, x2_p, y2_p = pred_box[:4]
    x1_g, y1_g, x2_g, y2_g = gt_box

    x1_i = max(x1_p, x1_g)
    y1_i = max(y1_p, y1_g)
    x2_i = min(x2_p, x2_g)
    y2_i = min(y2_p, y2_g)

    if x2_i > x1_i and y2_i > y1_i:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    else:
        intersection = 0

    area_p = (x2_p - x1_p) * (y2_p - y1_p)
    area_g = (x2_g - x1_g) * (y2_g - y1_g)
    union = area_p + area_g - intersection

    return intersection / union if union > 0 else 0


# 计算 AP
def calculate_ap(predictions, ground_truths, iou_threshold):
    sorted_preds = sorted(predictions, key=lambda x: x[4], reverse=True)  # 按置信度排序
    total_gt = len(ground_truths)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    matched_gt = set()  # 记录已匹配的真实框

    for pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truths):
            if i not in matched_gt:
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    # 11 点插值法
    recall_levels = np.linspace(0, 1, 11)
    interpolated_precisions = []
    for r in recall_levels:
        precisions_at_r = [p for p, rec in zip(precisions, recalls) if rec >= r]
        interpolated_precisions.append(max(precisions_at_r) if precisions_at_r else 0)

    ap = np.mean(interpolated_precisions)
    return ap


# 处理数据
image_to_preds = {}
image_to_gts = {}

# 提取真实框
image_id_to_name = {img["id"]: img["file_name"] for img in real_data["images"]}
for ann in real_data["annotations"]:
    img_id = ann["image_id"]
    img_name = image_id_to_name.get(img_id)
    if img_name:
        if img_name not in image_to_gts:
            image_to_gts[img_name] = []
        image_to_gts[img_name].append(convert_coco_box(ann["bbox"]))

# 提取预测框
for pred_item in pred_data:
    img_name = pred_item["img_name"]
    image_to_preds[img_name] = [box["bbox"] + [box["score"]] for box in pred_item["img_result"]["objects"]]

# 计算 mAP
ap_05_list = []
map_05_095_list = []

for img_name in image_to_preds:
    preds = image_to_preds.get(img_name, [])
    gts = image_to_gts.get(img_name, [])

    if gts:  # 仅处理有真实框的图像
        # 计算 mAP@0.5
        ap_05 = calculate_ap(preds, gts, iou_threshold=0.5)
        ap_05_list.append(ap_05)

        # 计算 mAP@0.5:0.95
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        aps = [calculate_ap(preds, gts, iou) for iou in iou_thresholds]
        map_05_095 = np.mean(aps)
        map_05_095_list.append(map_05_095)

        print(f"Image: {img_name}")
        print(f"Predictions: {preds}")
        print(f"Ground Truths: {gts}")
        print(f"AP@0.5: {ap_05:.4f}")
        print(f"AP@0.5~0.95: {map_05_095:.4f}\n")

# 计算平均 mAP
mean_ap_05 = np.mean(ap_05_list) if ap_05_list else 0
mean_map_05_095 = np.mean(map_05_095_list) if map_05_095_list else 0

print(f"mAP@0.5: {mean_ap_05:.4f}")
print(f"mAP@0.5~0.95: {mean_map_05_095:.4f}")