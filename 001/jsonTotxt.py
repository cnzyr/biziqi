import json
import os

def transfrom_dict_to_str(target_dict):
    # 根据字典重新组织数据
    list_target = []
    list_data = target_dict['shapes']
    for item in list_data:
        dict02 = {}
        # 把label 转化为transcription
        dict02['transcription'] = item['label']
        list03 = item['points']
        list04 = []
        for j in list03:
            list04.append([j[0], j[1]])
        dict02['points'] = list04
        list_target.append(dict02)
    target_str = str(list_target)

    # 将字符中的"符号提前进行转义
    # target_str = target_str.replace("\"", "\\"")
    # 将字符中的'符号转为"
    target_str = target_str.replace("'", "\"")
    return target_str

if __name__ == "__main__":
    # json文件的文件夹
    src = r"D:\10.0.79.35\PaddleOCR\检测数据\钢筋排布图"
    src_foler = r"D:\10.0.79.35\PaddleOCR"
    src_1 = os.path.join(src_foler, "rebar_arrangement_drawing_table_data_val.txt")
    f = open(src_1, 'a', encoding='utf-8')
    for dirpath, dirnames, filenames in os.walk(src):
        for filename in filenames:
            print(filename)
            if filename.endswith(".json"):
                fd = open(os.path.join(src, filename), encoding='utf-8')
                # fd = open(os.path.join(src, filename))
                data = fd.read()
                dict01 = json.loads(data)
                target_str = transfrom_dict_to_str(dict01)
                image_path = dict01['imagePath']
                f.write('SimpleDataSet/'+image_path.split('\\')[-1]+'\t')
                f.write(target_str+'\n')
    f.close()
