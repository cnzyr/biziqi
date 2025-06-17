import argparse

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--vlm_model_path', type=str, default="vlm_model/Qwen2.5-VL-32B-Instruct")
    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()
