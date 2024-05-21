#conda activate torch2
import argparse
import torch

from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.conversation import conv_templates, SeparatorStyle
from q_align.model.builder import load_pretrained_model
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm
from collections import defaultdict

import os




def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)





from PIL import Image
import numpy as np

using_local_model = False
#using local mode
if using_local_model == True:
    disable_torch_init()
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_name = get_model_name_from_path("q-future/one-align")
    tokenizer, model, image_processor, context_len = load_pretrained_model("/Repo/zyj/Project/Q/Q-Align/q-future/one-align", None, model_name, load_8bit=True, load_4bit=True, device="cuda:0")
else:
    #this code will automatically download the one-align weight from the huggingface
    disable_torch_init()
    model_name = get_model_name_from_path("q-future/one-align")
    tokenizer, model, image_processor, context_len = load_pretrained_model("q-future/one-align", None, model_name, load_8bit=True, load_4bit=True, device="cuda:0")

    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def get_image_tensor(image):
    image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to("cuda:0")
    return image_tensor


def load_img(img_path):
    image = Image.open(img_path).convert('RGB')
    width, height = image.size
    # 上半部分
    upper_patch = get_image_tensor(image.crop((0, 0, width, height // 2)))
    # 下半部分
    lower_patch = get_image_tensor(image.crop((0, height // 2, width, height)))
    return [upper_patch, lower_patch]




def get_hidden_states(img_path, save_path):
    img_tensors = load_img(img_path)


    conv_mode = "mplug_owl2"   
    inp = "How would you rate the quality of this image?"   
    conv = conv_templates[conv_mode].copy()
    inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
    conv.append_message(conv.roles[0], inp)
    image = None
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The quality of the image is"
    toks = ["good", "poor", "high", "fair", "low", "excellent", "bad", "fine", "moderate",  "decent", "average", "medium", "acceptable"]
    #print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    #print(ids_)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda:0")

    with torch.inference_mode():
        hidden_states = model(input_ids.repeat(len(img_tensors), 1),
            images=torch.cat(img_tensors, 0))['logits'] 
    average_features = torch.mean(hidden_states, dim=0)
    average_features = torch.mean(average_features, dim=0,keepdim=True)
    average_features_cpu = average_features.cpu()

    # 将PyTorch张量转换为NumPy数组
    average_features_numpy = average_features_cpu.numpy()
    print(average_features_numpy.shape)
    # 存储NumPy数组到文件，这里使用.npy格式
    np.save(save_path, average_features_numpy)
    return average_features_numpy



def process_videos(input_path, output_prefix):
    videos = os.listdir(input_path)
    for video in videos:
        video_path = os.path.join(input_path, video)
        imgs = os.listdir(video_path)
        for img in imgs:
            img_path = os.path.join(video_path, img)
            directory = os.path.join(output_prefix, video)
            if not os.path.exists(directory):
                os.mkdir(directory)
            save_path = os.path.join(directory, img.split('.')[0] + '.npy')
            print(img_path)
            print(save_path)
            get_hidden_states(img_path, save_path)

def main():
    parser = argparse.ArgumentParser(description="Process videos and extract features.")
    parser.add_argument('--input_path', type=str, help='Path to the directory containing the video frames', default='/data/sunwei_data/ntire_video/test_image_original/test/')
    parser.add_argument('--output_prefix', type=str, help='Prefix for the output npy files path', default='/home/sunwei/code/VQA/RQ-VQA/qalign_feature/')
    
    args = parser.parse_args()

    process_videos(args.input_path, args.output_prefix)

if __name__ == '__main__':
    main()
