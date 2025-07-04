import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
# from internvl.model import load_model_and_tokenizer, load_model_and_tokenizer_time
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel, InternVLChatModelDisTime
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, TIME_CONTEXT_TOKEN, FRAME_TIME_REF_TOKEN)
import os

from itertools import chain, repeat

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = 'DisTime/DisTime-InternVL2_5-1B'
model = InternVLChatModelDisTime.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1000, do_sample=False, num_beams=1)


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vlen = len(vr)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    duration = round(vlen / fps, 2)

    pixel_values_list, num_patches_list, frame_times = [], [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        frame_times.append(round(frame_index / fps, 2))
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, duration, frame_times

def postprocess_pred_time(time, duration, reg_max):

    time_stamps = []
    for interval in time:
        for start, end in interval:
            start_time_sec = (float(start) / reg_max * duration)
            end_time_sec = (float(end) / reg_max * duration)

            time_stamp_str = [start_time_sec, end_time_sec]
            time_stamps.append(time_stamp_str) 

    return time_stamps

video_path = 'data/Charades/Charades_v1_480/GL2JW.mp4'

pixel_values, num_patches_list, duration, frame_times = load_video(video_path, num_segments=16, max_num=1)
frame_times = list(chain.from_iterable(repeat(item, 2) for item in frame_times))
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = '\n'.join(['<FRAME_TIME_REF>: <image>' for i in range(len(num_patches_list))])

query = 'person closes a cupboard door.'

question = video_prefix + "Give you a textual query: '{}'. When does the described content occur in the video? Please return the timestamp in seconds.".format(query)

model.time_context_token_id = tokenizer.convert_tokens_to_ids(TIME_CONTEXT_TOKEN)
model.frame_time_context_token_id = tokenizer.convert_tokens_to_ids(FRAME_TIME_REF_TOKEN)

response, hstory, pred_time = model.chat_distime(tokenizer, pixel_values, num_patches_list=num_patches_list, question=question, id=None,
                                                 generation_config=generation_config, frame_times=frame_times,
                                duration=duration,
                                history=None, return_history=True)

if pred_time is not None:
    pred_time = postprocess_pred_time(pred_time, duration, reg_max=model.reg_max)

    for time_stamp in pred_time:
        time_str = str(time_stamp[0]) + "s - " + str(time_stamp[1]) + "s"
        response = response.replace('<TIME_STAMP> ', time_str, 1)

print('response:', response)
print(f'User: {question}\nAssistant: {response}')

