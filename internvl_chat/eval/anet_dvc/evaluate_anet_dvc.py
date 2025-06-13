import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from internvl.model import load_model_and_tokenizer, load_model_and_tokenizer_time
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, TIME_CONTEXT_TOKEN, FRAME_TIME_REF_TOKEN)
import torch.distributed as dist
from itertools import chain, repeat
import re

def collate_fn(batches):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    video_id = [_['video_id'] for _ in batches]
    num_patches_lists = [_['num_patches_list'] for _ in batches]
    duration = [_['duration'] for _ in batches]
    frame_times = [_['frame_times'] for _ in batches]
    
    return pixel_values, questions, video_id, num_patches_lists, duration, frame_times

class ANetDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, video_file, num_segments=16, input_size=224,
                 dynamic_image_size=False, use_thumbnail=False, max_num=6):
        self.decord_method = self.read_video
        self.input_size = input_size
        self.num_segments = num_segments
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

        self.video_dict = {}
        with open(video_file, 'r') as f_v:
            for line in f_v:
                path = line.strip()
                video_id = path.split('/')[-1].split('.')[0]
                self.video_dict[video_id] = path

        self.json_dict = []
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                item_dict = dict(
                    video_id=item,
                    video=self.video_dict[item],
                )
                self.json_dict.append(item_dict)
        

    def __len__(self):
        return len(self.json_dict)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        vlen = len(vr)
        duration = round(vlen / fps, 2)

        images_group = list()
        frame_times = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
            frame_time = round(frame_index / fps, 2)
            frame_times.append(frame_time)

        return images_group, frame_times, duration


    def __getitem__(self, idx):
        video_id = self.json_dict[idx]['video_id']
        video_path = self.json_dict[idx]['video']
     
        image_list, frame_times, duration =  self.decord_method(video_path)
        frame_times = list(chain.from_iterable(repeat(item, 2) for item in frame_times))

        video_prefix = '\n'.join(['<FRAME_TIME_REF>: <image>' for i in range(len(image_list))])        
        question = video_prefix + "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences."

        raw_images = []
        num_patches_list = []
        pixel_values = []
        for image in image_list:
            raw_images.append(image)
            if self.dynamic_image_size:
                patches = dynamic_preprocess(image, image_size=self.input_size,
                                             use_thumbnail=self.use_thumbnail,
                                             max_num=self.max_num)
            else:
                patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)
        return {
            'question': question,
            'pixel_values': pixel_values,
            'video_id':video_id,
            'num_patches_list': num_patches_list,
            'duration': duration,
            'frame_times': frame_times
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def postprocess_pred_time(time, duration, reg_max):
    time_stamps = []
    for interval in time:
        for start, end in interval:
            start_time_sec = (float(start) / reg_max * duration)
            end_time_sec = (float(end) / reg_max * duration)

            time_stamp_str = [start_time_sec, end_time_sec]
            time_stamps.append(time_stamp_str) 

    return time_stamps


def evaluate_chat_model():

    random.seed(args.seed)
    json_file = 'data_example/anet/val_2.json'
    video_file = 'data_example/anet/video_list.txt'

    vid_dataset = ANetDataset(
        json_file=json_file, 
        video_file=video_file,
        num_segments=args.num_segments,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num)
    dataloader = torch.utils.data.DataLoader(
        dataset=vid_dataset,
        sampler=InferenceSampler(len(vid_dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn),
    )

    result = {}

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for _, (pixel_values, questions, video_id, num_patches_lists, duration, frame_times) in tqdm(enumerate(dataloader), total=len(dataloader)):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=1000,
            min_new_tokens=1,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
        )
        model.time_context_token_id = tokenizer.convert_tokens_to_ids(TIME_CONTEXT_TOKEN)
        model.frame_time_context_token_id = tokenizer.convert_tokens_to_ids(FRAME_TIME_REF_TOKEN)

        pred, pred_time = model.chat_distime(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_lists[0],
            question=questions[0],
            generation_config=generation_config,
            frame_times=frame_times[0],
            duration=duration[0],
            verbose=True
        )   
        events = [s.strip() for s in re.split(r'<TIME_STAMP>\s*,\s*', pred)[1:] if s.strip()]

        if pred_time is not None:
            pred_time = postprocess_pred_time(pred_time, duration[0], reg_max=model.reg_max)

        answer_list = []


        for event, timestamp in zip(events, pred_time):
            answer_list.append({
                "event": event,
                "timestamps": f"{int(round(timestamp[0]))} - {int(round(timestamp[1]))}"
            })

        answer_str = json.dumps(answer_list)  
        escaped_answer_str = answer_str.replace('"', '\"') 
        result = dict(
            video_id=video_id[0],
            task='captioning',
            answer=escaped_answer_str 
        )

        log_path_template = os.path.join(args.out_dir, "results-{}.txt")
        log_path = log_path_template.format(rank)

        with open(log_path, 'a') as f:
            f.write(json.dumps(result) + '\n')

    if is_dist_avail_and_initialized():
        dist.barrier()
                
    if rank == 0:
        uni_videoid = set()
        merged_log_path = os.path.join(args.out_dir, "results.txt")
        part_log_paths = [log_path_template.format(i) for i in range(world_size)]
        with open(merged_log_path, 'w') as outfile:
            for part_log_path in part_log_paths:
                if not os.path.exists(part_log_path):
                    print(f"Warning: {part_log_path} not found, skipping.")
                    continue
                with open(part_log_path, 'r') as infile:
                    for line in infile:
                        try:
                            video_id = json.loads(line)['video_id']
                        except Exception as e:
                            print(f"Error parsing line: {line}, error: {e}")
                            continue
                        if video_id not in uni_videoid:
                            uni_videoid.add(video_id)
                            outfile.write(line)

        for part_log_path in part_log_paths:
            if os.path.exists(part_log_path):
                os.remove(part_log_path)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='anet_dvc')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results/ANet-Caption-DVC/1B')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=1)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--num_segments', type=int, default=16)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer_time(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
