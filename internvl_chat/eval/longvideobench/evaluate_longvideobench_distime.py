import argparse
import random
import itertools
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import os
import os.path as osp
from video_utils import read_frames_decord
from internvl.model import load_model_and_tokenizer, load_model_and_tokenizer_time
from internvl.train.dataset import build_transform, dynamic_preprocess
from dataclasses import dataclass, field
from PIL import Image
import json
from functools import partial
import logging
from pathlib import Path
import time
from tqdm import tqdm
import traceback
from icecream import ic
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, TIME_CONTEXT_TOKEN, START_TIME_TOKEN, END_TIME_TOKEN, FRAME_TIME_REF_TOKEN)
from itertools import chain, repeat

logger = logging.getLogger(__name__)

data_dir = 'data_example/longvideobench'

class LongVideoBenchDataset(Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                 insert_text=True,
                 insert_frame=True,
                ):
        super().__init__()
        self.data_path = data_path
        self.insert_text = insert_text

        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        
    def __getitem__(self, index):
        di = self.data[index]
        return {
            'video': os.path.join(self.data_path, "videos", di["video_path"]),
            'bound': [0 ,di["duration"]],
            'question': di["question"],
            'options': [chr(ord("A")+i)+'. '+ candidate for i, candidate in enumerate(di["candidates"])],
            'answer': chr(ord("A")+di.get("correct_choice", -1))+'. '+di["candidates"][di.get("correct_choice", 0)],
            "id": di["id"],
        }
    
    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]

@dataclass()
class StandardData:
    question: str = ''
    gt_answer: str = ''
    videos: list = field(default_factory=list)
    index: int = 0
    task_type: str = ''
    extra: dict = field(default_factory=dict) # for evaluation
    raw_model_answer = None
    model_answer = None

    def to_json(self,):
        return {
            'question': self.question,
            'gt_answer': self.gt_answer,
            'videos': self.videos,
            'index': self.index,
            'task_type': self.task_type,
            'extra': self.extra,
            'raw_model_answer': self.raw_model_answer,
            'model_answer': self.model_answer
        }


def check_answer(predict, gt_answer):
    flag = False

    predict = predict.lower()
    gt_answer = gt_answer.lower()

    pred_list = predict.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])

    gt_list = gt_answer.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    if pred_option.replace('.', '').strip() in gt_option:
        flag = True
    elif gt_content.strip() in pred_content:
        flag = True

    return flag

def empty_image():
    return Image.new('RGB', (800, 600), (255, 255, 255))

class ImageIO(object):
    def __init__(self):
        self.retry_num = 10

    def __call__(self, image_url, auto_retry=False, raise_error=False):
        for i in range(self.retry_num):
            try:
                if os.path.isfile(image_url):
                    image = Image.open(image_url).convert('RGB')
                return image
            except Exception as e:
                traceback.print_exc()
                if auto_retry:
                    pass
                else:
                    if raise_error:
                        raise RuntimeError(image_url)
                    ic()
                    return empty_image()

    def _load_video(self, video_url, num_frames=8):

        if isinstance(video_url, dict):
            if 'bound' in video_url:
                start_time = video_url['bound'][0]
                end_time = video_url['bound'][1]
            else:
                start_time = None
                end_time = None
            num_frames = video_url.get('num_frames', num_frames)
            video_url = video_url['video']
        else:
            start_time = None
            end_time = None
            video_url = str(video_url)

        video, frame_times, duration = read_frames_decord(video_url, num_frames=num_frames, sample='middle', start_time=start_time,
                                              end_time=end_time)

        to_pil = transforms.ToPILImage()
        frames = [to_pil(video[ti]) for ti in range(video.shape[0])]

        return frames, frame_times, duration

class LongVideoBenchTask():
    def __init__(self, num_frames=1, **kwargs) -> None:
        self.io = ImageIO()
        self.num_frames = num_frames
        self.prompt = 'Question: {question}\nOptions:\n{options}\nAnswer with the option’s letter from the given choices directly.'
        self.datas = LongVideoBenchDataset(data_dir, 'lvb_val.json')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        line = self.datas[index]
        if line['bound']:
            videos = [{'video': line['video'], 'bound': line['bound'], 'num_frames': self.num_frames}]
        else:
            videos = [{'video': line['video'], 'num_frames': self.num_frames}]
        question = line['question']

        # options = [f"{chr(ord('A')+i)}. {line[f'a{i}']}" for i in range(10) if f"a{i}" in line]
        gt_answer = '\n'.join(line['answer'])
        options = '\n'.join(line['options'])
        question = self.prompt.format(question=question, options=options)
        question = question.rstrip()
        task_type = self.datas[index]['id']

        return StandardData(question=question, gt_answer=gt_answer, videos=videos, index=index, task_type=task_type, extra=line)

    def postprocess(self, line: StandardData):
        output = {
            "raw": line.extra,
            "pred": line.raw_model_answer,
            "ground_truth": line.gt_answer,
        }
        return output

    def evaluate(self, merged_outputs):
        all_acc = []
        for line in merged_outputs:
            all_acc.append(check_answer(line['pred'], line['gt']))
        metrics = {}

        metrics['Accuracy_overall'] = np.mean(all_acc)
        print(metrics)
        return metrics, merged_outputs


class InferenceDataset(Dataset):
    def __init__(self, task, input_size=224, dynamic_image_size=False, use_thumbnail=False, max_num=6):
        self.task = task
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.task)

    def __getitem__(self, idx):
        task_data: StandardData = self.task[idx]
        question, gt_answer, videos, index, task_type = task_data.question, task_data.gt_answer, task_data.videos, task_data.index, task_data.task_type
      
        video_urls = videos

        batch_frames = [self.task.io._load_video(_)[0] for _ in video_urls]
        frame_times = [self.task.io._load_video(_)[1] for _ in video_urls]
        durations = [self.task.io._load_video(_)[2] for _ in video_urls]

        frame_times = list(chain.from_iterable(repeat(item, 2) for item in frame_times[0]))

        images = [_ for frames in batch_frames for _ in frames]

        special_tokens = '\n'.join(['<FRAME_TIME_REF>: <image>'.format(i + 1) for i in range(len(images))])
        question = special_tokens + '\n' + question

        num_patches_list = []
        pixel_values = []
        for image in images:
            if self.dynamic_image_size:
                patches = dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=self.use_thumbnail, max_num=self.max_num)
            else:
                patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)
        return{
            'question': question,
            'pixel_values': pixel_values,
            'answer': gt_answer,
            'num_patches_list': num_patches_list,
            'task_type': task_type,
            'durations': durations,
            'frame_times': frame_times
        }

class InferenceSamplerV2(Sampler):
    def __init__(self, size):
        if isinstance(size, float):
            logger.info(f"InferenceSampler(size=) expects an int but gets float, convert from {size} to {int(size)}.")
            size = int(size)
        elif not isinstance(size, int):
            raise TypeError(f"InferenceSampler(size=) expects an int. Got type {type(size)}.")
        self._size = size
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = [i for i in range(size) if i%self._world_size==self._rank]

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    num_patches_lists = [_['num_patches_list'] for _ in batches]
    task_types = [_['task_type'] for _ in batches]
    durations = [_['durations'] for _ in batches]
    frame_times = [_['frame_times'] for _ in batches]

    return pixel_values, questions, answers, num_patches_lists, task_types, durations, frame_times

def evaluate_chat_model():
    random.seed(args.seed)
    task = LongVideoBenchTask(num_frames=args.num_segments)
    vid_dataset = InferenceDataset(task=task, input_size=image_size, dynamic_image_size=args.dynamic,
                                   use_thumbnail=use_thumbnail, max_num=args.max_num)
    dataloader = torch.utils.data.DataLoader(
        dataset=vid_dataset,
        sampler=InferenceSamplerV2(len(vid_dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    debug = 0
    for _, (pixel_values, questions, answers, num_patches_lists, task_types, durations, frame_times) in enumerate(tqdm(dataloader)):
    
        model.time_context_token_id = tokenizer.convert_tokens_to_ids(TIME_CONTEXT_TOKEN)
        model.frame_time_context_token_id = tokenizer.convert_tokens_to_ids(FRAME_TIME_REF_TOKEN)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=1000,
            min_new_tokens=1,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
        )

        pred, time_dec = model.chat_distime(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_lists[0],
            question=questions[0],
            frame_times=frame_times[0],
            duration=durations[0],
            generation_config=generation_config,
            verbose=True
        )

        outputs.append({
            'question': questions[0],
            'pred': pred,
            'gt': answers[0],
            'task_type': task_types[0],
        })
    torch.distributed.barrier()


    world_size = dist.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if dist.get_rank() == 0:
        print(f'Evaluating LongVideoBench ...')
        metrics, _ = task.evaluate(merged_outputs)

        metrics_file = Path(args.out_dir, 'metrics.json')    
        results_file = Path(args.out_dir, 'result.json') 
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(merged_outputs, f, ensure_ascii=False, indent='\t')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent='\t')
        print('Results saved to {}'.format(results_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='longvideobench')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results/LongVideoBench/1B')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=64)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--num_segments', type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
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
    if hasattr(model.config, 'use_video_frames_compress'):
        if model.config.use_video_frames_compress == True:
            args.use_ffc = True
            args.num_video_query_token = model.config.num_video_query_token

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
