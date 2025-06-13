import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import re
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
import logging
from itertools import chain, repeat

def collate_fn_stage1(batches):
    question = [_['question'] for _ in batches]
    query = [_['query'] for _ in batches]
    qid = [_['qid'] for _ in batches]
    answer = [_['answer'] for _ in batches]
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    video_id = [_['video_id'] for _ in batches]
    num_patches_lists = [_['num_patches_list'] for _ in batches]
    duration = [_['duration'] for _ in batches]
    frame_times = [_['frame_times'] for _ in batches]
    return question, query, qid, answer, pixel_values, video_id, num_patches_lists, duration, frame_times

def collate_fn_stage2(batches):
    question = [_['question'] for _ in batches]
    query = [_['query'] for _ in batches]
    qid = [_['qid'] for _ in batches]
    timestamps_res = [_['timestamps_res'] for _ in batches]
    answer = [_['answer'] for _ in batches]
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    video_id = [_['video_id'] for _ in batches]
    num_patches_lists = [_['num_patches_list'] for _ in batches]
    duration = [_['duration'] for _ in batches]
    frame_times = [_['frame_times'] for _ in batches]
    return question, query, qid, timestamps_res, answer, pixel_values, video_id, num_patches_lists, duration, frame_times

DATA_PATHS = {
    "test_json": "data_example/nextgqa/test.json",
    "video_root": "data/NExT-QA/NExTVideo",
    "gt_json": "data_example/nextgqa/gsub_test.json"
}

class BaseNExtGQADataset(torch.utils.data.Dataset):
    def __init__(self, json_file, video_root, num_segments=16, input_size=224,
                 dynamic_image_size=False, use_thumbnail=False, max_num=6):
        self.decord_method = self.read_video
        self.input_size = input_size
        self.num_segments = num_segments
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.json_dict = []
        self.QA_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
        self.QA_question_prompt = 'Only give the best option.'        
        self.MR_prompt = "Give you the textual query: '{}'. When does the described content occur in the video? Please return the timestamp in seconds."
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
    def qa_template(self, query, candidates):
        question = f"Question1: {query}\n"
        question += 'Options:\n'
        for idx, c in enumerate(candidates):
            question += f"({chr(ord('0') + idx)}) {c}\n"
        question = question.rstrip()
        return question
    def __getitem__(self, idx):
        video_id = self.json_dict[idx]['video_id']
        video_path = self.json_dict[idx]['video_path']
        query = self.json_dict[idx]['query']
        qid = self.json_dict[idx]['qid']
        answer = self.json_dict[idx]['answer']
        candidates = self.json_dict[idx]['candidates']
        image_list, frame_times, duration =  self.decord_method(video_path)
        frame_times = list(chain.from_iterable(repeat(item, 2) for item in frame_times))
        QA_question = self.qa_template(self.json_dict[idx]['query'], self.json_dict[idx]['candidates'])
        video_prefix = '\n'.join(['<FRAME_TIME_REF>: <image>' for i in range(len(image_list))])
        QA_question = video_prefix + '\n' + self.QA_prompt + '\n' + QA_question + '\n' + self.QA_question_prompt
        MR_question =  video_prefix + '\n' + self.MR_prompt.format(query)
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
            'question': [MR_question, QA_question],
            'query': query,
            'qid': qid,
            'answer': answer,
            'pixel_values': pixel_values,
            'video_id':video_id,
            'num_patches_list': num_patches_list,
            'duration': duration,
            'frame_times': frame_times
        }

class NExtGQADatasetStage1(BaseNExtGQADataset):
    def __init__(self, json_file, video_root, num_segments=16, input_size=224,
                 dynamic_image_size=False, use_thumbnail=False, max_num=6):
        super().__init__(json_file, video_root, num_segments, input_size, dynamic_image_size, use_thumbnail, max_num)
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                video_id = item['video']
                video_name = video_id + '.mp4'
                video_path = os.path.join(video_root, video_name)
                query = item['question']
                qid = item['qid']
                answer = item['answer']
                candidates = []
                for idx in range(item['num_option']):
                    candidates.append(item[f'a{idx}'])
                item_dict = dict(
                    video_id=video_id,
                    video_path=video_path,
                    query=query,
                    qid=qid,
                    answer=answer,
                    candidates=candidates
                )
                self.json_dict.append(item_dict)

class NExtGQADatasetStage2(BaseNExtGQADataset):
    def __init__(self, json_file, video_root, num_segments=16, input_size=224,
                 dynamic_image_size=False, use_thumbnail=False, max_num=6, timestamps_json=None):
        super().__init__(json_file, video_root, num_segments, input_size, dynamic_image_size, use_thumbnail, max_num)
        if not timestamps_json or not os.path.exists(timestamps_json):
            raise ValueError('You must provide --timestamps-json (stage1 result) for stage2!')
        timestamps_data = json.load(open(timestamps_json, 'r'))
        qid2timestamp_res = {item['qid']: item['timestamp'] for item in timestamps_data}
       
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                video_id = item['video']
                video_name = video_id + '.mp4'
                video_path = os.path.join(video_root, video_name)
                query = item['question']
                qid = item['qid']
                timestamps_res = qid2timestamp_res[qid]
                answer = item['answer']
                candidates = []
                for idx in range(item['num_option']):
                    candidates.append(item[f'a{idx}'])
                item_dict = dict(
                    video_id=video_id,
                    video_path=video_path,
                    query=query,
                    qid=qid,
                    timestamps_res=timestamps_res,
                    answer=answer,
                    candidates=candidates
                )
                self.json_dict.append(item_dict)
    def __getitem__(self, idx):
        video_id = self.json_dict[idx]['video_id']
        video_path = self.json_dict[idx]['video_path']
        query = self.json_dict[idx]['query']
        qid = self.json_dict[idx]['qid']
        timestamps_res = self.json_dict[idx]['timestamps_res']
        answer = self.json_dict[idx]['answer']
        candidates = self.json_dict[idx]['candidates']

        if (
            isinstance(timestamps_res, list)
            and len(timestamps_res) == 2
            and all(isinstance(x, (int, float)) for x in timestamps_res)
        ):
            image_list, frame_times, duration = self.decord_method(video_path, timestamps_res)
        else:
            image_list, frame_times, duration = self.decord_method(video_path)

        frame_times = list(chain.from_iterable(repeat(item, 2) for item in frame_times))
        QA_question = self.qa_template(query, candidates)
        video_prefix = '\n'.join(['<FRAME_TIME_REF>: <image>' for _ in range(len(image_list))])
        QA_question = video_prefix + '\n' + self.QA_prompt + '\n' + QA_question + '\n' + self.QA_question_prompt
        MR_question = video_prefix + '\n' + self.MR_prompt.format(query)
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
            'question': [MR_question,QA_question],
            'query': query,
            'qid': qid,
            'timestamps_res': timestamps_res,
            'answer': answer,
            'pixel_values': pixel_values,
            'video_id': video_id,
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


def postprocess_pred_time(time: list, duration: float, reg_max: int) -> list:
    time_stamps = []
    for interval in time:
        for start, end in interval:
            start_time_sec = (float(start) / reg_max * duration)
            end_time_sec = (float(end) / reg_max * duration)
            time_stamps.append([round(float(start_time_sec), 2), round(float(end_time_sec), 2)])
    return time_stamps

def extract_answer(text):
    if isinstance(text, str):
        match = re.search(r'Answer: \((\d+)\)|The answer is \((\d+)\)|Answer is \((\d+)\)|\((\d+)\)', text)
        if match:
            for group in match.groups():
                if group:
                    return group
    return -1  


def evaluate_chat_model(args, model, tokenizer, image_size, use_thumbnail, stage=1):
    random.seed(args.seed)
    json_file = 'data_example/nextgqa/test.json'
    video_root = 'data/NExT-QA/NExTVideo' # NExTVideo video path
    if stage == 1:
        dataset = NExtGQADatasetStage1(
            json_file=json_file, 
            video_root=video_root,
            num_segments=args.num_segments,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num)
        collate_fn = collate_fn_stage1
    else:
        dataset = NExtGQADatasetStage2(
            json_file=json_file, 
            video_root=video_root,
            num_segments=args.num_segments,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            timestamps_json=args.timestamps_json)
        collate_fn = collate_fn_stage2
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    outputs = []
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if stage == 1:
            question, query, qid, answer, pixel_values, video_id, num_patches_lists, duration, frame_times = batch
            question = question[0][0]
        else:
            question, query, qid, timestamps_res, answer, pixel_values, video_id, num_patches_lists, duration, frame_times = batch
            question = question[0][1]
       
        
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
            question=question,
            frame_times=frame_times[0],
            duration=duration[0],
            generation_config=generation_config,
            verbose=True
        )
        
        if stage == 1:
            if len(pred_time):
                time_stamp_str = postprocess_pred_time(pred_time, duration[0], model.reg_max) if pred_time is not None else [[-1, -1]]
                pred_result = time_stamp_str[0]

            outputs.append({
                'qid': qid[0],
                'timestamp': pred_result,
            })
        else:
            outputs.append({
                'qid': qid[0],
                'prediction': extract_answer(pred),
                'timestamp': timestamps_res[0],
                'target': answer[0]
            })

    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))
    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
    if torch.distributed.get_rank() == 0:
        if stage == 1:
            results_file = f'result_timestamp'
        else:
            results_file = f'result'
        output_path = os.path.join(args.out_dir, results_file)
        with open(f'{output_path}.json', 'w') as f:
            json.dump(merged_outputs, f)

# ========== Main =============
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def save_result_merged(result, result_dir, filename):
    import json
    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)
    json.dump(result, open(result_file, "w"))
    if is_dist_avail_and_initialized():
        dist.barrier()
    if is_main_process():
        logging.warning("rank %d starts merging results." % get_rank())
        merged_results = {}
        for rank in range(get_world_size()):
            result_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            res = json.load(open(result_file, "r"))
            for video_id, video_data in res.items():
                if video_id not in merged_results:
                    merged_results[video_id] = video_data
                else:
                    for query, query_data in video_data.items():
                        if query not in merged_results[video_id]:
                            merged_results[video_id][query] = query_data
                        else:
                            print(f"Duplicate query found for video {video_id}: {query}")
        json.dump(merged_results, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)
        print(f"Total unique results: {len(merged_results)}")
        for rank in range(get_world_size()):
            intermediate_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            if os.path.exists(intermediate_file):
                os.remove(intermediate_file)
                print(f"Deleted intermediate file: {intermediate_file}")
    return final_result_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='nextgqa')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results/NExtGQA/1B')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=1)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--num_segments', type=int, default=16)
    parser.add_argument('--stage', type=int, default=1, help='Stage 1 or 2')
    parser.add_argument('--timestamps-json', type=str, default='results/NExtGQA/1B/result_timestamp.json', help='Path to stage1 result json for stage2')
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
    if args.stage == 1:
        evaluate_chat_model(args, model, tokenizer, image_size, use_thumbnail, stage=1)
    else:
        if not args.timestamps_json or not os.path.exists(args.timestamps_json):
            raise ValueError('For stage2, you must provide --timestamps-json (the result json from stage1)!')
        evaluate_chat_model(args, model, tokenizer, image_size, use_thumbnail, stage=2) 