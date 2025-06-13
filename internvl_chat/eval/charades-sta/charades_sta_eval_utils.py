import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

"""
modify from https://github.com/TencentARC/UMT/tree/main/tools
"""


def temporal_intersection(windows1, windows2, aligned=False):
    """
    Compute the intersections among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the intersections
            among aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed \
            intersection values.
    """
    if aligned:
        s = torch.max(windows1[:, 0], windows2[:, 0])
        e = torch.min(windows1[:, 1], windows2[:, 1])
    else:
        s = torch.max(windows1[:, None, 0], windows2[:, 0])
        e = torch.min(windows1[:, None, 1], windows2[:, 1])

    inter = (e - s).clamp(0)
    return inter


def temporal_area(windows):
    """
    Compute the areas of temporal windows.

    Args:
        windows (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed. They
            are expected to be in ``(start, end)`` format.

    Returns:
        :obj:`nn.Tensor[N]`: The computed areas.
    """
    return windows[:, 1] - windows[:, 0]


def temporal_iou(windows1, windows2, aligned=False):
    """
    Compute the intersection-over-unions (IoUs) among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the IoU among
            aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed pairwise \
            IoU values.
    """
    area1 = temporal_area(windows1)
    area2 = temporal_area(windows2)

    inter = temporal_intersection(windows1, windows2, aligned=aligned)

    if aligned:
        iou = inter / (area1 + area2 - inter)
    else:
        iou = inter / (area1[:, None] + area2 - inter)

    return iou


class MREvaluator:
    def __init__(self, rank=[1, 5], iou_thr=[0.3, 0.5, 0.7], detail=False, **kwargs):
        self.rank = rank
        self.iou_thr = iou_thr
        self.kwargs = kwargs
        self.labels = 0
        # self.hits_top1 = 0
        # self.hits_top5 = 0
        self.hits_dict = dict()
        for r in self.rank:
            for iou in iou_thr:
                self.hits_dict["%d-%.2f" % (r, iou)] = 0
        self.detail = detail
        self.vid_perform = dict()
        charades_sta_test_json = "internvl_chat/data_example/charades/charades_sta_test.json"
        self.gt_dict = self._load_json_db_mr(charades_sta_test_json)
        self.iou_list = []

    def evalute(self, results):

        # # results: from model
        # sorted_res = dict()
        # for one_vid in results:
        #     vid = one_vid["video_id"]
        #     sorted_res[vid] = dict()
        #
        #     for i in range(len(one_vid["segments"])):
        #         label = one_vid["labels"][i].cpu().item()
        #         seg = one_vid["segments"][i].cpu().tolist()
        #         score = one_vid["scores"][i].cpu().item()
        #         sorted_res[vid].setdefault(label, []).append([seg[0], seg[1], score])

        # gts: from dataset
        err = []
        iou_dict = dict()

        for vid in self.gt_dict.keys():
            one_gt = self.gt_dict[vid]
            assert vid == one_gt["id"]
            num_gt = one_gt["segments"].shape[0]
            r1_50 = 0
            r1_70 = 0
            for i in range(num_gt):
                query = one_gt["description"][i]
                self.labels += 1
                one_res = []
                if vid in results and query in results[vid]:
                    st = results[vid][query]["start"]
                    et = results[vid][query]["end"]
                    conf = results[vid][query]["conf"]
                    one_res = [[st, et, conf]]
                else:
                    err.append([vid, query])
                    print("=err", err[-1])
                    f.write(f"Error: {err[-1]}\n")
                    continue
                one_res = torch.Tensor(one_res)
                for k in self.rank:
                    for thr_i, thr in enumerate(self.iou_thr):
                        inds = torch.argsort(one_res[:, -1], descending=True)
                        keep = inds[:k]
                        bnd = one_res[:, :-1][keep]
                        gt = torch.from_numpy(one_gt["segments"][i])
                        iou = temporal_iou(gt[None], bnd)
                        if k == 1 and thr_i == 0:
                            max_iou = int(iou.max().item() * 100)
                            iou_dict.setdefault(max_iou, []).append(conf)
                            
                            self.iou_list.append(max_iou)
                        if iou.max() >= thr:
                            self.hits_dict["%d-%.2f" % (k, thr)] += 1
                            if k == 1 and thr == 0.7:
                                r1_70 += 1
                            elif k == 1 and thr == 0.5:
                                r1_50 += 1
            if self.detail:
                self.vid_perform[vid] = {
                    "r1_50": r1_50 / num_gt * 100,
                    "r1_70": r1_70 / num_gt * 100,
                }


        self.err = err
        self.iou_dict = iou_dict


    def cal_miou(self):
        return np.mean(self.iou_list)

    def temporal_iou_according_vid_query(self, vid, query, results, topk=1):
        one_gt = self.gt_dict[vid]
        assert vid == one_gt["id"]

        # print(results[vid][query])
        st = results[vid][query]["start"]
        et = results[vid][query]["end"]
        conf = results[vid][query]["viclip"]
        one_res = [[st, et, conf]]
        one_res = torch.Tensor(one_res)
        inds = torch.argsort(one_res[:, -1], descending=True)
        keep = inds[:topk]
        bnd = one_res[:, :-1][keep]

        gt_temporal = None
        for idx, _ in enumerate(one_gt["segments"]):
            if query == one_gt["description"][idx]:
                gt_temporal = one_gt["segments"][idx]
                break
        if gt_temporal is None:
            raise Exception("No such query in gt")

        gt = torch.from_numpy(gt_temporal)
        iou = temporal_iou(gt[None], bnd)
        iou = iou.max().item() * 100
        return iou

    def summary(self):
        for k, v in self.hits_dict.items():
            self.hits_dict[k] = v / (self.labels + 1e-5)
        # print("### AP in charades_sta : ###")
        # print("R1@70: %.2f" % self.hits_dict["1-0.70"])
        # print("R1@50: %.2f" % self.hits_dict["1-0.50"])
        # print("R5@70: %.2f" % self.hits_dict["5-0.70"])
        # print("R5@50: %.2f" % self.hits_dict["5-0.50"])
        return self.hits_dict

    def get_detail(self):
        return self.vid_perform


    def _load_json_db_mr(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # fill in the db (immutable afterwards)
        dict_db = dict()
        for key, value in json_db.items():
            # # get fps if available
            # if self.default_fps:
            #     fps = self.default_fps
            # elif 'fps' in value:
            #     fps = value['fps']
            # else:
            #     assert False, "Unknown video FPS."

            # get video duration if available
            # if 'duration' in value:
            duration = value['duration']
            # else:
            #     print("==warning: %s have no duration" % key)
            #     duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels, description = [], [], []
                for act_i, act in enumerate(value['annotations']):
                    if act["segment"][0] > act["segment"][1]:
                        print("left > right, switch", act["segment"][0], act["segment"][1])
                        segments.append([act["segment"][1], act["segment"][0]])
                        print(segments[-1])
                    else:
                        segments.append(act['segment'])
                    labels.append([act_i])
                    description.append(act["label"])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = []
                description = []

            dict_db[key] = {
                'id': key,
                # 'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels,
                'description': description,
                "num_classes": len(labels),
            }

        return dict_db
    

def parse_mrblip(json_path):
    print(json_path)
    with open(json_path, 'r') as in_f:
        results = json.load(in_f) 

    evaluator = MREvaluator(detail=True)
    evaluator.evalute(results)
    print(evaluator.cal_miou())
    summary = evaluator.summary()
    
    return summary

if __name__ == "__main__":
    json_path = "internvl_chat/results/Charades-STA/8B/results.json"
    summary = parse_mrblip(json_path)
    print(summary)
