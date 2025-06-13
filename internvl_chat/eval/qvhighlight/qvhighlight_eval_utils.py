import time
from contextlib import suppress
import torch
from tqdm import tqdm
import torch.distributed as dist

import torch.nn as nn
import re
import logging
import os
import json
import ast
from collections import OrderedDict, defaultdict
import numpy as np 
import multiprocessing as mp


def store_image_embedding(file_path, id, image_embeddings):
    file_path = os.path.join(file_path, id + '.npy')

    # image_features = []
    # for i in range(len(image_embeddings)):
    #     image_features[i] = image_embeddings[i]

    np.save(file_path, image_embeddings)
      

def post_process(pred):
    """Post process predicted output to be in the format of moments, i.e. [[0, 1], [4, 7]].
        - if no comma, i.e. " " → add comma, i.e. ", "
        - if t_start > t_end → swap them
        - if two comma: ",," → ","
    Args:
        pred (str): predicted output with potential errors, e.g. "[[0, 1], [4, 7]]"
    Returns:
        str: post processed predicted output, e.g. "[[0, 1], [4, 7]]"
    """

    # pred = pred.split("</s>")[0]
    pred = pred.split('<|end|>')[0]

    # check if the string has the right format of a nested list
    # the list should look like this: [[0, 1], [4, 7], ...]
    # if not, return "[[-1, -1]]"
    if not re.match(r"\[\[.*\]\]", pred):
        return "[[-1, -1]]"

    # remove the first and last bracket
    # e.g.
    #   [[0, 1] [4, 7]] -> [0, 1] [4, 7]
    #   [[0, 1], [4, 7]] -> [0, 1], [4, 7]
    pred = pred[1:-2]

    # split at any white space that is followed by a "[" to get a list of windows
    # e.g.
    #   "[0, 1] [4, 7]" → ["[0, 1]", "[4, 7]"]
    #   "[0, 1], [4, 7]" → ["[0, 1],", "[4, 7]"]
    windows = re.split(r"\s+(?=\[)", pred)

    output = []

    for window in windows:
        # if there is one or more comma at the end of the window, remove it
        # e.g.
        #   "[0, 1]," → "[0, 1]"
        #   "[0, 1],," → "[0, 1]"
        window = re.sub(r",+$", "", window)

        # if there is no comma in the window, add one
        # e.g.
        #   "[0 1]" → "[0, 1]"
        window = re.sub(r"(\d) (\d)", r"\1, \2", window)

        # if there are two or more commas in the window, remove all but one
        # e.g.
        #   "[0,, 1]" → "[0, 1]"
        window = re.sub(r",+", ",", window)

        # if the two numbers are not in the right order, swap them
        # e.g.
        #   "[1, 0]" → "[0, 1]"
        # find all numbers in the window
        numbers = re.findall(r"\d+", window)
        # get the two numbers
        if len(numbers) == 2:
            t_start, t_end = numbers
            if int(t_start) > int(t_end):
                window = "[" + t_end + ", " + t_start + "]"

        output.append(window)

    output = "[" + ", ".join(output) + "]"

    return output
        
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



def save_result(result, result_dir, filename, remove_duplicate=""):
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
        # combine results from all processes
        result = []
        for rank in range(get_world_size()):
            result_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            res = json.load(open(result_file, "r"))
            result += res
        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new
        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)
    return final_result_file

def save_result_ETBench(result, result_dir, filename, remove_duplicate=""):
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
        # combine results from all processes
        result = []
        for rank in range(get_world_size()):
            result_file = os.path.join(
                result_dir, "%s_rank%d.json" % (filename, rank)
            )
            res = json.load(open(result_file, "r"))
            result += res

        seen = set()
        results = []

        # 防止dataloader重复
        for res in result:
            identifier = (res['idx'], res['source'], res['task'])  # 假设我们根据'id'和'name'键来判断
            if identifier in seen:
                print("duplicates:", identifier)
            else:
                seen.add(identifier)
                results.append(res)
        result = results

        # if remove_duplicate:
        #     result_new = []
        #     id_list = []
        #     for res in result:
        #         if res[remove_duplicate] not in id_list:
        #             id_list.append(res[remove_duplicate])
        #             result_new.append(res)
        #     result = result_new
        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)
    return final_result_file


# def save_result(result, result_dir, filename, remove_duplicate=""):
#     import json
#     final_result_file = os.path.join(result_dir, "%s.json" % filename)

#     json.dump(result, open(final_result_file, "w"))
    
#     print("result file saved to %s" % final_result_file)

#     return final_result_file


def after_evaluation(eval_result_file, split_name, **kwargs):

    metrics = report_metrics(
        eval_result_file=eval_result_file, split_name=split_name
    )
    
    return metrics

def report_metrics(eval_result_file, split_name):
    """Report metrics to the logger and save to the output directory.
    Compute r1 for the moment retrieval task.
    Args:
        eval_result_file (str): path to the evaluation result file.
        split_name (str): split name, e.g. val, test
    """
    results = json.load(open(eval_result_file))
    total_num = len(results)
    # results_interpreted = [
    #     {
    #         "qid": r["qid"],
    #         "pred_relevant_windows": moment_str_to_list(r["prediction"]),
    #         "relevant_windows": moment_str_to_list(r["target"]),
    #     }
    #     for r in results
    # ]

    results_interpreted = []
    count = 0
    qid_count = {}
    for r in results:
        # qid_base = r["qid"].split("_")[0]

        pred_relevant_windows = moment_str_to_list(r["prediction"])
        relevant_windows = moment_str_to_list(str(r["target"]))
        
        # relevant_windows = eval(r["target"])
        # relevant_windows = [[round(num) for num in sublist] for sublist in relevant_windows]
        # query = r["query"]
        
        # if qid_base not in qid_count:
        #     qid_count[qid_base] = 0
        # else:
        #     qid_count[qid_base] += 1

        # if [[-1, -1]] in pred_relevant_windows or [[-1, -1]] in relevant_windows:

        # qid = r["qid"]

        qid = f"Charades-STA_{count}"

        # print(qid, pred_relevant_windows, relevant_windows)
        results_interpreted.append(
            {
                "qid": qid,
                "pred_relevant_windows": pred_relevant_windows,
                "relevant_windows": relevant_windows,
                # "query": query,
            }
        )
        count += 1



    # # compute metrics
    # # Recall at 1 and mean IoU
    # r1, r1_avg, mIoU, invalid_pred_num = r1_and_mIoU(submission=results_interpreted)
    # # mAP
    # if "QVHighlight" in results[0]["qid"]:
    #     mAP = compute_mr_ap(
    #         submission=results_interpreted,
    #         iou_thds=np.linspace(0.5, 0.95, 10),
    #         max_gt_windows=None,
    #         max_pred_windows=None,
    #         num_workers=8,
    #         chunksize=50,
    #     )
    # else:
    #     mAP = {str(t): 0 for t in iou_tresholds}
    # # if there are no keys 0.5 and 0.7, add them with value 0
    # if "0.5" not in mAP:
    #     mAP["0.5"] = 0
    # if "0.7" not in mAP:
    #     mAP["0.7"] = 0
    # if "0.75" not in mAP:
    #     mAP["0.75"] = 0
    all_metrics = eval_submission(results_interpreted, results_interpreted)
    r1_avg = all_metrics["brief"]["MR-full-R1-avg"]
    mIoU = all_metrics["brief"]["MR-full-mIoU"]
    invalid_pred_num = all_metrics["brief"]["MR-full-invalid_pred_num"]
    # log metrics
    metrics = {
        "agg_metrics": r1_avg,
        "r1": all_metrics["full"]["MR-R1"],
        "mAP": all_metrics["full"]["MR-mAP"],
        "mIoU": mIoU,
        "invalid_predictions": invalid_pred_num / total_num,
        "total": total_num,
    }
    print("metrics in _report_metrics: ", metrics)
    logging.info(metrics)
    return metrics


def moment_str_to_list(m):
    """Convert a string of moments to a list of moments.
    If predicted string is not a list, it means that the model has not yet learned to predict the right format.
    In that case, we return [[-1, -1]] to represent an error.
    This will then lead to an IoU of 0.
    Args:
        m (str): a string of moments, e.g. "[[0, 1], [4, 7]]"
    Returns:
        list: a list of moments, e.g. [[0, 1], [4, 7]]
    """
    if m == "[[-1, -1]]":
        return [[-1, -1]]
    # check if the string has the right format of a nested list using regex
    # the list should look like this: [[0, 1], [4, 7], ...]
    # if not, return [[-1, -1]]
    if not re.match(r"\[\[.*\]\]", m):
        return [[-1, -1]]
    
    try:
        _m = ast.literal_eval(m)
    except:
        return [[-1, -1]]
    # if _m is not a list, it means that the model has not predicted any relevant windows
    # return error
    if not isinstance(_m, list):
        # raise ValueError()
        return [[-1, -1]]
    # if not nested list, make it nested
    # if a sublist of _m has more than 2 elements, it means that the model has not learned to predict the right format
    # substitute that sublist with [-1, -1]
    for i in range(len(_m)):
        # if isinstance(i, int):
        #     _m[i] = [-1, -1]
        if len(_m[i]) != 2:
            # print(f"Got a sublist with more or less than 2 elements!{_m[i]}")
            _m[i] = [-1, -1]
    return _m


def eval_submission(submission, ground_truth, verbose=True, match_number=True):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, (
            f"qids in ground_truth and submission must match. "
            f"use `match_number=False` if you wish to disable this check"
        )
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]
    
    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, submission, verbose=verbose
        )
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            "MR-full-mAP": moment_ret_scores["full"]["MR-mAP"]["average"],
            "MR-full-mAP@0.5": moment_ret_scores["full"]["MR-mAP"]["0.5"],
            "MR-full-mAP@0.75": moment_ret_scores["full"]["MR-mAP"]["0.75"],
            "MR-short-mAP": moment_ret_scores["short"]["MR-mAP"]["average"],
            "MR-middle-mAP": moment_ret_scores["middle"]["MR-mAP"]["average"],
            "MR-long-mAP": moment_ret_scores["long"]["MR-mAP"]["average"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["MR-R1"]["0.5"],
            "MR-full-R1@0.7": moment_ret_scores["full"]["MR-R1"]["0.7"],
            "MR-full-R1-avg": moment_ret_scores["full"]["MR-R1-avg"],
            "MR-full-mIoU": moment_ret_scores["full"]["MR-mIoU"],
            "MR-full-invalid_pred_num": moment_ret_scores["full"][
                "MR-invalid_pred_num"
            ],
        }
        eval_metrics_brief.update(
            sorted(
                [(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]
            )
        )


    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(
        sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0])
    )
    return final_eval_metrics

    

def eval_moment_retrieval(submission, ground_truth, verbose=True):

    # To use API without much modification, keep names as they are.
    # They simply won't mean anything anymore since we removed iterating over different ranges
    # that were specific to the QVH dataset
    range_names = ["short", "middle", "long", "full"]

    ret_metrics = {}
    for name in range_names:
        if verbose:
            start_time = time.time()

        _submission = submission
        _ground_truth = ground_truth

        print(
            f"{name}: {len(_ground_truth)}/{len(ground_truth)}="
            f"{100*len(_ground_truth)/len(ground_truth):.2f} examples."
        )
        iou_thd2average_precision = compute_mr_ap(
            _submission, _ground_truth, num_workers=8, chunksize=50
        )
        iou_thd2recall_at_one, r1_avg, mIoU, invalid_pred_num = compute_mr_r1(
            _submission, _ground_truth
        )
        # iou_thd2recall_at_one, r1_avg, mIoU, invalid_pred_num = r1_and_mIoU(_submission)
        ret_metrics[name] = {
            "MR-mAP": iou_thd2average_precision,
            "MR-R1": iou_thd2recall_at_one,
            "MR-R1-avg": r1_avg,
            "MR-mIoU": mIoU,
            "MR-invalid_pred_num": invalid_pred_num,
        }
        if verbose:
            print(
                f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds"
            )
    return ret_metrics


def compute_mr_ap(
    submission,
    ground_truth,
    iou_thds=np.linspace(0.5, 0.95, 10),
    max_gt_windows=None,
    max_pred_windows=None,
    num_workers=8,
    chunksize=50,
):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    gt_qid2data = defaultdict(list)
    for d in submission:
        qid = d["qid"]

        # get predicted windows
        pred_windows = (
            d["pred_relevant_windows"][:max_pred_windows]
            if max_pred_windows is not None
            else d["pred_relevant_windows"]
        )
        for w in pred_windows:
            pred_qid2data[qid].append(
                {
                    "video-id": d["qid"],  # in order to use the API
                    "t-start": w[0],
                    "t-end": w[1],
                }
            )

        # get target windows
        gt_windows = (
            d["relevant_windows"][:max_gt_windows]
            if max_gt_windows is not None
            else d["relevant_windows"]
        )
        for w in gt_windows:
            gt_qid2data[qid].append(
                {"video-id": d["qid"], "t-start": w[0], "t-end": w[1]}
            )

    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [
        [qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data
    ]
    from functools import partial
  
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds
    )

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(
                compute_ap_from_triple, data_triples, chunksize=chunksize
            ):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap

def compute_average_precision_detection_wrapper(
    input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds
    )
    return qid, scores


def compute_average_precision_detection(
    ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', and 't-end'.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        dict: Average precision at different iou thresholds.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item["index"] = i
        ground_truth_by_videoid.setdefault(item["video-id"], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred["video-id"] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred["video-id"]]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array(
            [
                [pred["t-start"], pred["t-end"]],
            ]
        )
        _gt = np.array([[gt["t-start"], gt["t-end"]] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]["index"]] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(
            precision_cumsum[t_idx, :], recall_cumsum[t_idx, :]
        )
    return ap

def compute_temporal_iou_batch_cross(spans1, spans2):
    """
    Args:
        spans1: (N, 2) np.ndarray, each row defines a span [st, ed]
        spans2: (M, 2) np.ndarray, ...

    Returns:
        iou: (N, M) np.ndarray
        union: (N, M) np.ndarray
    >>> spans1 = np.array([[0, 0.2, 0.9], [0.5, 1.0, 0.2]])
    >>> spans2 = np.array([[0, 0.3], [0., 1.0]])
    >>> compute_temporal_iou_batch_cross(spans1, spans2)
    (tensor([[0.6667, 0.2000],
         [0.0000, 0.5000]]),
     tensor([[0.3000, 1.0000],
             [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])  # (N, M)
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])  # (N, M)

    inter = np.clip(right - left, 0, None)  # (N, M)
    union = areas1[:, None] + areas2[None, :] - inter  # (N, M)

    iou = inter / union
    return iou, union


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {
        d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission
    }  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(
            f"{np.mean(pred_gt_iou >= thd) * 100:.2f}"
        )

    # number of invalid predictions
    invalid_pred_num = 0
    for pred in pred_windows:
        if -1 in pred:
            invalid_pred_num += 1

    # avg recall@1
    r1_avg = np.mean(list(iou_thd2recall_at_one.values()))

    # mean IoU
    mIoU = np.mean(pred_gt_iou)

    return iou_thd2recall_at_one, r1_avg, mIoU, invalid_pred_num




def compute_temporal_iou_batch_paired(pred_windows, gt_windows):
    """compute intersection-over-union along temporal axis for each pair of windows in pred_windows and gt_windows.
    Args:
        pred_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
    Returns:
        iou (float): np.ndarray, (N, )

    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(
        0,
        np.minimum(pred_windows[:, 1], gt_windows[:, 1])
        - np.maximum(pred_windows[:, 0], gt_windows[:, 0]),
    )
    union = np.maximum(pred_windows[:, 1], gt_windows[:, 1]) - np.minimum(
        pred_windows[:, 0], gt_windows[:, 0]
    )  # not the correct union though
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union != 0
    )


if __name__ == "__main__":
    json_file = 'internvl_chat/results/QVHighlight/8B/results.json'
    _ = after_evaluation(json_file, 'test')