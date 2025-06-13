import numpy as np
import pandas as pd
import json
import os.path as osp

def load_file(filename):
    file_type = osp.splitext(filename)[-1]
    if file_type == '.csv':
        data = pd.read_csv(filename)
    else:
        with open(filename, 'r') as fp:
            if file_type == '.json':
                data = json.load(fp)
            elif file_type == '.txt':
                data = fp.readlines()
                data = [datum.rstrip('\n') for datum in data]
    return data

def get_tIoU(loc, span):
    
    if span[0] == span[-1]:
        if loc[0] <= span[0] and span[0] <= loc[1]:
            return 0, 1
        else:
            return 0, 0
    try:
        span_u =  (min(loc[0], span[0]), max(loc[-1], span[-1]))
        span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
    except:
        
        span_u =  (min(loc[0], -1), max(loc[-1], -1))
        span_i = (max(loc[0], -1), min(loc[-1], -1))

    dis_i = (span_i[1] - span_i[0])
    if span_u[1] > span_u[0]:
        IoU = dis_i / (span_u[1] - span_u[0]) 
    else: 
        IoU = 0.0
    if span[-1] > span[0]:
        IoP = dis_i / (span[-1] - span[0]) 
    else:
        IoP = 0.0

    return IoU, IoP

    
def eval_ground(gt_ground, pred_ground, pred_qa=None, subset=None, gs=False):
    
    mIoU, mIoP = 0, 0
    cnt, cqt = 0, 0
    crt3, crt5 = 0, 0
    crtp3, crtp5 = 0, 0

    for vid, anno in gt_ground.items():
        for qid, locs in anno['location'].items():
            if not (f'{vid}_{qid}' in pred_ground):
                print(vid, qid)
                continue
            if subset != None:
                if not (f'{vid}_{qid}' in subset):
                    continue
            max_tIoU, max_tIoP = 0, 0
            for loc in locs: 
                span = pred_ground[f'{vid}_{qid}']
                if gs: span = np.round(np.asarray(span)*anno['duration'], 1)
                tIoU, tIoP = get_tIoU(loc, span)
                if tIoU > max_tIoU:
                    max_tIoU = tIoU
                if tIoP > max_tIoP:
                    max_tIoP = tIoP
            if max_tIoP >= 0.3:
                crtp3 += 1
                if  max_tIoP >= 0.5:
                    crtp5 += 1
                    kid = f'{vid}_{qid}'
                    if pred_qa:
                        if pred_qa[kid]['answer'] == pred_qa[kid]['prediction']:
                            cqt+= 1
            if max_tIoU >= 0.3:
                crt3 += 1
                if max_tIoU >= 0.5:
                    crt5 += 1
            cnt += 1
            mIoU += max_tIoU
            mIoP += max_tIoP
            
    mIoU = mIoU /cnt * 100
    mIoP = mIoP/cnt * 100
    print('Acc&GQA mIoP TIoP@0.3 TIoP@0.5 mIoU TIoU@0.3 TIoU@0.5 ')
    print('{:.1f} \t {:.1f}\t {:.1f}\t {:.1f} \t {:.1f} \t {:.1f} \t {:.1f}'.format(cqt*1.0/cnt*100, mIoP,
          crtp3*1.0/cnt*100, crtp5*1.0/cnt*100, 
          mIoU, crt3*1.0/cnt*100, crt5*1.0/cnt*100))


def combine(pred1, pred2, gt):
    """
    pred1: ground segment by gaussian mask
    pred2: ground segment by post-hoc attention
    gt: to get NExT-GQA subset
    """
    def _cb_seg(seg1, seg2, way='uni'):
        # print(seg1, seg2)
        if way == 'uni':
            ts = [seg1[0], seg1[1], seg2[0], seg2[1]]
            ts = sorted(ts)
            new_seg = [ts[0], ts[-1]]
        elif way == 'itsc':
            start = seg1[0] if seg1[0] > seg2[0] else seg2[0]
            end = seg1[1] if seg1[1] < seg2[1] else seg2[1]
            if not (start <= end):
                new_seg = seg2.tolist() #trust more on attention
            else:
                new_seg = [start, end]
        return new_seg
    
    cb_ground = {}
    for vqid, seg in pred1.items():
        vid, qid = vqid.split('_')
        if not (vid in gt and qid in gt[vid]['location']):
            continue 
        duration = gt[vid]['duration']
        seg = np.round(np.asarray(seg)*duration, 1)
        seg_att = np.asarray(pred2[vqid])
        new_seg  = _cb_seg(seg, seg_att, way='itsc')
        cb_ground[vqid] = new_seg
    
    # save_to()
    return cb_ground


def main():
    gt_file = 'internvl_chat/data_example/nextgqa/gsub_test.json'
    pred_file = 'internvl_chat/results/NExtGQA/8B/result_gd.json'
    qa_file = 'internvl_chat/results/NExtGQA/8B/result_qa.json'

    gt_ground = load_file(gt_file)
    pred_ground = load_file(pred_file)
    pred_qa = load_file(qa_file)
    
    eval_ground(gt_ground, pred_ground, pred_qa, subset=None, gs=False)


if __name__ == "__main__":
    main()
