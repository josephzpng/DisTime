import json
import os.path as osp

def process_and_split(res_file):
    with open(res_file, 'r') as fp:
        data = json.load(fp)
    test_qa = {}
    test_gd = {}
    for cnt, item in enumerate(data):
        qid = item['qid']
        pred = item['prediction']
        target = item['target']
        pred_time = item['timestamp']

        vid = qid.split('_')[1]
        qs_id = qid.split('_')[-1]
        key_id = vid + '_' + qs_id

        if isinstance(pred_time, list):
            loc = [pred_time[0], pred_time[-1]]
        else:
            loc = [pred_time, pred_time]

        try:
            pred_int = int(pred)
        except Exception:
            pred_int = pred

        test_qa[key_id] = {"prediction": pred_int, "answer": target}
        test_gd[key_id] = loc
        if cnt % 500 == 0:
            print({"qa": test_qa[key_id], "gd": test_gd[key_id]})

    qa_file = res_file.replace('.json', '_qa.json')
    gd_file = res_file.replace('.json', '_gd.json')
    with open(qa_file, 'w') as fp:
        json.dump(test_qa, fp)
    with open(gd_file, 'w') as fp:
        json.dump(test_gd, fp)

def main():
    res_file = 'internvl_chat/results/NExtGQA/1B/result.json'
    process_and_split(res_file)

if __name__ == "__main__":
    main() 