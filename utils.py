from IPython import embed

import os
import json
import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    
def get_has_qrel_label_sample_ids(qrel_file):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    qids = set()
    for line in qrel_data:
        line = line.strip().split("\t")
        query = line[0]
        qids.add(query)
    return qids


def get_finished_sample_ids(output_file_path):
    finished_samples = {}
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            data = f.readlines()
        for line in data:
            line = json.loads(line)
            finished_samples[line['sample_id']] = {}
            if "predicted_rewrite" in line:
                finished_samples[line['sample_id']]["predicted_rewrite"] = line['predicted_rewrite']
            if "predicted_response" in line:
                finished_samples[line['sample_id']]["predicted_response"] = line['predicted_response']
            if "cot" in line:
                finished_samples[line['sample_id']]["cot"] = line['cot']
            if "rewrite_part_text" in line:
                finished_samples[line['sample_id']]["rewrite_part_text"] = line['rewrite_part_text']
                
    return finished_samples    
