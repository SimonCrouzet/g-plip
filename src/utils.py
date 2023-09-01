import subprocess
import numpy as np
import torch
import json, os


def get_sample_weights(target, class_weight):
    return torch.Tensor([class_weight[t.int()] for t in target])


def correct_negative_edges(target):
    return target - 1, [i for i, t in enumerate(target) if t.item() != 0]


def dump_to_json(dict, path):
    if os.path.isfile(path):
        existing_json = True
    else:
        existing_json = False
    if existing_json:
        j = json.loads(open(path, encoding="utf-8").read())
    else:
        j = {}
    for k, v in dict.items():
        j[str(k)] = v
    with open(path, "w+", encoding="utf-8") as jFile:
        json.dump(j, jFile)


def dataframe_raw_vectorization_with_numpy(df, keys: list):
    return np.stack([df[k].to_numpy() for k in keys], axis=-1).tolist()


def submit_to_lsf(job_name, config_file):
    f = open("pda_gpu_template.bsub", "r+")
    out_f = open("temp.bsub", "w+")
    out_f.write(f.read() + config_file)
    out_f.close()
    subprocess.check_output("ml Anaconda3; conda activate pda_network", shell=True)
    out = subprocess.check_output("bsub < temp.bsub", shell=True)
    os.remove("temp.bsub")
