import argparse
import tqdm
import json
import numpy as np
import os

from collections import defaultdict
from os import path as osp
from matplotlib import pyplot as plt

VALIDATION_SCENE_IDS = [4, 0, 3, 5]

val_eps = []
for s_id in VALIDATION_SCENE_IDS:
    val_eps.extend(list(range(s_id * 71, (s_id + 1) * 71)))
test_eps = list(filter(lambda x: x not in val_eps, range(994)))

parser = argparse.ArgumentParser()
parser.add_argument('json_dir')
args = parser.parse_args()

json_dir = args.json_dir
json_paths = []

# Gather paths to all json files
for root, dirs, files in os.walk(json_dir):
    for f in files:
        if f.endswith('.json'):
            json_paths.append(osp.join(root, f))


# Recursive defaultdict
def rec_dd():
    return defaultdict(rec_dd)


# Dict of dicts
all_j_data = rec_dd()

# Load data from all json files
print('Parsing all jsons...')
for j_path in tqdm.tqdm(json_paths):
    # Load from JSON file
    with open(j_path) as f:
        stats = json.load(f)

    j_key = osp.abspath(j_path).split('/jsons/')[-1]
    exp_type = j_key.split('/')[0]  # either social or pointnav
    model_name_seed = osp.basename(osp.dirname(j_key))
    model_name = model_name_seed.split('_seed')[0]
    seed = model_name_seed[len(model_name) + 1:]
    ckpt_index = int(j_key.split('_')[-1].split('.')[0])

    all_j_data[exp_type][model_name][seed][ckpt_index] = stats


# Get mean value filtered by split
def get_mean_val(key, stats, eps):
    return np.mean([
        v[key] for k, v in stats.items()
        if k != 'agg_stats' and int(k) in eps
    ])


# Find best checkpoint for each seed
print('Determining best checkpoints...')
best_ckpts = rec_dd()
parse_count = 0
for exp_type, v in all_j_data.items():
    for model_name, vv in v.items():
        for seed, vvv in vv.items():
            ckpt_stats = [
                (
                    get_mean_val('success', stats, val_eps),
                    ckpt_index
                )
                for ckpt_index, stats in vvv.items()
            ]
            best_ckpts[exp_type][model_name][seed] = max(ckpt_stats)[1]
            parse_count += len(vvv)
            print(f'Parsed {parse_count}/{len(json_paths)}\r', end='')
print('')


# Get metrics on test split for best checkpoints

#
# # Filter best results
#
#     # Iterate through episodes
#     for k,v in all_episode_stats.items():
#         if k == 'agg_stats':
#             continue
#
#
