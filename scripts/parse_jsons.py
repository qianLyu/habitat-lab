import argparse
import tqdm
import glob
import json
import numpy as np

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

# Gather paths to all json files
json_paths = glob.glob(osp.join(json_dir, '*.json'))


# Recursive defaultdict of defaultdicts
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

    ckpt_id = int(j_path.split('_')[-1].split('.')[0])
    all_j_data[ckpt_id] = stats


# Get mean value filtered by split
def get_mean_val(key, stats, eps):
    return np.mean([
        v[key] for k, v in stats.items()
        if k != 'agg_stats' and int(k) in eps
    ])


# Identify best checkpoint
all_succ_vals = [
    (get_mean_val('success', stats, val_eps), ckpt_id)
    for ckpt_id, stats in all_j_data.items()
]

best_ckpt_id = max(all_succ_vals)[1]

# Use best checkpoint to calculate avg succ on test set
test_set_succ = get_mean_val('success', all_j_data[best_ckpt_id], test_eps)

print(f'Best ckpt ID: {best_ckpt_id}')
print(f'Test set avg succ: {test_set_succ*100:.2f}%')
