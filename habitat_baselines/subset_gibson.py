import os
import json
import gzip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('json_gz')
parser.add_argument('num_out_files', type=int)
parser.add_argument('-o','--out_dir', default='./')
args = parser.parse_args()
json_gz_path = args.json_gz
with gzip.open(json_gz_path,'r') as f:
    data = f.read()
data = json.loads(data.decode('utf-8'))
num_eps = len(data['episodes'])
batch_size = len(data['episodes']) // args.num_out_files
batch_dict = {'episodes': []}
batch_num = 0

scene_dataset = {'data/scene_datasets/gibson/Adrian.glb': 1, \
    'data/scene_datasets/gibson/Albertville.glb': 2, \
    'data/scene_datasets/gibson/Anaheim.glb': 3, \
    'data/scene_datasets/gibson/Andover.glb': 4, \
    'data/scene_datasets/gibson/Angiola.glb': 5, \
    'data/scene_datasets/gibson/Annawan.glb': 6, \
    'data/scene_datasets/gibson/Applewold.glb': 7, \
    'data/scene_datasets/gibson/Arkansaw.glb': 8}

for idx, ep in enumerate(data['episodes']):
    '''
    YOUR CODE HERE
    '''
    if ep['scene_id'] in scene_dataset:
        batch_dict['episodes'].append(ep)
    
out_basename = (
    os.path.basename(args.json_gz)
).replace('.json',f'_{batch_num}.json')
file_name = os.path.join(args.out_dir, out_basename)
json_str = json.dumps(batch_dict) + "\n"
json_bytes = json_str.encode('utf-8')
with gzip.open(file_name, 'w') as f:
    f.write(json_bytes)
    
print(f"Saved {file_name} with {len(batch_dict['episodes'])} episodes")