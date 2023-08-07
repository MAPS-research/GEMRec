
import os
import json
import math
import pickle

from tqdm import tqdm
from collections import defaultdict

# load bins
with open('bins.pkl', 'rb') as f:
    BINS = pickle.load(f)
    print(f'\n==> bins: {BINS}')

# util function
def count2idx(count):
    log_count = math.log10(count+1)
    for i in range(len(BINS-1)):
        if BINS[i] <= log_count <= BINS[i+1]:
            return i
    raise ValueError('Invalid bins.')

if __name__ == '__main__':
    model2bin = {}
    bin2model = defaultdict(list)
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    full_set_dir = os.path.join(root_dir, 'everything', 'models')
    for filename in tqdm(os.listdir(full_set_dir)):
        meta_path = os.path.join(full_set_dir, filename)
        with open(meta_path, 'r') as f:
            metadata = json.loads(f.read())
        bin_idx = count2idx(metadata['stats']['downloadCount'])
        mid, mvid = tuple(filename[:-len('.json')].split('_'))
        bin2model[bin_idx].append(int(mvid))
        model2bin[int(mvid)] = bin_idx
    with open('bin2model.pkl', 'wb') as f:
        pickle.dump(bin2model, f)
    with open('model2bin.pkl', 'wb') as f:
        pickle.dump(model2bin, f)