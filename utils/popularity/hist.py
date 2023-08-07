
import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from math import log10

NUM_BINS = 20
plt.figure(figsize=(10, 5))

# full model set
download_count = []
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
full_set_dir = os.path.join(root_dir, 'everything', 'models')
for filename in tqdm(os.listdir(full_set_dir)):
    meta_path = os.path.join(full_set_dir, filename)
    with open(meta_path, 'r') as f:
        download_count.append(json.loads(f.read())['stats']['downloadCount'])
plt.subplot(1, 2, 1)
hist, bins, _ = plt.hist([log10(i+1) for i in download_count], bins=NUM_BINS)
with open('bins.pkl', 'wb') as f:
    pickle.dump(bins, f)
plt.xlabel('log10(download_count)')
plt.ylabel('model_count')
plt.title('full set')

# current model set
idx2model, model2idx, model2pop = {}, {}, {}
metadata_path = os.path.join(root_dir, 'generated', 'train', 'metadata.csv')
metadata = pd.read_csv(metadata_path)
model_ids = sorted(metadata.modelVersion_id.unique())
for i, mvid in enumerate(model_ids):
    idx2model[i] = mvid
    model2idx[mvid] = i
roster_path = os.path.join(root_dir, 'roster.csv')
roster = pd.read_csv(roster_path)
for mvid in tqdm(model_ids):
    model2pop[mvid] = roster[roster.modelVersion_id == mvid].model_download_count.values[0]
plt.subplot(1, 2, 2)
plt.hist([log10(i+1) for i in model2pop.values()], bins=NUM_BINS)
plt.xlabel('log10(download_count)')
plt.ylabel('model_count')
plt.title('old subset')

plt.savefig('pop_hist_old.png')
