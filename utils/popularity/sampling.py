import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def map_count(x):
    x = round(x*194/3436)
    return x+1 if x == 0 else x

if __name__ == '__main__':

    # init variables
    random.seed(0)
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    roster_path = os.path.join(root_dir, 'roster.csv')
    roster = pd.read_csv(roster_path)
    with open('bin2model.pkl', 'rb') as f:
        bin2model = pickle.load(f)
    with open('model2bin.pkl', 'rb') as f:
        model2bin = pickle.load(f)

    # determine subset size
    dist_all = [len(bin2model[i]) for i in range(len(bin2model))]       # sum = 3436
    dist_sub = [map_count(x) for x in dist_all]                         # sum = 197
    print(f'dist_all ({sum(dist_all)}): {dist_all}')
    print(f'dist_sub ({sum(dist_sub)}): {dist_sub}')

    # load model cache & handle special cases
    excep_list = []
    cache_dict = defaultdict(set)
    mvids = roster.modelVersion_id.unique()
    mask = np.isin(mvids, [1, 4, 5, 96284])
    for mvid in mvids[~mask]:
        try:
            cache_dict[model2bin[mvid]].add(mvid)
        except:
            excep_list.append(mvid)
    print(f'excep_list ({len(excep_list)}): {excep_list}')
    print(f'cache_dict: {cache_dict}')

    # sampling with cache
    num_bins = 20
    selected_dict = {}
    candidate_dict = {}
    todo_count_dict = {}
    for i in range(num_bins):
        candidate_dict[i] = list(set(bin2model[i]) - cache_dict[i])
        random.shuffle(candidate_dict[i])
        # print(candidate_dict[i][:5])
        if dist_sub[i] - len(cache_dict[i]) > 0:
            selected_dict[i] = list(cache_dict[i])
            todo_count_dict[i] = dist_sub[i] - len(cache_dict[i])
        else:
            selected_dict[i] = random.sample(list(cache_dict[i]), k=dist_sub[i])
            todo_count_dict[i] = 0
        print(f'[bin {i:2d}]: {dist_sub[i]:2d} models needed, '
              f'{len(cache_dict[i]):2d} models in cache, '
              f'{todo_count_dict[i]:2d} more model(s) needed.')
    to_save = dict(
        num_bins=num_bins,                  # num of bins
        dist_all=dist_all,                  # full set distribution
        dist_sub=dist_sub,                  # subset distribution
        excep_list=excep_list,              # invalid cache models
        cache_dict=cache_dict,              # all cache models
        selected_dict=selected_dict,        # selected cache models
        candidate_dict=candidate_dict,      # candidates to be downloaded
        todo_count_dict=todo_count_dict,    # num of candidates needed
    )
    with open('subset.pkl', 'wb') as f:
        pickle.dump(to_save, f)
    
    # plot histogram for the new subset
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(20), dist_all)
    plt.xlabel('bin_index by log10(download_count)')
    plt.ylabel('model_count')
    plt.title('full set')
    plt.subplot(1, 2, 2)
    plt.bar(range(20), dist_sub)
    plt.xlabel('bin_index by log10(download_count)')
    plt.ylabel('model_count')
    plt.title('new subset')
    plt.savefig('pop_hist_new.png')
