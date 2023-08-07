import requests
import json
import os

import pandas as pd
from tqdm import tqdm

DUMP_DIR = os.path.join(os.getcwd(), 'models')

def fetch_model_data(pbar=None, fetch_tag=None, sort="Most Downloaded", period="AllTime", nsfw='false', limit=100, start_page=None):

    endpoint = 'https://civitai.com/api/v1/models'

    params = {
        'limit': limit,
        'types': "Checkpoint",
        'sort': sort,
        'period': period,
    }

    if start_page:
        params['page'] = start_page
    
    if fetch_tag:
        params['tag'] = fetch_tag

    if nsfw:
        params['nsfw'] = nsfw

    # get the requested results
    res = requests.get(endpoint, params=params)
    items, metadata = res.json()['items'], res.json()['metadata']

    # create progress bar
    if pbar is None:
        pbar = tqdm(total = metadata['totalPages'], desc=f"Page {metadata['currentPage']} fetched")
        pbar.update(1)
    else:
        pbar.set_description(f"Page {metadata['currentPage']} fetched")
        pbar.update(1)
    
    for idx in range(len(items)):
        model_info = items[idx]
        try:
            model_id = model_info['id']
            latestVersion_id = model_info['modelVersions'][0]['id']

            with open(os.path.join(DUMP_DIR, f'{model_id}_{latestVersion_id}.json'), 'w') as fp:
                json.dump(model_info, fp)
        
        except:
            pass
     
    if metadata['currentPage'] < metadata['totalPages']:
        fetch_model_data(pbar, fetch_tag, sort, period, nsfw, limit, start_page=metadata['currentPage']+1)


if __name__ == '__main__':

    # check models folder 
    os.makedirs(DUMP_DIR, exist_ok=True)
    print("==> Start fetching model")
    fetch_model_data()



