import requests
import pandas as pd
import random
from tqdm import tqdm
import os
import csv
from pprint import pprint

def fetch_new_prompts(total=30, limit=100, modelId=None, modelVersionId=None, nsfw='false', sort='Most Reactions', period='AllTime', start_page=None):

    endpoint = 'https://civitai.com/api/v1/images'

    params ={
        'limit': limit,
        'sort': sort,
        'period': period,
    }

    if start_page:
        params['page'] = start_page

    if nsfw:
        params['nsfw'] = nsfw
    
    if modelId:
        params['modelId'] = modelId
    
    if modelVersionId:
        params['modelVersionId'] = modelVersionId

    # get the requested results
    res = requests.get(endpoint, params=params)
    # metadata here stands for api metadata
    items, metadata = res.json()['items'], res.json()['metadata']
    print('Data Fetched')
    
    for idx in range(len(items)):
        if total == 0:
            break
        
        # meta here stands for image metadata
        meta = items[idx]['meta']
        if meta is not None:
            data_needed = ['Size', 'seed', 'prompt', 'negativePrompt', 'sampler', 'cfgScale']
            if all(x in meta.keys() for x in data_needed):
                if all(meta[x] is not None for x in data_needed):
                    with open('./promptset_c1.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [items[idx]['id'], 'civitai', meta['Size'], meta['seed'], meta['prompt'].replace('\n', ', '), meta['negativePrompt'].replace('\n', ', '), meta['sampler'], meta['cfgScale']]
                        )
                    total -= 1

    if total > 0:
        start_page = metadata['currentPage']+1
        fetch_new_prompts(total, limit, modelId, modelVersionId, nsfw, sort, period, start_page)


if __name__ == '__main__':

    # create promptset_c1 csv
    if not os.path.exists('./promptset_c1.csv'):
        with open('./promptset_c1.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['prompt_id','tag','size','seed','prompt','negativePrompt','sampler','cfgScale']
            )

    fetch_new_prompts()




