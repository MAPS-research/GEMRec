import requests
import pandas as pd
import random
from tqdm import tqdm
import os
import csv
from pprint import pprint
from transformers import CLIPTokenizerFast


def fetch_new_prompts(total=30, limit=100, modelId=None, modelVersionId=None, nsfw='None', sort='Most Reactions', period='AllTime', start_page=None):

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
                    print(items[idx]['nsfw'], items[idx]['nsfwLevel'])
                    # process prompt and negativePrompt
                    prompt = process_prompt(meta['prompt'])
                    negativePrompt = process_prompt(meta['negativePrompt'])

                    with open('./promptset_v5.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [items[idx]['id'], 'civitai', meta['Size'], meta['seed'], prompt, negativePrompt, meta['sampler'], meta['cfgScale']]
                        )
                    total -= 1

    if total > 0:
        start_page = metadata['currentPage']+1
        fetch_new_prompts(total, limit, modelId, modelVersionId, nsfw, sort, period, start_page)


def process_prompt(prompt):
    prompt = prompt.replace('\n', ', ').replace('(', '').replace(')', '').replace(':', '').replace('[', '').replace(']', '').replace('|', '')
    prompt = ', '.join([p.strip() for p in prompt.split(',') if not p.strip().startswith('<lora') and p.strip() != ''])
    prompt = ''.join([i for i in prompt if not i.isdigit()])
    return prompt


def check():
    tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-large-patch14')

    promptset = pd.read_csv('./promptset_v5.csv')
    for idx in promptset.index:
        prompt = promptset['prompt'][idx]
        negativePrompt = promptset['negativePrompt'][idx]

        promptset.loc[idx, 'prompt'] = truncate(prompt, tokenizer)
        promptset.loc[idx, 'negativePrompt'] = truncate(negativePrompt, tokenizer)
    
    promptset.to_csv('./promptset_v5.csv', index=False)


def truncate(prompt, tokenizer):
    if len(tokenizer(prompt).input_ids) > 77:
        prompt = prompt[:prompt.rfind(' ')]
        prompt = truncate(prompt, tokenizer)
    return prompt


def compare():
    v4 = pd.read_csv('./promptset_v4.csv')
    v5 = pd.read_csv('./promptset_v5.csv')

    for idx in v4.index:
        if v4['prompt'][idx] != v5['prompt'][idx]:
            print(v4['prompt'][idx])
            print(v5['prompt'][idx])
            print('\n')


if __name__ == '__main__':
   # create promptset_v5 csv
    promptset = pd.read_csv('./promptset_v4.csv')
    promptset.to_csv('./promptset_v5.csv', index=False)

    fetch_new_prompts(total=10, nsfw='None')

    check()
