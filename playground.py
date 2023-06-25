import pandas as pd
import numpy as np
import requests
import json
from pprint import pprint
import uuid
from datasets import Dataset, load_dataset
from imgsim.similarity import ImagePairSimilarity
from tqdm import tqdm
import torch

# roster = pd.read_csv('roster.csv')

# tag = 'girl'
# model_id = 123
# modelVersion_id = 7890

# print(((roster['tag'] == tag) & (roster['model_id'] == model_id) & (roster['modelVersion_id'] == modelVersion_id)).any())

def get_models(model_id):

    endpoint = 'https://civitai.com/api/v1/' + 'models/' + str(model_id)
    # get the requested results
    res = requests.get(endpoint)
    # items, metadata = res.json()['items'], res.json()['metadata']
    pprint(res.json())
    with open('./checkresponse{model_id}.json', 'w') as fp:
        json.dump(res.json(), fp)

def select_one_row():
    df = pd.read_csv('roster.csv')

    tag = 'character'
    model_id = 91002
    modelVersion_id = 96988

    row = df[(df['tag'] == tag) & (df['model_id'] == model_id) & (df['modelVersion_id'] == modelVersion_id)]
    row['model_download_count'] = 100
    print(row)


def test_prompt_set():
    parti_prompts = load_dataset('diffusers-parti-prompts/sd-v1-5', split='train')
    print(parti_prompts)

    df = pd.DataFrame(parti_prompts)

    test_prompts = df[df['Challenge']!='Basic'][['Prompt', 'Category', 'Challenge']]

    test_prompts.loc[:, 'Category'] = test_prompts['Category'].apply(lambda x: x.lower().strip('s'))

    selected_prompts = test_prompts.groupby('Category').apply(lambda x: x.sample(5)).reset_index(drop=True)
    print(selected_prompts)


def alter_metadata():
    df = pd.read_csv('./generated/train/metadata.csv')
    # df.loc[df['tag']=='female', 'tag'] = 'people'
    # df.loc[df['tag']=='male', 'tag'] = 'people'
    # df.loc[df['tag']=='fantasy', 'tag'] = 'illustration'
    # df.loc[df['prompt']=="masterpiece, best quality, best shadow, intricate,1 bowl of ramen", 'prompt'] = "masterpiece, best quality, best shadow, intricate, 1 bowl of ramen"
    # print(df.columns)
    df.loc[:, 'sampler'] = 'DPM++'
    print(df.loc[:, 'sampler'])
    df.to_csv('./generated/train/metadata.csv', index=False)

def alter_experiment_metadata():
    df = pd.read_csv('./generated/experiment/metadata.csv')
    df.loc[:, 'negativePrompt'] = ' '
    df.to_csv('./generated/experiment/metadata.csv', index=False)

def check_metadata():
    df = pd.read_csv('./generated/train/metadata.csv')
    df = df.fillna(np.nan).replace([np.nan], [None])

    print(df.iloc[1])
    print(df.iloc[1]['clip_score']==None)

    
def check_torch():
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())


def list_similarity():
    sim = ImagePairSimilarity()
    df = pd.read_csv('./generated/train/metadata.csv')
    selected = df[df['prompt_id'] == 1]
    # print(selected)

    images = selected['file_name']
    print(images)

    l = len(images)
    print(l, type(l))

    sim_matrix = np.zeros((len(images), len(images)))

    for (i, row) in enumerate(images):
        for (j, col) in enumerate(images):
            img1 = f'./generated/train/{row}'
            img2 = f'./generated/train/{col}'
            similarity = round(float(sim.cosine_similarity(img1, img2)), 4)
            sim_matrix[i, j] = similarity
            print(f'i: {i}, j: {j}, sim: {similarity}')
            
            
    sim_matrix = pd.DataFrame(sim_matrix, images.tolist(), images.tolist())
    print(sim_matrix)
    sim_matrix.to_csv('./imgsim/similiar_1.csv')
    
    # cos_sim = sim


def read_similarity_checkpoint():
    pt = "/scratch/hl3797/PIG-misc/similarity/cosine.pt"
    similarity_dic = torch.load(pt)
    print(similarity_dic)



if __name__ == "__main__":
    # get_models(88546)
    # id = uuid.uuid4().int
    # print(id, type(id))
    # select_one_row()
    # test_prompt_set()
    # alter_metadata()
    # check_metadata()
    # list_similarity()
    # read_similarity_checkpoint()
    alter_experiment_metadata()

