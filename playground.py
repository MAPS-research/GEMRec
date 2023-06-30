import pandas as pd
import numpy as np
import requests
import json
from pprint import pprint
import uuid
from datasets import Dataset, load_dataset
# from imgsim.similarity import ImagePairSimilarity
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import pickle
import os
import shutil
import re

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


def remove_embedding():
    promptbook = pd.read_csv('./generated/train/metadata.csv')
    promptbook = promptbook.drop(columns=['image_embedding', 'prompt_embedding'])
    promptbook.to_csv('./generated/train/metadata.csv', index=False)

def remove_index_col():
    df = pd.read_csv('./generated/train/metadata.csv')
    df = df.drop(columns='Unnamed: 0')
    print(df.columns)
    df.to_csv('./generated/train/metadata.csv', index=False)
    

def downloadCount_hist():
    df = pd.read_csv('./roster.csv')
    
def move_extra_models():
    DISTRIBUTION = '/scratch/hl3797/PIG-misc/popularity/subset.pkl'
    ALLMODELINFO = '/scratch/yg2709/ModelCoffer/everything/models'
    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)
    
    all_model_info = os.listdir(ALLMODELINFO)

    print('distribution loaded')
    print('invalid cache models: ', distribution['excep_list'])
    # print(len(distribution['candidate_dict']))

    BASE_DIR = '/scratch/yg2709/ModelCoffer'
    ARCHIVE_DIR = '/scratch/yg2709/ModelArchive'

    # move meta
    print('moving meta')
    for m in os.listdir(os.path.join(BASE_DIR, 'meta')):
        if int(m.strip('.txt')) in distribution['excep_list']:
            os.system(f"mv {os.path.join(BASE_DIR, 'meta', m)} {os.path.join(ARCHIVE_DIR, 'meta')}")


    # move download
    print('moving download')
    for m in os.listdir(os.path.join(BASE_DIR, 'download')):
        if int(m) in distribution['excep_list']:
            os.system(f"mv {os.path.join(BASE_DIR, 'download', m)} {os.path.join(ARCHIVE_DIR, 'download')}")

    # move ouput
    print('moving output')
    for m in os.listdir(os.path.join(BASE_DIR, 'output')):
        if int(m) in distribution['excep_list']:
            os.system(f"mv {os.path.join(BASE_DIR, 'output', m)} {os.path.join(ARCHIVE_DIR, 'output')}")


def move_unselected_models():
    BASE_DIR = '/scratch/yg2709/ModelCoffer'
    ARCHIVE_DIR = '/scratch/yg2709/ModelArchive'

    roster = pd.read_csv('./roster_error.csv')
    selected = roster['modelVersion_id'].unique().tolist()

    # move meta
    print('moving meta')
    for m in os.listdir(os.path.join(BASE_DIR, 'meta')):
        if int(m.strip('.txt')) not in selected:
            os.system(f"mv {os.path.join(BASE_DIR, 'meta', m)} {os.path.join(ARCHIVE_DIR, 'meta')}")
    
    # move download
    print('moving download')
    for m in os.listdir(os.path.join(BASE_DIR, 'download')):
        if int(m) not in selected:
            os.system(f"mv {os.path.join(BASE_DIR, 'download', m)} {os.path.join(ARCHIVE_DIR, 'download')}")
    
    # move output
    print('moving output')
    for m in os.listdir(os.path.join(BASE_DIR, 'output')):
        if int(m) not in selected:
            os.system(f"mv {os.path.join(BASE_DIR, 'output', m)} {os.path.join(ARCHIVE_DIR, 'output')}")


def check_roster():
    df = pd.read_csv('./roster.csv')
    items = df['modelVersion_id'].unique()
    print(len(items))

    local_files = os.listdir('/scratch/yg2709/ModelCoffer/output')
    print(len(local_files))

    images = pd.read_csv('./generated/train/metadata.csv')
    # images = images['modelVersion_id'].unique()
    
    # check whether all prompts are used to generate images
    print("==> Checking whether all prompts are used to generate images")
    # find the number of images for each model
    image_groups = images.groupby('modelVersion_id').count()
    # print modelVersion_id whose number of images is less than 80
    print(image_groups[image_groups['file_name'] < 80].index.tolist())


    metadatas = os.listdir('/scratch/yg2709/ModelCoffer/everything/models')

    DISTRIBUTION = '/scratch/hl3797/PIG-misc/popularity/subset.pkl'
    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)

    # check cache distribution
    print("==> Checking cache distribution")
    current_cache_distribution = {}
    for bin, models in distribution['selected_dict'].items():
        for item in items:
            if item in models:
                current_cache_distribution[bin] = current_cache_distribution.get(bin, []) + [item]

    # compare the two distributions:
    for bin, models in distribution['selected_dict'].items():
        try:
            if len(models) != len(current_cache_distribution[bin]):
                print('cache distribution not match')
                print(bin, len(models), len(current_cache_distribution[bin]))
                print('original: ', models)
                print('current: ', current_cache_distribution[bin])
        except:
            print(f'bin {bin} {models} not in current cache distribution')

    # check candidate distribution
    print("==> Checking candidate distribution")
    current_candidate_distribution = {}
    for bin, models in distribution['candidate_dict'].items():
        for item in items:
            if item in models:
                current_candidate_distribution[bin] = current_candidate_distribution.get(bin, []) + [item]
    
    # compare the current distribution with todo count
    for bin, todo_count in distribution['todo_count_dict'].items():
        try:
            if todo_count != len(current_candidate_distribution[bin]):
                print('candidate distribution not match')
                print(bin, todo_count, len(current_candidate_distribution[bin]))

        except:
            print(f'bin {bin} with {todo_count} not in current candidate distribution')

    # check model in roster outside of distribution
    print("==> Checking model in roster outside of distribution")
    models_in_current_distribution = []
    for bin, models in current_cache_distribution.items():
        models_in_current_distribution += models
    for bin, models in current_candidate_distribution.items():
        models_in_current_distribution += models
    
    for item in items:
        if item not in models_in_current_distribution:
            print(item, 'not in current distribution')
            # check whether it is in metadata
            for metadata in metadatas:
                if str(item)+'.json' == metadata.split('_')[1]:
                    print('but in metadata')
                    break

    # check current overall distribution in all bins
    print("==> Checking current overall distribution in all bins")
    current_overall_distribution = {}
    for bin in range(20):
        current_overall_distribution[bin] = current_cache_distribution.get(bin, []) + current_candidate_distribution.get(bin, [])
    print('current overall distribution: ', current_overall_distribution)
    print('current overall distribution length: ', [len(models) for models in current_overall_distribution.values()])

    # # check which bin a specific model is in
    # print("==> Checking which bin a specific model is in")
    # tobe_checked = [5829, 5960]
    # for item in tobe_checked:
    #     for bin, models in current_overall_distribution.items():
    #         if item in models:
    #             print(item, 'in bin', bin)


def append_models_to_bin(bin=10, model_num=1):
    DISTRIBUTION = '/scratch/hl3797/PIG-misc/popularity/subset.pkl'
    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)

    print(f'==> Appending {model_num} models to bin {bin}')
    # print('original todo: ', distribution['todo_count_dict'][bin])
    # print('original total: ', len(distribution['cache_dict'][bin])+distribution['todo_count_dict'][bin])

    roster = pd.read_csv('./roster.csv')
    metadatas = os.listdir('/scratch/yg2709/ModelCoffer/everything/models')

    # get the models in the bin
    candidate_models = distribution['candidate_dict'][bin]
    candidate_left = []
    for model in candidate_models:
        if model not in roster['modelVersion_id'].unique().tolist():
            candidate_left.append(model)
    
    # download the extra model
    for modelVersion_id in candidate_left:
        if model_num == 0:
            break

        model_id = [m.split('_')[0] for m in metadatas if int(m.split('_')[1].split('.')[0]) == modelVersion_id][0]
        print('==> Trying to download model', model_id, modelVersion_id, 'to bin', bin, 'with', model_num, 'left')
        try:
            os.system(f"python3 download_and_generate.py -f {model_id} {modelVersion_id}")
            model_num -= 1
        except EnvironmentError or ValueError:
            print(f'==> Model {model_id} {modelVersion_id} download failed, skip')
            os.system(f"rm -rf `find . -name '{modelVersion_id}*'`")
            # remove the model from roster
            remove_models([modelVersion_id])
        


def remove_models(modelVersion_ids=[92458, 92124, 94743, 94175]):
    df = pd.read_csv('./roster.csv')
    for modelVersion_id in modelVersion_ids:
        df = df[df['modelVersion_id'] != modelVersion_id]
    df.reset_index(drop=True, inplace=True)
    df.to_csv('./roster.csv', index=False)


def clear_up_roster():
    roster = pd.read_csv('./roster.csv')
    images = pd.read_csv('./generated/train/metadata.csv')

    # remove images not generated by the models in roster
    print('==> Removing images not generated by the models in roster')
    roster_ids = roster['modelVersion_id'].unique().tolist()
    for idx in images.index:
        if images.loc[idx, 'modelVersion_id'] not in roster_ids:
            
            print(images.loc[idx, 'modelVersion_id'], images.loc[idx, 'file_name'])

            os.system(f"rm -rf /scratch/yg2709/ModelCoffer/generated/train/{images.loc[idx, 'file_name']}")
            images.drop(idx, inplace=True)
            
    images.reset_index(drop=True, inplace=True)
    images.to_csv('./generated/train/metadata.csv', index=False)
    
    # check whether there are models in roster but not in images
    print('==> Checking models in roster but generate no images')
    tobe_generated = []
    images_ids = images['modelVersion_id'].unique().tolist()
    for idx in range(len(roster)):
        if roster.loc[idx, 'modelVersion_id'] not in images_ids:
            if roster.loc[idx, 'modelVersion_id'] not in tobe_generated:
                
                tobe_generated.append(roster.loc[idx, 'modelVersion_id'])
                
                model_id = roster.loc[idx, 'model_id']
                modelVersion_id = roster.loc[idx, 'modelVersion_id']
                print(f'model id: {model_id}, modelVersion id: {modelVersion_id} not in images')
                # generate images
                os.system(f"python3 download_and_generate.py -f {model_id} {modelVersion_id}")
            



def check_promptset_v3():
    promptset = pd.read_csv('./promptsets/promptset_v3.csv')
    prompts = promptset['prompt'].tolist()
    for prompt in prompts:
        # p_list = re.split(' |,', prompt)
        p_list = prompt.split(' ')
        p_list = [p for p in p_list if p != '']
        print(prompts.index(prompt), len(p_list))


def remove_civitai_images():
    images = pd.read_csv('./generated/train/metadata.csv')
    for idx in images.index:
        if images.loc[idx, 'tag'] == 'civitai' or images.loc[idx, 'prompt_id'] == '70':
            os.system(f"rm -rf /scratch/yg2709/ModelCoffer/generated/train/{images.loc[idx, 'file_name']}")
            images.drop(idx, inplace=True)
    images.reset_index(drop=True, inplace=True)
    images.to_csv('./generated/train/metadata.csv', index=False)



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
    # alter_experiment_metadata()
    # remove_embedding()
    # remove_index_col()
    # move_extra_models()
    # move_unselected_models()
    check_roster()
    # remove_models([5829, 5960])
    # clear_up_roster()
    # check_promptset_v3()
    # append_models_to_bin(4, 1)
    # append_models_to_bin(5, 1)
    # remove_civitai_images()

