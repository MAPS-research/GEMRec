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
import math
from torchmetrics.multimodal import CLIPScore
from PIL import Image
from torchvision import transforms
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns

# import sys
# sys.path.append(os.path.join(os.getcwd(), 'PIG-misc'))
# from evaluation import compute_metrics

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


def remove_image_not_in_meta():
    local_images = os.listdir('./generated/train')
    metadata = pd.read_csv('./generated/train/metadata.csv')
    for image in local_images:
        if image not in metadata['file_name'].tolist():
            os.system(f"rm -rf ./generated/train/{image}")


# util function
def count2idx(count, BINS):
    log_count = math.log10(count+1)
    for i in range(len(BINS-1)):
        # print(type(BINS[i]), type(BINS[i].item()), type(log_count))
        # print(log_count, 'success')
        if round(BINS[i].item(), 4) <= round(log_count, 4) <= round(BINS[i+1].item(), 4):
            return i
    # raise ValueError(f'count {count} not in bins')
    return count


def plot_distribution():
    # load bins
    with open('./PIG-misc/popularity/bins.pkl', 'rb') as f:
        BINS = pickle.load(f)
        print(f'\n==> bins: {BINS}')

    model_infos = os.listdir('/scratch/yg2709/ModelCoffer/everything/models')
    all_popularity_distribution = {}
    modelVersion_ids = []
    full_bins = {}
    for model_info in model_infos:
        with open(f'/scratch/yg2709/ModelCoffer/everything/models/{model_info}') as f:
            info = json.load(f)

        modelVersion_id = int(model_info.split('_')[1].split('.')[0])
        modelVersion_ids.append(modelVersion_id)
        popularity = info['stats']['downloadCount']
        log_pop = math.log(popularity)
        all_popularity_distribution[modelVersion_id] = log_pop
        bin = count2idx(popularity, BINS)
        full_bins[bin] = full_bins.get(bin, 0) + 1
    
    roster = pd.read_csv('./roster.csv')
    roster = roster[roster['modelVersion_id'].isin(modelVersion_ids)]

    # remove duplicate modelVersion_id
    roster.drop_duplicates(subset=['modelVersion_id'], inplace=True)

    roster.reset_index(drop=True, inplace=True)
    
    # alter download count in roster
    for idx in roster.index:
        modelVersion_id = roster.loc[idx, 'modelVersion_id']
        roster.loc[idx, 'model_download_count'] = all_popularity_distribution[modelVersion_id]
    

    sample_popularity_distribution = []
    sample_bis = {}
    for idx in roster.index:
        modelVersion_id = roster.loc[idx, 'modelVersion_id']
        # popularity = roster.loc[idx, 'model_download_count']
        # log_pop = math.log(popularity)
        log_pop = all_popularity_distribution[modelVersion_id]
        sample_popularity_distribution.append(log_pop)
        bin = count2idx(popularity, BINS)
        sample_bis[bin] = sample_bis.get(bin, 0) + 1
    
    # create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Popularity Distribution')
    
    # ax1.bar(full_bins.keys(), full_bins.values(), color='b')
    # ax1.set_title('Full Distribution')
    # ax2.bar(sample_bis.keys(), sample_bis.values(), color='r')
    # ax2.set_title('Sample Distribution')
    mi = min([a for id, a in all_popularity_distribution.items()])
    ma = max([a for id, a in all_popularity_distribution.items()])

    ax1.hist([a for id, a in all_popularity_distribution.items()], bins=12, range=[mi, ma], rwidth=0.8)
    ax1.set_title('Full Distribution')
    ax2.hist(sample_popularity_distribution, bins=12, range=[mi, ma], rwidth=0.8)
    ax2.set_title('Sample Distribution')

    plt.savefig('./popularity_distribution.png')


def test_clip_score():
    metadata = pd.read_csv('./generated/train/metadata.csv')
    metadata = metadata[:20]

    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").cuda()

    clip_score_hm_all = compute_clip_score(metadata)
    for idx in tqdm(range(len(metadata))):
        clip_score_hm = clip_score_hm_all[idx]
        # clip_score_hm = compute_clip_score(metadata[idx:idx+1])
        clip_score_current = metadata.iloc[idx]['clip_score']

        with torch.no_grad():
            image = Image.open(f"./generated/train/{metadata.loc[idx, 'file_name']}")
            image = transforms.ToTensor()(image)
            image = torch.unsqueeze(image, dim=0).cuda()
        
            prompts =[metadata.loc[idx, 'prompt']]
            clip_score = metric(image, prompts)

        print(f'==> clip_score_hm: {clip_score_hm}, clip_score: {clip_score}, clip_score_current: {clip_score_current}')

def compute_clip_score(promptbook, drop_negative=False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = len(promptbook)
    DATASET_DIR = './generated/train'
    # if 'clip_score' in promptbook.columns:
    #     print('==> Skipping clip_score computation')
    #     return
    clip_scores = []
    to_tensor = transforms.ToTensor()
    # metric = CLIPScore(model_name_or_path='openai/clip-vit-base-patch16').to(DEVICE)
    metric = CLIPScore(model_name_or_path='openai/clip-vit-large-patch14').to(DEVICE)
    for i in tqdm(range(0, len(promptbook), BATCH_SIZE)):
        images = []
        prompts = list(promptbook.prompt.values[i:i+BATCH_SIZE])
        for file_name in promptbook.file_name.values[i:i+BATCH_SIZE]:
            images.append(to_tensor(Image.open(os.path.join(DATASET_DIR, file_name))))
        with torch.no_grad():
            x = metric.processor(text=prompts, images=images, return_tensors='pt', padding=True)
            img_features = metric.model.get_image_features(x['pixel_values'].to(DEVICE))
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            txt_features = metric.model.get_text_features(x['input_ids'].to(DEVICE), x['attention_mask'].to(DEVICE))
            txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)
            scores = 100 * (img_features * txt_features).sum(axis=-1).detach().cpu()
        if drop_negative:
            scores = torch.max(scores, torch.zeros_like(scores))
        clip_scores += [round(s.item(), 4) for s in scores]
    return np.asarray(clip_scores)


def chech_tag_distribution():
    # stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english")

    stem = False

    sampled_roster = pd.read_csv('./roster.csv')

    if stem:
        sampled_roster.loc[:, 'tag'] = sampled_roster['tag'].apply(lambda x: stemmer.stem(x))
    tag_distribution = {}
    for idx in sampled_roster.index:
        tag = sampled_roster.loc[idx, 'tag']
        tag_distribution[tag] = tag_distribution.get(tag, 0) + 1
    tag_distribution = sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)
    
    palette1 = sns.color_palette("light:#79C", as_cmap=True)
    palette2 = sns.color_palette("light:#5A9", as_cmap=True)

    sns.set()

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'hspace': 5}, figsize=(12, 5))

    # ax1.pie([a for _, a in tag_distribution], labels=[t for t, _ in tag_distribution], autopct='%1.1f%%')
    ax2.pie([a for _, a in tag_distribution[:10]], labels=[t for t, _ in tag_distribution[:10]], autopct='%1.1f%%', startangle=90, colors=palette1(np.linspace(0, 1, 10)))
    ax2.set_title('our sampled subset')
    # ax1.legend()

    all_models = os.listdir('./everything/models')
    full_tag_distribution = {}
    for m in all_models:
        with open(f'./everything/models/{m}') as f:
            info = json.load(f)
        # print(info['tags'])
        for t in info['tags']:
            if stem:
                t = stemmer.stem(t)
            full_tag_distribution[t] = full_tag_distribution.get(t, 0) + 1
        
    full_tag_distribution = sorted(full_tag_distribution.items(), key=lambda x: x[1], reverse=True)
    
    ax1.pie([a for _, a in full_tag_distribution[:10]], labels=[t for t, _ in full_tag_distribution[:10]], autopct='%1.1f%%', startangle=90, colors=palette2(np.linspace(0, 1, 10)))
    ax1.set_title('full set of models from civitai')
    # ax2.legend()


    plt.savefig('./tag_distribution.png')


def check_promptset_v6():
    promptset = pd.read_csv('./promptsets/promptset_v6.csv')
    # get all possible 'note' datatype
    datatypes = []
    for idx in promptset.index:
        data = promptset.loc[idx, 'note']
        datatype = type(data)
        if datatype not in datatypes:
            datatypes.append(datatype)

    print(datatypes)


def add_version_to_roster():
    roster = pd.read_csv('./roster.csv')
    
    model_infos = os.listdir('./everything/models')
    model_infos = [m for m in model_infos if m.endswith('.json')]

    for idx in tqdm(roster.index):
        model_id = roster.loc[idx, 'model_id']
        modelVersion_id = roster.loc[idx, 'modelVersion_id']

        if modelVersion_id == 1000004:
            roster.loc[idx, 'baseModel'] = 'SD 1.4'
        elif modelVersion_id == 1000005:
            roster.loc[idx, 'baseModel'] = 'SD 1.5'
        elif modelVersion_id == 2000001:
            roster.loc[idx, 'baseModel'] = 'SD 2.1'
        else:
            model_info = [m for m in model_infos if m.startswith(f'{model_id}_{modelVersion_id}')][0]
            with open(f'./everything/models/{model_info}') as f:
                info = json.load(f)
            roster.loc[idx, 'baseModel'] = info['modelVersions'][0]['baseModel']
    
    roster.to_csv('./roster.csv', index=False)

def compute_clip_score_batch(promptbook, drop_negative=False):
    BATCH_SIZE = 200
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_DIR = '/scratch/yg2709/ModelCoffer/generated/train'

    clip_scores = []
    to_tensor = transforms.ToTensor()
    # metric = CLIPScore(model_name_or_path='openai/clip-vit-base-patch16').to(DEVICE)
    metric = CLIPScore(model_name_or_path='openai/clip-vit-large-patch14').to(DEVICE)
    for i in tqdm(range(0, len(promptbook), BATCH_SIZE)):
        images = []
        prompts = list(promptbook.prompt.values[i:i+BATCH_SIZE])
        for file_name in promptbook.file_name.values[i:i+BATCH_SIZE]:
            images.append(to_tensor(Image.open(os.path.join(DATASET_DIR, file_name))))
        with torch.no_grad():
            x = metric.processor(text=prompts, images=images, return_tensors='pt', padding=True)
            img_features = metric.model.get_image_features(x['pixel_values'].to(DEVICE))
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
            txt_features = metric.model.get_text_features(x['input_ids'].to(DEVICE), x['attention_mask'].to(DEVICE))
            txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)
            scores = 100 * (img_features * txt_features).sum(axis=-1).detach().cpu()
        if drop_negative:
            scores = torch.max(scores, torch.zeros_like(scores))
        clip_scores += [round(s.item(), 4) for s in scores]
    promptbook['clip_score'] = np.asarray(clip_scores)
    # promptbook.to_csv('./generated/train/metadata.csv', index=False)
    return promptbook

def compute_clip_score_iter(promptbook, drop_negative=False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_DIR = '/scratch/yg2709/ModelCoffer/generated/train'

    metric = CLIPScore(model_name_or_path='openai/clip-vit-large-patch14').to(DEVICE)
    for idx in tqdm(promptbook.index):
        row = promptbook.loc[idx, :]
        with torch.no_grad():
            image = Image.open(f"{DATASET_DIR}/{row['file_name']}")
            image = transforms.ToTensor()(image)
            image = torch.unsqueeze(image, dim=0).cuda()
        
            prompts =[row['prompt']]
            clip_score = round(float(metric(image, prompts).detach()), 4)
        promptbook.loc[idx, 'clip_score'] = clip_score
    return promptbook


def test_clip_calculation():
    # check torch version
    print(torch.__version__)

    promptbook_hmd = pd.read_csv('./generated/train/metadata.csv').sort_values(['prompt_id', 'modelVersion_id']).head(200)
    # print(len(promptbook))
    promptbook_new = promptbook_hmd.drop(columns=['clip_score'])
    promptbook_batch = compute_clip_score_batch(promptbook_new)
    promptbook_iter = compute_clip_score_iter(promptbook_new)

    for idx in promptbook_new.index:
        print('image id:', promptbook_batch.loc[idx, 'image_id'], 
              'rc (batch):', promptbook_batch.loc[idx, 'clip_score'], 
              'rc (iter):', promptbook_iter.loc[idx, 'clip_score'], 
              'hmd:', promptbook_hmd.loc[idx, 'clip_score']
              )



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
    # check_roster()
    # remove_models([5829, 5960])
    # clear_up_roster()
    # check_promptset_v3()
    # append_models_to_bin(4, 1)
    # append_models_to_bin(5, 1)
    # remove_civitai_images()
    # remove_image_not_in_meta()
    # plot_distribution()
    # test_clip_score()
    # chech_tag_distribution()
    # check_promptset_v6()
    # add_version_to_roster()
    test_clip_calculation()
