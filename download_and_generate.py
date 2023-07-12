import requests
from pprint import pprint
from tqdm import tqdm
import json
import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
import uuid
import torch
import pickle

from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler

CIVITAI2DIFFUSERS_DIR = os.path.join(os.getcwd(), 'PIG-misc', 'civitai2diffusers')
DISTRIBUTION = '/scratch/hl3797/PIG-misc/popularity/subset.pkl'
ALLMODELINFO = '/scratch/yg2709/ModelCoffer/everything/models'

def fetch_new_models(promptbook, fetch_tag=None, types=None, sort="Most Downloaded", period="AllTime", nsfw='false', limit=100, start_page=None, pick_version='latest', generate=True, split='train'):

    endpoint = 'https://civitai.com/api/v1/' + 'models'

    # handel the query params
    if types is None:
        types = ["Checkpoint", "TextualInversion", "Hypernetwork", "AestheticGradient", "LORA", "Controlnet", "Poses"]

    params = {
        'limit': limit,
        'types': types,
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
    pprint(metadata)

    # loop through models in the results
    for idx in range(len(items)):
        model_info = items[idx]
        download_and_convert_diffusers(promptbook = promptbook, model_info = model_info, pick_version=pick_version, generate=generate, split=split)

    # recurse if there exists the next page
    if metadata['currentPage'] < metadata['totalPages']:
        fetch_new_models(promptbook = promptbook, fetch_tag=fetch_tag, types=types, sort=sort, period=period, nsfw=nsfw, limit=limit, start_page=metadata['currentPage']+1, pick_version=pick_version, generate=generate, split=split)


def fetch_specific_model(promptbook, model_id, modelVersion_id=None, generate=True, split='train'):
    endpoint = 'https://civitai.com/api/v1/' + f'models/{model_id}'
    model_info = requests.get(endpoint).json()
    download_and_convert_diffusers(promptbook=promptbook, model_info=model_info, pick_version=modelVersion_id, generate=generate, split=split)

    
def download_and_convert_diffusers(promptbook, model_info: dict, pick_version=None, generate=True, split='train'):
    eliminate_ids = [91218, 90245, 90365, 3666, 3627, 90406]
    # eliminate model 91218, which cause conversion error
    # eliminate model 90245, whose first file is actually lora
    # eliminate model 90365, which is an inpainting model and has different structure
    # eliminate model 3666, clip error during conversion
    # eliminate model 3627, conversion error (strange since it's a very popular model)
    # eliminate model 90406, wrong vae
    if 'mode' in model_info or model_info['id'] in eliminate_ids:  # mode is either archieve or taken down, which will lead to empty model page
        # print(f"model {model_info['id']} is either archieved or taken down, skip it")
        print(f"model is either archieved or taken down, skip it")
        raise ValueError()
    else:
        modelVersions = model_info['modelVersions']
        # skip early access (earlyAccessTimeFrame: 1 means in early access only)
        modelVersions = [v for v in modelVersions if v['earlyAccessTimeFrame']==0]
        tags = model_info['tags']
        print('tags: ', tags)
        if len(tags) == 0:
            print(f"model {model_info['id']} has no tags, skip it")
            raise ValueError()
        elif not isinstance(tags[0], str):
            print(f"tags is not a list of string, but {tags}")
            raise ValueError()

        if pick_version == 'latest':
            print('==> pick_version: ', pick_version)
            modelVersions = [modelVersions[0]]
        elif isinstance(pick_version, int):
            print('==> pick_version: ', pick_version)
            modelVersions = [v for v in modelVersions if v['id'] == pick_version]
            if len(modelVersions) == 0:
                print(f"Version {pick_version} is not found in model {model_info['id']}")
                raise ValueError()
            elif len(modelVersions) > 1:
                print(f"Version {pick_version} is found multiple times in model {model_info['id']}")
                raise ValueError()

        print(f"==> model {model_info['id']} has versions {[v['id'] for v in modelVersions]}")
        for version in modelVersions:

            # download and convert
            local_repo_id = f"./output/{version['id']}"
            modelVersion_id = version['id']
            if not os.path.exists(local_repo_id) or len(os.listdir(local_repo_id)) == 0:
                os.system(f"python3 {os.path.join(CIVITAI2DIFFUSERS_DIR, 'convert.py')} --model_version_id {modelVersion_id}")
            else:
                print(f"model {modelVersion_id} already exists, skip conversion")

            # record the model and modelVersion into roster csv if no error occurs
            if len(version['trainedWords']) > 0:
                trainedWords = [",".join(version['trainedWords'])]
            else:
                trainedWords = [""]
            
            roster = pd.read_csv('roster.csv')
            for tag in tags:
                if not ((roster['tag'] == tag) & (roster['model_id'] == model_info['id']) & (roster['modelVersion_id'] == version['id'])).any():
                    print('model not in roster, add it')
                    with open('roster.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            tag, model_info['name'], model_info['id'], version['name'], version['id'], version['downloadUrl'], trainedWords, model_info['stats']['downloadCount']
                            ])
                else:
                    print('model already registed in the roster, update it with latest download count')
                    roster[(roster['tag'] == tag) & (roster['model_id'] == model_info['id']) & (roster['modelVersion_id'] == version['id'])].loc[:, 'model_download_count'] = model_info['stats']['downloadCount']       

            # local_repo_id = f"./output/{version['id']}"
            # modelVersion_id = version['id']
            # if not os.path.exists(local_repo_id):
            #     os.system(f"python3 {os.path.join(CIVITAI2DIFFUSERS_DIR, 'convert.py')} --model_version_id {modelVersion_id}")
            # else:
            #     print(f"model {modelVersion_id} already exists, skip conversion")

            # generate images if necessary
            if generate:
                generate_images(promptbook=promptbook, model_id=model_info['id'], modelVersion_id=version['id'], repo_id=local_repo_id, split=split)  


def generate_single_image(pipeline, metadata, image_id, model_id, modelVersion_id, split='train'):
    width, height = metadata['size'].split('x')
    seed = int(metadata['seed'])
    generator = torch.Generator(device='cuda').manual_seed(seed)

    image = pipeline(
        prompt = metadata['prompt'],
        negative_prompt = metadata['negativePrompt'],
        height = int(height),
        width = int(width),
        num_inference_steps=30,
        guidance_scale = metadata['cfgScale'],
        generator = generator
        ).images[0]

    # save to metadata.csv
    with open(f'generated/{split}/metadata.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([f'{image_id}.png', image_id, metadata['tag'], model_id, modelVersion_id, metadata['prompt_id'], metadata['size'], metadata['seed'], metadata['prompt'], metadata['negativePrompt'], metadata['cfgScale'], 'DPM++ Karras'])

    return image


def generate_images(promptbook, model_id, modelVersion_id, repo_id, split='train'):

    # create image directory
    if not os.path.exists(f'generated/{split}'):
        os.makedirs(f'generated/{split}')
    
    # create metadata csv
    if not os.path.exists(f'generated/{split}/metadata.csv'):
        with open(f'generated/{split}/metadata.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'image_id', 'tag', 'model_id', 'modelVersion_id', 'prompt_id', 'size', 'seed', 'prompt', 'negativePrompt', 'cfgScale', 'sampler'])

    pipeline = DiffusionPipeline.from_pretrained(repo_id, safety_checker = None, custom_pipeline="lpw_stable_diffusion")

    # set custom scheduler (sampler)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras=True)
    pipeline.to("cuda")
    # print(pipeline.scheduler)
    # print("scheduler now being used: ", pipeline.scheduler.config._class_name) 

    for prompt_idx in range(len(promptbook)):
        metadata = promptbook.iloc[[prompt_idx]].squeeze()

        generated = pd.read_csv(f'generated/{split}/metadata.csv')

        # generate if no duplicate exists
        if not ((generated['prompt_id']==metadata['prompt_id']) & (generated['model_id']==model_id) & (generated['modelVersion_id']==modelVersion_id)).any():
            image_id = uuid.uuid4().int
            image = generate_single_image(pipeline, metadata, image_id, model_id, modelVersion_id, split)
            image.save(f'generated/{split}/{image_id}.png')
        else:
            print('image already generated, skip!')


def remove_images(modelVersion_id, split="train"):
    # function for both remove and regenerate images
    generated = pd.read_csv(f'generated/{split}/metadata.csv')
    past_generations = generated[generated['modelVersion_id']==modelVersion_id]
    try:
        model_id = past_generations['model_id'].unique()[0]

        print(f'found generated images my model {model_id} version {modelVersion_id} in the past')
        # remove generated images in local storage
        for (idx, past) in past_generations.iterrows():
            # print(idx, past['file_name'])
            local_image = os.path.join(os.getcwd(), 'generated', split, past['file_name'])
            os.remove(local_image)
            generated.drop(index=idx, inplace=True)
            print(f"image {past['file_name']} is removed")

        generated.to_csv(f'./generated/{split}/metadata.csv', index=False)
        return model_id, modelVersion_id

    except:
        print(f'no image generated by model version {modelVersion_id} is found')
        return None, modelVersion_id
    


def regenerate_images(promptbook, modelVersion_id, split="train"):
    model_id, modelVersion_id = remove_images(modelVersion_id, split)

    # generate new images
    repo_id = f"./output/{modelVersion_id}"
    generate_images(promptbook, model_id, modelVersion_id, repo_id, split)


def popularity_distribution(promptbook, loop_cache=True, loop_candidate=True, split='train'):
    # get models based on popularity distribution
    print('==> Getting models based on popularity distribution')

    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)
    
    all_model_info = os.listdir(ALLMODELINFO)

    print('distribution loaded')
    print('selected cached models: ', distribution['selected_dict'])
    print('candidate models: ', distribution['candidate_dict'])
    print('subset of distribution: ', distribution['dist_sub'])
    print('num of candidates needed: ', distribution['todo_count_dict'])
    print(len(distribution['candidate_dict']))
    
    if loop_cache:

        total_cache = sum([len(modelVersion_ids) for bin, modelVersion_ids in distribution['selected_dict'].items()])
        cache_pbar = tqdm(total=total_cache, desc='looping through models in cache')

        # loop throught models in cache
        for bin, modelVersion_ids in distribution['selected_dict'].items():
            print(bin, modelVersion_ids)
            for modelVersion_id in modelVersion_ids:
                # convert numpy int64 to int
                if isinstance(modelVersion_id, np.int64):
                    modelVersion_id = modelVersion_id.item()
                model_id = [info.split('_')[0] for info in all_model_info if str(modelVersion_id)+'.json'==info.split('_')[1]][0]
                cache_pbar.set_description(f'generate with model {model_id} version {modelVersion_id} in bin {bin}')
                # print(f'type of model_id: {type(model_id)}, type of modelVersion_id: {type(modelVersion_id)}')
                fetch_specific_model(promptbook = promptbook, model_id = model_id, modelVersion_id = modelVersion_id, generate=True, split=split)
                cache_pbar.update(1)
    
    if loop_candidate:
        
        # loop through models in candidate
        popularity_distribution_candidate_download(promptbook=promptbook, bins=[b for b in range(len(distribution['candidate_dict']))], generate=True, split=split)


def popularity_distribution_candidate_download(promptbook, bins, generate=False, split='train'):
    print('==> Downloading candidate models based on popularity distribution')
    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)
    all_model_info = os.listdir(ALLMODELINFO)

    total_todo = sum([distribution['todo_count_dict'][bin] for bin in bins])
    pbar = tqdm(total=total_todo, desc=f'looping through models in candidate from bin {bins[0]} to {bins[-1]}')
    for bin in bins:
        modelVersion_ids = distribution['candidate_dict'][bin]
        todo = distribution['todo_count_dict'][bin]

        print(bin, modelVersion_ids)

        append_models_from_candidates(promptbook=promptbook, pbar=pbar, candidate_ids=modelVersion_ids, model_num=todo, generate=generate, split=split)

def append_models_from_candidates(promptbook, pbar, candidate_ids, model_num, generate, split='train'):
    
    for modelVersion_id in candidate_ids:
        # convert numpy int64 to int
        if isinstance(modelVersion_id, np.int64):
            modelVersion_id = modelVersion_id.item()

        # break if all models needed in this bin are generated
        if model_num == 0:
            break
        
        # find corresponding model_id
        try:
            model_id = [info.split('_')[0] for info in all_model_info if str(modelVersion_id)+'.json'==info.split('_')[1]][0]
        except IndexError:
            print("modelVersion_id not found in all_model_info, skip")

        try:
            fetch_specific_model(promptbook = promptbook, model_id = int(model_id), modelVersion_id = int(modelVersion_id), generate=generate, split=split)
            model_num -= 1
            pbar.update(1)
            pbar.set_description(f'cope with model {model_id} version {modelVersion_id} in bin {bin}, with {model_num} todo models left')
        except ValueError:
            print(f"Error in Model {model_id} version {modelVersion_id}")
            remove_images(modelVersion_id, split=split)
        except EnvironmentError:
            print(f"Local files for model {model_id} version {modelVersion_id} might be corrupted, remove them and try again")
            os.system(f"rm -rf `find . -name {modelVersion_id}`")
            try:
                print(f"Try to fetch model {model_id} version {modelVersion_id} again")
                fetch_specific_model(promptbook = promptbook, model_id = int(model_id), modelVersion_id = int(modelVersion_id), generate=generate, split=split)
                model_num -= 1
                pbar.update(1)
                pbar.set_description(f'cope with model {model_id} version {modelVersion_id} in bin {bin}, with {todo} todo models left')
            except EnvironmentError:
                print(f"Error in Model {model_id} version {modelVersion_id}, skip it")
                remove_images(modelVersion_id, split=split)
                # remove model from roster
                roster = pd.read_csv('roster.csv')
                roster = roster[roster['modelVersion_id']!=modelVersion_id]
                roster.to_csv('roster.csv', index=False)
                print(f"Model {model_id} version {modelVersion_id} removed from roster")

def append_models_to_bin(promptbook, bin, model_num, generate=True, split=args.split):
    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)

    all_model_info = os.listdir(ALLMODELINFO)
    roster = pd.read_csv('./roster.csv')

    print(f'==> Appending {model_num} models to bin {bin}')

    # get the models in the bin
    candidate_models = distribution['candidate_dict'][bin]
    candidate_left = []
    for model in candidate_models:
        if model not in roster['modelVersion_id'].unique().tolist():
            candidate_left.append(model)

    # quick fix for model 49998_54532
    if 54532 in candidate_left:
        candidate_left.remove(54532)

    pbar = tqdm(total=model_num, desc=f'Appending {model_num} models to bin {bin}')

    append_models_from_candidates(promptbook, pbar, candidate_left, model_num, generate, split)
    

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-rm", "--remove", default=None, type=int, help="Type in the model version id to remove all images it generates")
    parser.add_argument("-rg", "--regenerate", default=None, type=int, help="Type in the model version id to remove all images it generates, and then generate with it again")
    parser.add_argument("-gb", "--gen_with_base_sd", action="store_true", default=False, help="Generate images with original stable diffusion")
    parser.add_argument("-lr", "--loop_through_roster", action="store_true", default=False, help="Generate image with models in the roster")
    parser.add_argument("-fn", "--fetch_new_models", action="store_true", default=False, help="Generate images by fetching from civitai")
    parser.add_argument("-f", "--fetch_specific_model", default=[None, None], nargs=2, help="Type in model id and model version id to fetch it from civitai")
    parser.add_argument("-s", "--split", default="train", type=str, help="Determine what promptset to use and which split to save generated images")
    parser.add_argument("-p", "--popularity_distribution", default=None, type=str, help="Generate images with models in the popularity distribution, 'all' for both cache and candidate, 'cache' for cache only, 'candidate' for candidate only")
    parser.add_argument("-pd", "--popularity_distribution_candidate_download", default=None, nargs='+', type=int, help="Download candidate models in the popularity distribution, type in the bins you want to download")
    parser.add_argument("-ab", "--append_models_to_bin", default=[None, None], nargs=2, type=int, help="Append models to a bin, type in the bin number and the number of models you want to append")
    parser.add_argument("-rp", "--replace_models", default=None, nargs='+', type=int, help="Replace models in the popularity distribution, type in the modelVersion ids you want to replace")

    args = parser.parse_args()

    # create the roster csv
    if not os.path.exists('roster.csv'):
        with open('roster.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'tag', 'model_name', 'model_id', 'modelVersion_name', 'modelVersion_id', 'modelVersion_url', 'modelVersion_trainedWords', 'model_download_count'
            ])

    # load the promptbook base one split
    if args.split == 'train':
        promptbook = pd.read_csv('./promptsets/promptset_v5.csv')  # a global variable
    elif args.split == 'val':
        promptbook = pd.read_csv('./promptsets/promptset_e1.csv')
        promptbook = promptbook[promptbook['tag']=='abstract']
        promptbook['negativePrompt'] = None
        print('experiment promptset is loaded')
    elif args.split == 'test':
        promptbook == pd.read_csv('./promptsets/promptset_c1.csv')
    else:
        raise Exception("Split can only be train, val or test")

    if args.remove is not None:
        remove_images(args.remove, args.split)

    if args.regenerate is not None:
        regenerate_images(promptbook, args.regenerate, args.split)

    if args.gen_with_base_sd:
        # generate images with original stable diffusion
        print('Generate images with original stable diffusion')
        sd_repos = [{'model_id': 1000000, 'modelVersion_id': 1000004, 'repo_id':'CompVis/stable-diffusion-v1-4'},
                    {'model_id': 1000000, 'modelVersion_id': 1000005, 'repo_id':'runwayml/stable-diffusion-v1-5'},
                    {'model_id': 2000000, 'modelVersion_id': 2000001, 'repo_id':'stabilityai/stable-diffusion-2-1'},
        ]
        for sd_repo in tqdm(sd_repos):
            generate_images(promptbook=promptbook, model_id = sd_repo['model_id'], modelVersion_id = sd_repo['modelVersion_id'], repo_id = sd_repo['repo_id'], split=args.split)

        print('Original sd generation done')


    if args.loop_through_roster:
        # generate image with models in the roster
        roster = pd.read_csv('./roster.csv')[3:]  # skip the first 3 original stable diffusions

        models = roster[['model_id', 'modelVersion_id']].drop_duplicates(ignore_index=True)
        for idx in tqdm(range(len(models)), desc='Generating with models in the roster'):
            model_data = models.iloc[idx]
            model_id = model_data['model_id']
            modelVersion_id = model_data['modelVersion_id']
            repo_id = f"./output/{modelVersion_id}"
            generate_images(promptbook=promptbook, model_id=model_id, modelVersion_id=modelVersion_id, repo_id=repo_id, split=args.split)


    if args.fetch_new_models:
        # generate images by fetching from civitai    
        fetch_new_models(promptbook=promptbook, types=["Checkpoint"], sort="Most Downloaded", pick_version='latest', generate=True, split = args.split)

    if args.fetch_specific_model[0] is not None and args.fetch_specific_model[0] != 'None':
        # argparse automatically turns input type into string, while the real input could be either int, str, or None
        # the following code fixes this issue
        model_id = int(args.fetch_specific_model[0])
        modelVersion_id = args.fetch_specific_model[1]
        if modelVersion_id != 'None':
            try:
                modelVersion_id = int(modelVersion_id)
            except:
                pass
        else:
            modelVersion_id = None

        print('fetching ', 'model id: ', model_id, 'modelVersion_id: ', modelVersion_id)
        fetch_specific_model(promptbook=promptbook, model_id=model_id, modelVersion_id=modelVersion_id, generate=True, split=args.split)

    if args.popularity_distribution is not None:
        assert args.popularity_distribution in ['all', 'cache', 'candidate'], "popularity_distribution can only be 'all', 'cache', or 'candidate'"
        if args.popularity_distribution == 'all':
            popularity_distribution(promptbook, loop_cache=True, loop_candidate=True, split=args.split)
        elif args.popularity_distribution == 'cache':
            popularity_distribution(promptbook, loop_cache=True, loop_candidate=False, split=args.split)
        elif args.popularity_distribution == 'candidate':
            popularity_distribution(promptbook, loop_cache=False, loop_candidate=True, split=args.split)
    
    if args.popularity_distribution_candidate_download is not None:
        bins = args.popularity_distribution_candidate_download
        popularity_distribution_candidate_download(promptbook, bins, generate=False, split=args.split)

    if args.append_models_to_bin[0] is not None and args.append_models_to_bin[1] is not None:
        bin = args.append_models_to_bin[0]
        model_num = args.append_models_to_bin[1]

        append_models_to_bin(promptbook, bin, model_num, generate=True, split=args.split)

    if args.replace_models:
        print('to be replaced:', args.replace_models, type(args.replace_models))
        with open(DISTRIBUTION, 'rb') as f:
            distribution = pickle.load(f)

        for model in tqdm(args.replace_models):
            in_bin = None
            # find the bin for the model to be replaced
            # search for cache_dict
            for bin in distribution['cache_dict']:
                if model in distribution['cache_dict'][bin]:
                    in_bin = bin
                    break
            
            # search for candidate_dict
            if not in_bin:
                for bin in distribution['candidate_dict']:
                    if model in distribution['candidate_dict'][bin]:
                        in_bin = bin
                        break
            
            # if the model is not in cache or candidate, skip
            if not in_bin:
                print(f'==> Model {model} is not in cache or candidate, skip')
            
            # add one model in the corresponding bin from candidate
            append_models_to_bin(promptbook, in_bin, 1, generate=True, split=args.split)

            # remove images generated by the model
            remove_images(model, args.split)
            print(f'==>Images generated by model {model} is removed from {args.split} set')

            # remove the model from roster
            roster = pd.read_csv('./roster.csv')
            roster = roster[roster['modelVersion_id'] != model]
            roster.to_csv('./roster.csv', index=False)
            print(f'==>Model {model} is removed from roster')

