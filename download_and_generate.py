import requests
from pprint import pprint
from tqdm import tqdm
import json
import argparse
import os
import pandas as pd
from matplotlib import pyplot as plt
import csv
import uuid
import torch

from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler

CIVITAI2DIFFUSERS_DIR = os.path.join(os.getcwd(), 'PIG-misc', 'civitai2diffusers')

def fetch_new_models(fetch_tag=None, types=None, sort="Most Downloaded", period="AllTime", nsfw='false', limit=100, start_page=None, pick_version='latest', generate=True, split='train'):

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
        download_and_convert_diffusers(model_info = model_info, pick_version=pick_version, generate=generate, split=split)

    if page == 'all' and metadata['currentPage'] < metadata['totalPages']:
        fetch_new_models(fetch_tag=fetch_tag, types=types, sort=sort, period=period, nsfw=nsfw, limit=limit, start_page=metadata['currentPage']+1, pick_version=pick_version, generate=generate, split=split)


def fetch_specific_model(model_id, modelVersion_id=None, generate=True, split='train'):
    endpoint = 'https://civitai.com/api/v1/' + f'models/{model_id}'
    model_info = requests.get(endpoint).json()
    download_and_convert_diffusers(model_info=model_info, pick_version=modelVersion_id, generate=True, split=split)

    
def download_and_convert_diffusers(model_info: dict, pick_version=None, generate=True, split='train'):
    eliminate_ids = [91218, 90245, 90365, 3666, 3627]
    # eliminate model 91218, which cause conversion error
    # eliminate model 90245, whose first file is actually lora
    # eliminate model 90365, which is an inpainting model and has different structure
    # eliminate model 3666, clip error during conversion
    # eliminate model 3627, conversion error (strange since it's a very popular model)
    if not 'mode' in model_info and model_info['id'] not in eliminate_ids:  # mode is either archieve or taken down, which will lead to empty model page

        modelVersions = model_info['modelVersions']
        # skip early access (earlyAccessTimeFrame: 1 means in early access only)
        modelVersions = [v for v in modelVersions if v['earlyAccessTimeFrame']==0]
        tags = model_info['tags']
        print('tags: ', tags)
        assert isinstance(tags[0], str)

        if pick_version == 'latest':
            modelVersions = [modelVersions[0]]
        elif isinstance(pick_version, int):
            modelVersions = [v for v in modelVersions if v['id'] == pick_version]
            # raise error is 
            assert len(modelVersions) == 0, f"Version {pick_version} is not found in model {model_info['id']}"

        for version in modelVersions:
            if len(version['trainedWords']) > 0:
                trainedWords = [",".join(version['trainedWords'])]
            else:
                trainedWords = [""]
            
            # record the model and modelVersion into roster csv
            roster = pd.read_csv('roster.csv')
            for tag in tags:
                if not ((roster['tag'] == tag) & (roster['model_id'] == model_info['id']) & (roster['modelVersion_id'] == version['id'])).any():
                    with open('roster.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            tag, model_info['name'], model_info['id'], version['name'], version['id'], version['downloadUrl'], trainedWords, model_info['stats']['downloadCount']
                            ])
                else:
                    print('model already registed in the roster, update it with latest download count')
                    roster[(roster['tag'] == tag) & (roster['model_id'] == model_info['id']) & (roster['modelVersion_id'] == version['id'])].loc[:, 'model_download_count'] = model_info['stats']['downloadCount']       

            local_repo_id = f"./output/{version['id']}"
            modelVersion_id = version['id']
            if not os.path.exists(local_repo_id):
                os.system(f'python3 {os.path.join(CIVITAI2DIFFUSERS_DIR, 'convert.py')} --model_version_id {modelVersion_id}')
            else:
                print(f"model {modelVersion_id} already exists, skip conversion")

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
        writer.writerow([f'{image_id}.png', image_id, metadata['tag'], model_id, modelVersion_id, metadata['prompt_id'], metadata['size'], metadata['seed'], metadata['prompt'], metadata['negativePrompt'], metadata['cfgScale'], pipeline.scheduler.config._class_name])

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
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")
    print(pipeline.scheduler)
    print("scheduler now being used: ", pipeline.scheduler.config._class_name) 

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


def remove_images(promptbook, modelVersion_id, split="train"):
    # function for both remove and regenerate images
    generated = pd.read_csv(f'generated/{split}/metadata.csv')
    past_generations = generated[generated['modelVersion_id']==modelVersion_id]
    model_id = past_generations['model_id'].unique()[0]
    if len(past_generations) <= 0:
        print('no image generated by model {model_id} version {model_Version_id} is found')
        
    else:
        print('found generated images my model {model_id} version {model_Version_id} in the past')
        # remove generated images in local storage
        for (idx, past) in past_generations.iterrows():
            # print(idx, past['file_name'])
            local_image = os.path.join(os.getcwd(), 'generated', split, past['file_name'])
            os.remove(local_image)
            generated.drop(index=idx, inplace=True)
            print(f"image {past['file_name']} is removed")
    
    generated.to_csv(f'./generated/{split}/metadata.csv', index=False)

    return model_id, modelVersion_id


def regenerate_images(promptbook, modelVersion_id, split="train"):
    model_id, modelVersion_id = remove_images(promptbook, modelVersion_id, split)

    # generate new images
    repo_id = f"./output/{modelVersion_id}"
    generate_images(promptbook, model_id, modelVersion_id, repo_id, split)


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
        promptbook = pd.read_csv('./promptsets/promptset_v2.csv')  # a global variable
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
        remove_images(promptbook, args.remove, args.split)

    if args.regenerate is not None:
        regenerate_images(promptbook, args.regenerate, args.split)

    if args.gen_with_base_sd:
        # generate images with original stable diffusion
        print('Generate images with original stable diffusion')
        sd_repos = [{'model_id': 1, 'modelVersion_id': 4, 'repo_id':'CompVis/stable-diffusion-v1-4'},
                    {'model_id': 1, 'modelVersion_id': 5, 'repo_id':'runwayml/stable-diffusion-v1-5'},
                    {'model_id': 2, 'modelVersion_id': 1, 'repo_id':'stabilityai/stable-diffusion-2-1'},
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
        fetch_new_models(types=["Checkpoint"], sort="Most Downloaded", pick_version='latest', generate=True, split = args.split)

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
        fetch_specific_model(model_id, modelVersion_id, args.split)