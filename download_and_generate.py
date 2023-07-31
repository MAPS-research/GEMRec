import argparse
import csv
import json
import logging
import os
import pickle
import uuid
import warnings

import requests
import torch
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from controlnet_aux import OpenposeDetector
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from pprint import pprint
from tqdm import tqdm

def fetch_new_models(promptbook, fetch_tag=None, types=None, sort="Most Downloaded", period="AllTime", nsfw='false', limit=100, start_page=None, pick_version='latest', generate=True):

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
        download_and_convert_diffusers(promptbook = promptbook, model_info = model_info, pick_version=pick_version, generate=generate)

    # recurse if there exists the next page
    if metadata['currentPage'] < metadata['totalPages']:
        fetch_new_models(promptbook = promptbook, fetch_tag=fetch_tag, types=types, sort=sort, period=period, nsfw=nsfw, limit=limit, start_page=metadata['currentPage']+1, pick_version=pick_version, generate=generate)


def fetch_specific_model(promptbook, model_id, modelVersion_id=None, generate=True):
    endpoint = 'https://civitai.com/api/v1/' + f'models/{model_id}'
    model_info = requests.get(endpoint).json()
    download_and_convert_diffusers(promptbook=promptbook, model_info=model_info, pick_version=modelVersion_id, generate=generate)

    
def download_and_convert_diffusers(promptbook, model_info: dict, pick_version=None, generate=True):
    eliminate_ids = [91218, 90245, 90365, 3666, 3627, 90406, 67363]
    # eliminate model 91218, which cause conversion error
    # eliminate model 90245, whose first file is actually lora
    # eliminate model 90365, which is an inpainting model and has different structure
    # eliminate model 3666, clip error during conversion
    # eliminate model 3627, conversion error (strange since it's a very popular model)
    # eliminate model 90406, wrong vae
    # eliminate model 67363, tagged futa
    if 'mode' in model_info or model_info['id'] in eliminate_ids:  # mode is either archieve or taken down, which will lead to empty model page
        # print(f"model {model_info['id']} is either archieved or taken down, skip it")
        logger.warning(f"** Model is either archieved or taken down, skip it")
        raise ValueError()
    else:
        modelVersions = model_info['modelVersions']
        # skip early access (earlyAccessTimeFrame: 1 means in early access only)
        modelVersions = [v for v in modelVersions if v['earlyAccessTimeFrame']==0]
        tags = model_info['tags']
        print('tags: ', tags)
        if len(tags) == 0:
            logger.warning(f"** Model {model_info['id']} has no tags, skip it")
            raise ValueError()
        elif not isinstance(tags[0], str):
            logger.warning(f"** Tags is not a list of string, but {tags}")
            raise ValueError()

        if pick_version == 'latest':
            modelVersions = [modelVersions[0]]
        elif isinstance(pick_version, int):
            modelVersions = [v for v in modelVersions if v['id'] == pick_version]
            if len(modelVersions) == 0:
                logger.warning(f"** Version {pick_version} is not found in model {model_info['id']}")
                raise ValueError()
            elif len(modelVersions) > 1:
                logger.warning(f"** Version {pick_version} is found multiple times in model {model_info['id']}")
                raise ValueError()

        logger.info(f"=> Picking version(s) {[v['id'] for v in modelVersions]} of model {model_info['id']}")
        for version in modelVersions:

            # download and convert
            local_repo_id = os.path.join(MODEL_DIR, version['id'])
            modelVersion_id = version['id']
            if not os.path.exists(local_repo_id) or len(os.listdir(local_repo_id)) == 0:
                os.system(f"python3 {os.path.join(CIVITAI2DIFFUSERS_DIR, 'convert.py')} --model_version_id {modelVersion_id}")
            else:
                logger.info(f"## Model {modelVersion_id} already exists, skip conversion")

            # record the model and modelVersion into roster csv if no error occurs
            if len(version['trainedWords']) > 0:
                trainedWords = [",".join(version['trainedWords'])]
            else:
                trainedWords = [""]
            
            roster = pd.read_csv(ROSTER)
            for tag in tags:
                if not ((roster['tag'] == tag) & (roster['model_id'] == model_info['id']) & (roster['modelVersion_id'] == version['id'])).any():
                    logger.info('## Model not in roster, add it')
                    with open(ROSTER, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            tag, model_info['name'], model_info['id'], version['name'], version['id'], version['downloadUrl'], trainedWords, model_info['stats']['downloadCount'], version['baseModel']
                            ])
                else:
                    logger.info('## Model already registed in the roster, update it with latest download count')
                    roster[(roster['tag'] == tag) & (roster['model_id'] == model_info['id']) & (roster['modelVersion_id'] == version['id'])].loc[:, 'model_download_count'] = model_info['stats']['downloadCount']       

            # generate images if necessary
            if generate:
                generate_images(promptbook=promptbook, model_id=model_info['id'], modelVersion_id=version['id'], baseModel=version['baseModel'], repo_id=local_repo_id)  


def generate_single_image(pipeline, metadata, image_id, model_id, modelVersion_id):
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
    with open(os.path.join(DUMP_DIR, 'metadata.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([f'{image_id}.png', image_id, metadata['tag'], model_id, modelVersion_id, metadata['prompt_id'], metadata['size'], metadata['seed'], metadata['prompt'], metadata['negativePrompt'], metadata['cfgScale'], metadata['sampler'], metadata['note']])

    return image

def generate_in_batch(pipeline, promptset, image_ids, model_id, modelVersion_id, ref_imgs=None):
    

    width, height = promptset['size'].unique()[0].split('x')
    seeds = promptset['seed'].tolist()
    generators = [torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]

    pipeline.set_progress_bar_config(leave=False, desc=f"==> Generating images of size {width}x{height}")

    # logger.warning("** All cfgScale are set to 7 for now")
    # logger.info(f"** Image size: {width}x{height}")

    if ref_imgs is None:
        images = pipeline(
            prompt = promptset['prompt'].tolist(),
            negative_prompt = promptset['negativePrompt'].tolist(),
            height = int(height),
            width = int(width),
            num_inference_steps=30,
            guidance_scale = 7,
            generator = generators
            ).images

    else:
        images = pipeline(
            prompt = promptset['prompt'].tolist(),
            image = ref_imgs,
            negative_prompt = promptset['negativePrompt'].tolist(),
            height = int(height),
            width = int(width),
            num_inference_steps=30,
            guidance_scale = 7,
            generator = generators
            ).images

    
    # save to metadata.csv
    with open(os.path.join(DUMP_DIR, 'metadata.csv'), 'a') as f:
        writer = csv.writer(f)
        for i, image_id in enumerate(image_ids):
            writer.writerow([f'{image_id}.png', image_id, promptset['tag'].tolist()[i], model_id, modelVersion_id, promptset['prompt_id'].tolist()[i], promptset['size'].tolist()[i], promptset['seed'].tolist()[i], promptset['prompt'].tolist()[i], promptset['negativePrompt'].tolist()[i], promptset['cfgScale'].tolist()[i], promptset['sampler'].tolist()[i], promptset['note'].tolist()[i]])
    
    return images


def generate_images(promptbook, model_id, modelVersion_id, baseModel, repo_id):

    # create image directory
    if not os.path.exists(DUMP_DIR):
        os.makedirs(DUMP_DIR)
    
    # create metadata csv
    if not os.path.exists(os.path.join(DUMP_DIR, 'metadata.csv')):
        with open(os.path.join(DUMP_DIR, 'metadata.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'image_id', 'tag', 'model_id', 'modelVersion_id', 'prompt_id', 'size', 'seed', 'prompt', 'negativePrompt', 'cfgScale', 'sampler', 'note'])

    # remove already generated prompts
    generated = pd.read_csv(os.path.join(DUMP_DIR, 'metadata.csv'))
    promptset = promptbook[~promptbook['prompt_id'].isin(generated[generated['modelVersion_id']==modelVersion_id]['prompt_id'])]


    # logger.info(f"=> Generating {len(promptset)} new images, {len(promptbook)-len(promptset)} images already generated")

    if len(promptset) == 0:
        return
    
    # temporary fix for future warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        # load pipeline
        pipeline = DiffusionPipeline.from_pretrained(repo_id, safety_checker = None, requires_safety_checker=False, custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16)
        # set custom scheduler (sampler)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

        # tricks for better performance
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()

    if CONTROLNET:

        # temporary fix for future warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

            if baseModel.startswith('SD 2'):
                controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-openpose-diffusers", torch_dtype=torch.float16)
            else:
                controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            ctrl_pipeline = StableDiffusionControlNetPipeline.from_pretrained(repo_id, safety_checker = None, requires_safety_checker=False, controlnet=controlnet, torch_dtype=torch.float16)

            ctrl_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

            # tricks for better performance
            ctrl_pipeline.enable_xformers_memory_efficient_attention()
            ctrl_pipeline.enable_model_cpu_offload()
            ctrl_pipeline.enable_vae_slicing()


    # seperate promptset by size
    sizes = promptset['size'].unique()

    pbar = tqdm(total=len(promptset), leave=False, desc=f"=> Generating {len(promptset)} new images, {len(promptbook)-len(promptset)} images already generated")
    for size in sizes:
        subset = promptset[promptset['size']==size]
        
        if CONTROLNET:
            os.makedirs(CACHE_DIR, exist_ok=True)

            # pick metadata with note as int only
            civitaiset = subset[subset['note'].apply(lambda x: isinstance(x, str) and x.isdigit())]

            if len(civitaiset) > 0:
                # exclude civitaiset from subset
                subset = subset[~subset['prompt_id'].isin(civitaiset['prompt_id'])]

                ref_imgs = []
                # download reference image from civitai
                for idx in civitaiset.index:
                    civitai_id = civitaiset.loc[idx, 'note']
                    # check if image already downloaded
                    if not os.path.exists(os.path.join(CACHE_DIR, f"civitai_{civitai_id}_pose.png")):
                        res = requests.get(f'https://civitai.com/images/{civitai_id}')
                        assert res.status_code == 200
                        soup = BeautifulSoup(res.text, 'html.parser')
                        image_section = soup.find('div', {'class': 'mantine-12rlksp'})
                        image_url = image_section.find('img')['src']
                        image = load_image(image_url)
                        pose = openpose(image)
                        image.save(os.path.join(CACHE_DIR, f"civitai_{civitai_id}.png"))
                        pose.save(os.path.join(CACHE_DIR, f"civitai_{civitai_id}_pose.png"))
                        ref_imgs.append(pose)
                    else:
                        ref_imgs.append(load_image(os.path.join(CACHE_DIR, f"civitai_{civitai_id}_pose.png")))
            
                image_ids = [uuid.uuid4().int for _ in range(len(civitaiset))]
                images = generate_in_batch(ctrl_pipeline, civitaiset, image_ids, model_id, modelVersion_id, ref_imgs)
                for i, image in enumerate(images):
                    image.save(os.path.join(DUMP_DIR, f'{image_ids[i]}.png'))
          
                pbar.update(len(civitaiset))
        
        if len(subset) > 0:
            image_ids = [uuid.uuid4().int for _ in range(len(subset))]
            images = generate_in_batch(pipeline, subset, image_ids, model_id, modelVersion_id, None)
            for i, image in enumerate(images):
                image.save(os.path.join(DUMP_DIR, f'{image_ids[i]}.png'))

            pbar.update(len(subset))


def remove_images(modelVersion_id, remove_from_roster=False):
    # function for both remove and regenerate images
    generated = pd.read_csv(os.path.join(DUMP_DIR, 'metadata.csv'))
    past_generations = generated[generated['modelVersion_id']==modelVersion_id]

    # remove model from roster
    if remove_from_roster:
        roster = pd.read_csv(ROSTER)
        roster = roster[roster['modelVersion_id']!=modelVersion_id]
        roster.to_csv(ROSTER, index=False)
        logger.info(f"## Model version {modelVersion_id} no longer in roster")

    try:
        model_id = past_generations['model_id'].unique()[0]
        base_model = past_generations[past_generations['model_id']==model_id]['base_model'].unique()[0]

        # remove generated images in local storage
        for (idx, past) in past_generations.iterrows():
            # print(idx, past['file_name'])
            local_image = os.path.join(DUMP_DIR, past['file_name'])
            os.remove(local_image)
            generated.drop(index=idx, inplace=True)
        
        logger.info(f"## {len(past_generations)} images generated by model version {modelVersion_id} are removed")

        generated.to_csv(os.path.join(DUMP_DIR, 'metadata.csv'), index=False)

        return model_id, modelVersion_id, base_model

    except:
        logger.info(f'## No image generated by model version {modelVersion_id} is found')
        return None, modelVersion_id, None


def regenerate_images(promptbook, modelVersion_id):
    model_id, modelVersion_id, baseModel = remove_images(modelVersion_id, remove_from_roster=False)

    if model_id is not None:
        # generate new images
        repo_id = os.path.join(MODEL_DIR, modelVersion_id)
        generate_images(promptbook, model_id, modelVersion_id, baseModel, repo_id)


def popularity_distribution(promptbook, loop_cache=True, loop_candidate=True):
    # get models based on popularity distribution
    logger.info('=> Getting models based on popularity distribution')

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
                fetch_specific_model(promptbook = promptbook, model_id = model_id, modelVersion_id = modelVersion_id, generate=True,)
                cache_pbar.update(1)
    
    if loop_candidate:
        
        # loop through models in candidate
        popularity_distribution_candidate_download(promptbook=promptbook, bins=[b for b in range(len(distribution['candidate_dict']))], generate=True)


def popularity_distribution_candidate_download(promptbook, bins, generate=False):
    logger.info('=> Downloading candidate models based on popularity distribution')
    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)
    all_model_info = os.listdir(ALLMODELINFO)

    total_todo = sum([distribution['todo_count_dict'][bin] for bin in bins])
    pbar = tqdm(total=total_todo, desc=f'looping through models in candidate from bin {bins[0]} to {bins[-1]}')
    for bin in bins:
        modelVersion_ids = distribution['candidate_dict'][bin]
        todo = distribution['todo_count_dict'][bin]

        print(bin, modelVersion_ids)

        append_models_from_candidates(promptbook=promptbook, all_model_info=all_model_info, pbar=pbar, candidate_ids=modelVersion_ids, model_num=todo, generate=generate)

def append_models_from_candidates(promptbook, all_model_info, pbar, candidate_ids, model_num, generate):
    
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

        # give the model n chances to download
        chances = 2
        for chance in range(chances):
            try:
                fetch_specific_model(promptbook = promptbook, model_id = int(model_id), modelVersion_id = int(modelVersion_id), generate=generate)
                model_num -= 1
                pbar.update(1)
                pbar.set_description(f'cope with model {model_id} version {modelVersion_id} in bin {bin}, with {model_num} todo models left')
                break
            except (ValueError, EnvironmentError) as e:
                if chance < chances-1:
                    logger.warning(f"** {e} in Model {model_id} version {modelVersion_id}, remove local files and try again")
                    
                else:
                    logger.warning(f"** {e} in Model {model_id} version {modelVersion_id}, skip it")

                os.system(f"rm -rf `find . -name {modelVersion_id}`")
                remove_images(modelVersion_id, remove_from_roster=True)
            
                # try:
                #     print(f"Try to fetch model {model_id} version {modelVersion_id} again")
                #     fetch_specific_model(promptbook = promptbook, model_id = int(model_id), modelVersion_id = int(modelVersion_id), generate=generate)
                #     model_num -= 1
                #     pbar.update(1)
                #     pbar.set_description(f'cope with model {model_id} version {modelVersion_id} in bin {bin}, with {model_num} todo models left')
                # except EnvironmentError:
                #     print(f"Error in Model {model_id} version {modelVersion_id}, skip it")
                #     remove_images(modelVersion_id, remove_from_roster=True)


def append_models_to_bin(promptbook, bin, model_num, generate=True):
    with open(DISTRIBUTION, 'rb') as f:
        distribution = pickle.load(f)

    all_model_info = os.listdir(ALLMODELINFO)
    roster = pd.read_csv(ROSTER)

    logger.info(f'=> Appending {model_num} models to bin {bin}')

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

    append_models_from_candidates(promptbook, all_model_info, pbar, candidate_left, model_num, generate)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s - %(levelname)s - %(asctime)s ")
    logger = logging.getLogger(__name__)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", default="train", type=str, help="Determine what promptset to use and which split to save generated images")
    parser.add_argument("-c", "--controlnet", action="store_false", default=True, help="Use controlnet when generating images with prompts from civitai")

    parser.add_argument("-fn", "--fetch_new_models", action="store_true", default=False, help="Generate images by fetching from civitai")
    parser.add_argument("-f", "--fetch_specific_model", default=[None, None], nargs=2, help="Type in model id and model version id to fetch it from civitai")

    parser.add_argument("-p", "--popularity_distribution", default=None, type=str, help="Generate images with models in the popularity distribution, 'all' for both cache and candidate, 'cache' for cache only, 'candidate' for candidate only")
    parser.add_argument("-pd", "--popularity_distribution_candidate_download", default=None, nargs='+', type=int, help="Download and convert candidate models in the popularity distribution without generating images, type in the bins you want to download")
    parser.add_argument("-ab", "--append_models_to_bin", default=[None, None], nargs=2, type=int, help="Append models to a bin, type in the bin number and the number of models you want to append")
    parser.add_argument("-rp", "--replace_models", default=None, nargs='+', type=int, help="Replace models in the popularity distribution, type in the modelVersion ids you want to replace")

    parser.add_argument("-sd", "--gen_with_base_sd", action="store_true", default=False, help="Generate images with original stable diffusion")
    parser.add_argument("-lr", "--loop_through_roster", action="store_true", default=False, help="Generate image with models in the roster")

    parser.add_argument("-rm", "--remove", default=None, type=int, help="Type in the model version id to remove all images it generates")
    parser.add_argument("-rg", "--regenerate", default=None, type=int, help="Type in the model version id to remove all images it generates, and then generate with it again")

    args = parser.parse_args()

    CIVITAI2DIFFUSERS_DIR = os.path.join(os.getcwd(), 'PIG-misc', 'civitai2diffusers')
    CACHE_DIR = os.path.join(os.getcwd(), 'tmp')
    MODEL_DIR = os.path.join(os.getcwd(), 'output')
    DISTRIBUTION = '/scratch/hl3797/PIG-misc/popularity/subset.pkl'
    ALLMODELINFO = '/scratch/yg2709/ModelCoffer/everything/models'
    ROSTER = '/scratch/yg2709/ModelCoffer/roster.csv'

    CONTROLNET = args.controlnet

    # load the promptbook base one split
    SPLIT = args.split
    if SPLIT == 'train':
        promptbook = pd.read_csv('./promptsets/promptset_v6.csv')  # a global variable

    # TODO: add other splits

    else:
        raise Exception("Split can only be train")

    DUMP_DIR = os.path.join(os.getcwd(), 'generated', SPLIT)


    # create the roster csv
    if not os.path.exists(ROSTER):
        with open(ROSTER, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'tag', 'model_name', 'model_id', 'modelVersion_name', 'modelVersion_id', 'modelVersion_url', 'modelVersion_trainedWords', 'model_download_count', 'baseModel'
            ])

    if args.remove is not None:
        remove_images(args.remove, remove_from_roster=True)

    if args.regenerate is not None:
        regenerate_images(promptbook, args.regenerate)

    if args.gen_with_base_sd:
        # generate images with original stable diffusion
        sd_repos = [{'model_id': 1000000, 'modelVersion_id': 1000004, 'repo_id':'CompVis/stable-diffusion-v1-4', 'baseModel': 'SD 1.4'},
                    {'model_id': 1000000, 'modelVersion_id': 1000005, 'repo_id':'runwayml/stable-diffusion-v1-5', 'baseModel': 'SD 1.5'},
                    {'model_id': 2000000, 'modelVersion_id': 2000001, 'repo_id':'stabilityai/stable-diffusion-2-1', 'baseModel': 'SD 2.1'},
        ]
        
        for sd_repo in tqdm(sd_repos, desc='Generating with original stable diffusion'):
            generate_images(promptbook=promptbook, model_id = sd_repo['model_id'], modelVersion_id = sd_repo['modelVersion_id'], baseModel= sd_repo['baseModel'], repo_id = sd_repo['repo_id'])


    if args.loop_through_roster:
        # generate image with models in the roster
        roster = pd.read_csv(ROSTER)[3:]  # skip the first 3 original stable diffusions

        models = roster[['model_id', 'modelVersion_id', 'baseModel']].drop_duplicates(ignore_index=True)
        for idx in tqdm(range(len(models)), desc='Generating with models in the roster'):
            model_data = models.iloc[idx]
            model_id = model_data['model_id']
            modelVersion_id = model_data['modelVersion_id']
            baseModel = model_data['baseModel']
            repo_id = os.path.join(MODEL_DIR, modelVersion_id)
            generate_images(promptbook=promptbook, model_id=model_id, modelVersion_id=modelVersion_id, baseModel=baseModel, repo_id=repo_id)


    if args.fetch_new_models:
        # generate images by fetching from civitai    
        fetch_new_models(promptbook=promptbook, types=["Checkpoint"], sort="Most Downloaded", pick_version='latest', generate=True)

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

        logger.debug('## Fetching model id: ', model_id, 'modelVersion_id: ', modelVersion_id)
        fetch_specific_model(promptbook=promptbook, model_id=model_id, modelVersion_id=modelVersion_id, generate=True)

    if args.popularity_distribution is not None:
        assert args.popularity_distribution in ['all', 'cache', 'candidate'], "popularity_distribution can only be 'all', 'cache', or 'candidate'"
        if args.popularity_distribution == 'all':
            popularity_distribution(promptbook, loop_cache=True, loop_candidate=True)
        elif args.popularity_distribution == 'cache':
            popularity_distribution(promptbook, loop_cache=True, loop_candidate=False)
        elif args.popularity_distribution == 'candidate':
            popularity_distribution(promptbook, loop_cache=False, loop_candidate=True)
    
    if args.popularity_distribution_candidate_download is not None:
        bins = args.popularity_distribution_candidate_download
        popularity_distribution_candidate_download(promptbook, bins, generate=False)

    if args.append_models_to_bin[0] is not None and args.append_models_to_bin[1] is not None:
        bin = args.append_models_to_bin[0]
        model_num = args.append_models_to_bin[1]

        append_models_to_bin(promptbook, bin, model_num, generate=True)

    if args.replace_models:
        # check if the model exists in roster
        roster = pd.read_csv(ROSTER)[3:]  # skip the first 3 original stable diffusions
        roster_models = roster[['modelVersion_id']].drop_duplicates(ignore_index=True)
        tobe_replaced = []
        for model in args.replace_models:
            if int(model) in roster_models['modelVersion_id'].tolist():
                tobe_replaced.append(model)

        if len(tobe_replaced) == 0:
            logger.debug('## No model in the roster to be replaced')
            exit()

        with open(DISTRIBUTION, 'rb') as f:
            distribution = pickle.load(f)

        for model in tqdm(tobe_replaced):
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
                logger.warning(f'** Model {model} is not in cache or candidate, skip')
            
            else:
                # add one model in the corresponding bin from candidate
                append_models_to_bin(promptbook, in_bin, 1, generate=True)

                # remove images generated by the model
                remove_images(model, remove_from_roster=True)

