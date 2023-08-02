
import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoFeatureExtractor, CLIPProcessor, CLIPModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from utils.evaluation.NSFWScore import NSFWScore

BATCH_SIZE = 200
OUTPUT_DIR = '/scratch/hl3797/PIG-misc/evaluation'
ROSTER_PATH = '/scratch/yg2709/ModelCoffer/roster.csv'
DATASET_DIR = '/scratch/yg2709/ModelCoffer/generated/train'
METADATA_PATH = '/scratch/hl3797/PIG-misc/evaluation/metadata_base.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_clip_score(promptbook, resize_flag=True):
    if 'clip_score' in promptbook.columns:
        print('==> Skipping CLIP-Score computation')
        return
    print('==> CLIP-Score computation started')
    mcos_dict = {}
    mcos_scores = []
    clip_scores = []
    proc = transforms.Resize((224, 224)) if resize_flag else lambda x: x
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    for i in tqdm(range(0, len(promptbook), BATCH_SIZE)):
        images = []
        prompt_id = promptbook.prompt_id.values[i]
        prompts = list(promptbook.prompt.values[i:i+BATCH_SIZE])
        for file_name in promptbook.file_name.values[i:i+BATCH_SIZE]:
            images.append(proc(Image.open(os.path.join(DATASET_DIR, file_name))))
        with torch.no_grad():
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
            output = model(**{k:v.to(DEVICE) for k, v in inputs.items()})
            scores = output.logits_per_text[0].detach().cpu()
            mcos_dict[prompt_id] = F.cosine_similarity(
                x1=output.image_embeds.unsqueeze(1),
                x2=output.image_embeds.unsqueeze(0),
                dim=-1
            ).detach().cpu()
        mcos_scores += [round(s.item(), 4) for s in mcos_dict[prompt_id].mean(dim=1)]
        clip_scores += [round(s.item(), 4) for s in scores]
    promptbook['mcos_score'] = np.asarray(mcos_scores)
    promptbook['clip_score'] = np.asarray(clip_scores)
    torch.save(mcos_dict, os.path.join(OUTPUT_DIR, 'cosine_resize.pt'))
    print('==> CLIP-Score computation completed')

def compute_nsfw_score(promptbook):
    if 'nsfw_score' in promptbook.columns:
        print('==> Skipping NSFW-Score computation')
        return
    print('==> NSFW-Score computation started')
    nsfw_scores = []
    safety_model_id = 'CompVis/stable-diffusion-safety-checker'
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = NSFWScore(
        checker=StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    ).to(DEVICE)
    for i in tqdm(range(0, len(promptbook), BATCH_SIZE)):
        images = []
        for file_name in promptbook.file_name.values[i:i+BATCH_SIZE]:
            images.append(Image.open(os.path.join(DATASET_DIR, file_name)))
        feats = safety_feature_extractor(images, return_tensors='pt')
        scores = safety_checker(feats.pixel_values.to(DEVICE))
        nsfw_scores += [round(s.item(), 4) for s in scores]
    promptbook['nsfw_score'] = np.asarray(nsfw_scores)
    print('==> NSFW-Score computation completed')

def normalize_metrics(promptbook):
    print('==> Metrics normalization started')
    # compute normalized popularity (by model_download_count)
    norm_pop = []
    roster = pd.read_csv(ROSTER_PATH)
    for mvid in sorted(roster.modelVersion_id.unique()):
        count = roster[roster.modelVersion_id == mvid].model_download_count.values[0]
        norm_pop.append(math.log10(count+1))
    p_max, p_min = max(norm_pop), min(norm_pop)
    norm_pop = list(map(lambda p: round((p-p_min)/(p_max-p_min), 4), norm_pop)) * 90
    # compute normalized CLIP score
    norm_clip = []
    for i in range(0, len(promptbook), BATCH_SIZE):
        clip = list(promptbook.clip_score.values[i:i+BATCH_SIZE])
        s_max, s_min = max(clip), min(clip)
        norm_clip += list(map(lambda s: round((s-s_min)/(s_max-s_min), 4), clip))
    # compute normalized mCos score
    norm_mcos = []
    for i in range(0, len(promptbook), BATCH_SIZE):
        mcos = list(promptbook.mcos_score.values[i:i+BATCH_SIZE])
        s_max, s_min = max(mcos), min(mcos)
        norm_mcos += list(map(lambda s: round(1-(s-s_min)/(s_max-s_min), 4), mcos))
    # compute normalized NSFW score
    nsfw_scores = list(promptbook['nsfw_score'].values)
    s_min, s_max = min(nsfw_scores), max(nsfw_scores)
    norm_nsfw = list(map(lambda s: round((s-s_min)/(s_max-s_min), 4), nsfw_scores))
    # update promptbook
    promptbook['norm_clip'] = np.asarray(norm_clip)
    promptbook['norm_mcos'] = np.asarray(norm_mcos)
    promptbook['norm_nsfw'] = np.asarray(norm_nsfw)
    promptbook['norm_pop'] = np.asarray(norm_pop)
    print('==> Metrics normalization completed')

if __name__ == '__main__':

    # init promptbook
    promptbook = pd.read_csv(METADATA_PATH)
    promptbook = promptbook.fillna(np.nan).replace([np.nan], [None])
    promptbook = promptbook.sort_values(['prompt_id', 'modelVersion_id'])
    
    # compute metrics
    try:
        compute_nsfw_score(promptbook)
        compute_clip_score(promptbook)
        normalize_metrics(promptbook)
    except Exception as err:
        print('[Error]:', err)
        breakpoint()

    # dump promptbook
    promptbook = promptbook.sort_index()
    promptbook.to_csv(os.path.join(OUTPUT_DIR, 'metadata_resize.csv'), index=False)
