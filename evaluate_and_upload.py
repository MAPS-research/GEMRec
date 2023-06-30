from datasets import Dataset, load_dataset
import datasets
from PIL import Image
from piq import inception_score, brisque
from torchvision import transforms
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from torchmetrics.multimodal import CLIPScore
from functools import partial
import logging
import os
import sys
import argparse

sys.path.append(os.path.join(os.getcwd(), 'PIG-misc', 'similarity'))
import embedder

def evaluate(split='train', brisque=False, clip=True):
    assert split in ['train', 'val', 'test'], 'the split does not exist'

    # create a temp file to store results, in case the runtime failes halfway
    if os.path.exists(f'./metadata_temp_{split}.csv'):
        print(f'==> Continue evaluating from {split} last time')
        promptbook = pd.read_csv(f'./metadata_temp_{split}.csv')
    else:
        print(f'==> Evaluation starts from a new promptbook for {split}')
        promptbook = pd.read_csv(f'./generated/{split}/metadata.csv')

    # replace nan in dataframe with None
    promptbook = promptbook.fillna(np.nan).replace([np.nan], [None])

    # add evaluation columns if not exist
    if brisque:
        if 'brisque_score' not in promptbook.columns:
            promptbook['brisque_score'] = None
        
    if clip:
        if 'clip_score' not in promptbook.columns:
            promptbook['clip_score'] = None


    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    
    # evaluations
    for idx in tqdm(range(len(promptbook))):
        row = promptbook.loc[idx, :]

        if brisque:
        
            # calculate and store brisque score
            if row['brisque_score'] is None:
                b_score = round(float(brisque(image)), 4)
                promptbook.loc[idx, 'brisque_score'] = b_score

        if clip:
            # calculate and store clip score
            if row['clip_score'] is None:
                with torch.no_grad():
                    image = Image.open(f"./generated/{split}/{row['file_name']}")
                    image = transforms.ToTensor()(image)
                    image = torch.unsqueeze(image, dim=0)

                    prompts = [row['prompt']]
                    original_prompt = row['prompt']
                    
                    # handle prompt length limit
                    exceptions = {}
                    if prompts[0] in exceptions.keys():
                        prompts = [exceptions[prompts]]
                    loop = True
                    while loop:
                        try:
                            c_score = round(float(metric(image, prompts).detach()), 4)
                            promptbook.loc[idx, 'clip_score'] = c_score
                            loop = False
                            if len(prompts[0]) < len(original_prompt):
                                print(f'==> CLIP score calculated for {original_prompt} with length{len(original_prompt)}, truncated to {prompts[0]} with length {len(prompts[0])}')
                                exceptions = {original_prompt: prompts[0]}
                                print('==> Add to exceptions')

                        except:
                            logging.warning(f'==> CLIP score calculation failed for {row["prompt"]}, truncate and retry...')
                            prompts = [row['prompt'][:-1]]
                            
        # save a checker after 300 steps
        if idx % 300 == 0:
            promptbook.to_csv(f'./metadata_temp_{split}.csv', index=False)

    # print(promptbook)
    
    # save evaluated promptbook
    promptbook.to_csv(f'./generated/{split}/metadata.csv', index=False)

    # remove temp file
    print('==> Evaluation done, delete temporary file')
    os.system(f'rm -rf metadata_temp_{split}.csv')

    return promptbook

def embed(split='train'):

    embedder.DUMP_DIR = os.path.join(os.getcwd(), 'feats', split)
    embedder.ROOT_DIR = os.path.join(os.getcwd(), 'generated', split)

    if os.path.exists(embedder.DUMP_DIR):
        print(f'==> Embedding checkpoints saved half way are found. Continue with {embedder.DUMP_DIR}')
    else:
        print(f'==> Embedding starts from the beginning')

    emb = embedder.Embedder()
    promptbook = emb.metadata
    # add embedding columns
    if 'image_embedding' not in promptbook.columns:
        promptbook['image_embedding'] = None
    if 'prompt_embedding' not in promptbook.columns:
        promptbook['prompt_embedding'] = None
    
    emb.embed_images()
    emb.embed_prompts()

    weights = os.listdir(embedder.DUMP_DIR)
    weights = [w for w in weights if w[:-3].isdigit()]
    # print(weights)

    prompt_emb_dict = torch.load(os.path.join(embedder.DUMP_DIR, 'prompts.pt'))
    # print(prompt_emb_dict.keys())

    print(f'==> Adding embeddings to dataset')
    for weight in tqdm(weights):
        prompt_id = int(weight[:-3])
        
        # print(type(prompt_emb_dict[1]), prompt_emb_dict[1].shape)
        # promptbook[promptbook['prompt_id']==prompt_id]['prompt_embedding'] = prompt_emb_dict[prompt_id]

        image_emb_dict = torch.load(os.path.join(embedder.DUMP_DIR, weight))
        # print(image_emb_dict.keys())

        modelVersion_ids = [k for k in image_emb_dict.keys() if k != 'all']
        # print(modelVersion_ids)
        # print(prompt_emb_dict[prompt_id].detach().cpu().numpy().flatten().shape)

        for modelVersion_id in modelVersion_ids:
            idx = promptbook.loc[((promptbook['prompt_id']==prompt_id) & (promptbook['modelVersion_id']==modelVersion_id))].index
            # print(image_emb_dict[modelVersion_id].shape)

            if promptbook.loc[idx, 'prompt_embedding'] is None or promptbook.loc[idx, 'image_embedding'] is None:
                promptbook.loc[idx, 'prompt_embedding'] = [prompt_emb_dict[prompt_id].detach().cpu().numpy()]
                promptbook.loc[idx, 'image_embedding'] = [image_emb_dict[modelVersion_id].detach().cpu().numpy()]

    print(type(promptbook.loc[0, 'prompt_embedding']), promptbook.loc[0, 'prompt_embedding'])
    promptbook.to_csv(f'./generated/{split}/metadata.csv', index=False)
        

# def eva_and_push(split='train'):
    # # upload evaluated promptbook to hf dataset
    # dataset_promptbook = load_dataset('generated', data_dir='./', split=split)
    # dataset_promptbook.push_to_hub("NYUSHPRP/ModelCofferPromptBook", split=split)

    # # upload evaluated promptbook without images to hf dataset
    # # dataset_metadata = Dataset.from_pandas(promptbook)
    # dataset_metadata = dataset_promptbook.remove_columns('image')
    # dataset_metadata.push_to_hub("NYUSHPRP/ModelCofferMetadata", split=split)

if __name__ == '__main__':
    
    splits = ['train']
    for split in splits:
        # evaluate(split = split)
        # embed(split)
        pass

    dataset_promptbook = load_dataset('imagefolder', data_dir='./generated', split='train')
    print(dataset_promptbook)

    dataset_metadata = dataset_metadata = dataset_promptbook.remove_columns(['image'])
    print(dataset_metadata)

    dataset_promptbook.push_to_hub("NYUSHPRP/ModelCofferPromptBook")
    dataset_metadata.push_to_hub("NYUSHPRP/ModelCofferMetadata")


    # upload roster to hf dataset
    roster = pd.read_csv('./roster.csv')
    roster = Dataset.from_pandas(roster)
    print(roster)
    roster.push_to_hub("NYUSHPRP/ModelCofferRoster", split='train')
