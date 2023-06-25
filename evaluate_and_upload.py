from datasets import Dataset, load_dataset
import datasets
from PIL import Image
from piq import inception_score, brisque
from torchvision import transforms
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import logging
import os

# build a dataloading script
# class ModelCofferPromptBookConfig(datasets.BuilderConfig):
    
#     def __init__(self, data_url, metadata_urls, **kwargs):
#         super().__init__(version=datasets.Version("1.0.0"), **kwargs)
#         self.data_url = data_url
#         self.metadata_urls = metadata_urls

# class ModelCofferPromptBook(datasets.GenerateBasedBuilder):

#     BUILDER_CONFIGS = [
#         ModelCofferPromptBookConfig(
#             name = "train",
#             description="Images generated with our own promptset",
#             data_url = "./generated/train"
#             metadata_urls={
#                 'train': "./generated/train/metadata.csv"
#             },
#         ),
#         ModelCofferPromptBookConfig(
#             name = "experiment",
#             description="Images generated with our own experiment promptset",
#             data_url = "./generated/experiment"
#             metadata_urls={
#                 'train': "./generated/experiment/metadata.csv"
#             },
#         ),
#     ]



def evaluate(promptbook, split='train'):
    assert split in ['train', 'val', 'test'], 'the split does not exist'

    # create a temp file to store results, in case the runtime failes halfway
    if os.path.exists(f'./metadata_temp_{split}.csv'):
        print(f'continue from {split} last time')
        promptbook = pd.read_csv(f'./metadata_temp_{split}.csv')
    else:
        print(f'start from a new promptbook for {split}')
        promptbook = pd.read_csv(f'./generated/{split}/metadata.csv')

    # replace nan in dataframe with None
    promptbook = promptbook.fillna(np.nan).replace([np.nan], [None])

    # add evaluation columns
    if 'brisque_score' not in promptbook.columns:
        promptbook['brisque_score'] = None
    if 'clip_score' not in promptbook.columns:
        promptbook['clip_score'] = None

    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    
    # evaluations
    for idx in tqdm(range(len(promptbook))):
        row = promptbook.loc[idx, :]

        image = Image.open(f"./generated/{split}/{row['file_name']}")
        prompts = [row['prompt']]
        image = transforms.ToTensor()(image)
        image = torch.unsqueeze(image, dim=0)

        if row['clip_score'] is None:
            c_score = round(float(clip_score_fn(image, prompts)), 4)
            promptbook.loc[idx, 'clip_score'] = c_score

        if row['brisque_score'] is None:
            b_score = round(float(brisque(image)), 4)
            promptbook.loc[idx, 'brisque_score'] = b_score

        # save a checker after 300 steps
        if idx % 300 == 0:
            promptbook.to_csv('./metadata_temp.csv', index=False)

    print(promptbook)
    
    # save evaluated promptbook
    promptbook.to_csv(f'./generated/{split}/metadata.csv', index=False)

    # remove temp file
    print('evaluation done, delete temporary file')
    os.system(f'rm -rf metadata_temp_{split}.csv')

    return promptbook


# def eva_and_push(split='train'):
    # # upload evaluated promptbook to hf dataset
    # dataset_promptbook = load_dataset('generated', data_dir='./', split=split)
    # dataset_promptbook.push_to_hub("NYUSHPRP/ModelCofferPromptBook", split=split)

    # # upload evaluated promptbook without images to hf dataset
    # # dataset_metadata = Dataset.from_pandas(promptbook)
    # dataset_metadata = dataset_promptbook.remove_columns('image')
    # dataset_metadata.push_to_hub("NYUSHPRP/ModelCofferMetadata", split=split)

if __name__ == '__main__':
    
    # subsets = ['train', 'experiment', 'test']
    # for sub in subsets:
    #     evaluate(sub)

    dataset_promptbook = load_dataset('imagefolder', data_dir='./generated')
    print(dataset_promptbook)
    dataset_promptbook.push_to_hub("NYUSHPRP/ModelCofferPromptBook")
    # experiment_promptbook = load_dataset('imagefolder', data_dir='./generated/experiment', split='train')
    # print(experiment_promptbook)

    # # a problem here, only train split is detected
    # dataset_promptbook = load_dataset('imagefolder', data_dir = './generated')
    # print(dataset_promptbook)
    # dataset_promptbook.push_to_hub("NYUSHPRP/ModelCofferPromptBook")

    # upload roster to hf dataset
    roster = pd.read_csv('./roster.csv')
    roster = Dataset.from_pandas(roster)
    print(roster)
    roster.push_to_hub("NYUSHPRP/ModelCofferRoster", split='train')
