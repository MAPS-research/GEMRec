import argparse
import logging
import os
import sys

import boto3
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from pathlib import Path
from tqdm import tqdm

# sys.path.append(os.path.join(os.getcwd(), 'PIG-misc'))
# from similarity import embedder


def upload_to_s3(bucket_name='modelcofferbucket', folder_path='./generated'):
    s3 = boto3.resource('s3')
    
    for path in tqdm(Path(folder_path).rglob('*.png')):
        if path.is_file():
            file_name = path.name
            s3.meta.client.upload_file(
                str(path), 
                bucket_name, 
                file_name, 
                ExtraArgs={"ACL": "public-read"}
            )
    
    print('Upload complete')


def set_s3_public(bucket_name='modelcofferbucket'):
    s3 = boto3.client('s3')
    bucket_name = 'modelcofferbucket'
    
    objects = s3.list_objects(Bucket=bucket_name)

    while True:
        for obj in tqdm(objects['Contents']):
            key = obj['Key']
            s3.put_object_acl(ACL='public-read', Bucket=bucket_name, Key=key)

        if objects['IsTruncated']:
            objects = s3.list_objects(
                Bucket=bucket_name, 
                Marker=objects['Contents'][-1]['Key']
            )
        else:
            break
        
    print('Set public complete')


if __name__ == '__main__':
    DATASET_DIR = '/scratch/yg2709/ModelCoffer/generated'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BUCKET = 'modelcofferbucket'
    splits = ['train']

    # upload generated images to s3
    upload_to_s3(bucket_name=BUCKET, folder_path=DATASET_DIR)
    set_s3_public(bucket_name=BUCKET)
    
    # upload generated images to hf dataset
    dataset_promptbook = load_dataset('imagefolder', data_dir=DATASET_DIR, split='train')
    dataset_promptbook.push_to_hub("NYUSHPRP/ModelCofferPromptBook")

    dataset_metadata = dataset_metadata = dataset_promptbook.remove_columns(['image'])
    dataset_metadata.push_to_hub("NYUSHPRP/ModelCofferMetadata")

    # upload roster to hf dataset
    roster = pd.read_csv('./roster.csv')
    roster = Dataset.from_pandas(roster)
    roster.push_to_hub("NYUSHPRP/ModelCofferRoster", split='train')

    print('Upload complete')
