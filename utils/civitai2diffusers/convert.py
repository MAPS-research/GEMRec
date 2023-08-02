""" Download & convert civitai checkpoints to diffusers format. """

import os
import sys
import json
import torch
import argparse
import requests

from tqdm import tqdm
from pt_to_diffusers import vae_pt_to_vae_diffuser
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

SAFE_MODE = True
DOWNLOAD_CHECK = False

def download_file(url, save_path):
    if DOWNLOAD_CHECK:
        download_flag = input(f'\nDownload from: {url}? (y/n): ').strip()
        if download_flag != 'y':
            print('\n==> Download aborted')
            sys.exit(1)
    else:
        print(f'\nDownloading from: {url}')
    try:
        response = requests.get(url, stream=True)
        assert response.status_code == 200, 'Abnormal status code.'
        file_size = int(response.headers.get('Content-Length', 0))
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
    except Exception as err:
        print('\n[Download Error]:', err)
        sys.exit(1)

def map_config_file(base_model):
    if 'SD2' not in base_model:
        return './v1-inference.yaml'
    return './v2-inference-v.yaml' if '768' in base_model else './v2-inference.yaml'

def convert_single(args):

    # init directories
    assert os.path.exists('./v1-inference.yaml')
    assert os.path.exists('./v2-inference.yaml')
    assert os.path.exists('./v2-inference-v.yaml')
    os.makedirs(f'./download/{args.model_version_id}', exist_ok=True)
    os.makedirs(f'./output/{args.model_version_id}', exist_ok=True)
    os.makedirs(f'./meta', exist_ok=True)

    # fetch meta data
    meta_path = f'./meta/{args.model_version_id}.txt'
    if not os.path.exists(meta_path):
        try:
            url = f'https://civitai.com/api/v1/model-versions/{args.model_version_id}'
            print(f'\n==> Fetching meta data from: {url}')
            response = requests.get(url)
            # print(response.text)
            assert response.status_code == 200, 'Abnormal status code.'
            meta_data = json.loads(response.text)
            with open(meta_path, 'w') as f:
                f.write(response.text)
        except Exception as err:
            print('[Error]:', err)
            sys.exit(1)
    else:
        print(f'\n==> Found meta data at: {meta_path}')
        with open(meta_path, 'r') as f:
            meta_data = json.loads(f.read())
    # print(f'Meta Data: {meta_data.keys()}')

    # check the files to be downloaded
    vae_info, model_info = None, None
    for file_info in meta_data['files']:
        if file_info['primary']:
            model_info = file_info
        if file_info['type'] == 'VAE':
            vae_info = file_info
    model_path = os.path.join('./download', args.model_version_id, model_info['name'])
    if vae_info is not None:
        vae_path = os.path.join('./download', args.model_version_id, vae_info['name'])

    # check download cache & download the model (if needed)
    while not (os.path.exists(model_path) and (int(os.stat(model_path).st_size / 1024) == int(model_info['sizeKB']))):
        try:
            download_file(url=model_info['downloadUrl'], save_path=model_path)
        except Exception as err:
            print('[Download Error]:', err)
    if vae_info is not None:
        while not (os.path.exists(vae_path) and (int(os.stat(vae_path).st_size / 1024) == int(vae_info['sizeKB']))):
            try:
                download_file(url=vae_info['downloadUrl'], save_path=vae_path) 
            except Exception as err:
                print('[Download Error]:', err)

    # complement the weights (if possible)
    try:
        if (meta_data['baseModel'] == 'SD 1.5') and model_path.endswith('safetensors'):
            print('\n==> Complementing weights (SD 1.5 - safetensors)')
            from safetensors import safe_open
            from safetensors.torch import save_file
            sd_ckpt, model_ckpt = {}, {}
            # !wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
            with safe_open('./v1-5-pruned-emaonly.safetensors', framework='pt', device='cpu') as f:
                for key in f.keys():
                    sd_ckpt[key] = f.get_tensor(key)
            with safe_open(model_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    model_ckpt[key] = f.get_tensor(key)
            comp_keys = set(sd_ckpt.keys()) - set(model_ckpt.keys())
            for key in comp_keys:
                model_ckpt[key] = sd_ckpt[key]
            model_path = model_path.replace('.safetensors', '_comp.safetensors')
            save_file(model_ckpt, model_path)
            print('==> Dump complemented weights at:', model_path)
        else:
            print('\n==> Skip complementing weights')
    except Exception as err:
        print('\n[Complement Error]:', err)
        sys.exit(1)

    # convert model to diffusers format
    dump_path = f'./output/{args.model_version_id}'
    base_model = meta_data['baseModel'].replace(' ', '')
    model_kwargs = dict(
        device=None,
        checkpoint_path=model_path,
        controlnet=args.controlnet,
        image_size=(768 if '768' in base_model else 512),
        original_config_file=map_config_file(base_model),
        from_safetensors=model_path.endswith('safetensors'),
        prediction_type=('v_prediction' if 'SD2' in base_model else 'epsilon'),
    )
    print('\nModel conversion kwargs:', model_kwargs)
    print('\n==> Conversion started')
    pipe = download_from_original_stable_diffusion_ckpt(**model_kwargs)
    if args.half:
        pipe.to(torch_dtype=torch.float16)
    if args.controlnet:
        pipe.controlnet.save_pretrained(dump_path, safe_serialization=False)
    else:
        pipe.save_pretrained(dump_path, safe_serialization=False)
    print('\n==> Model conversion completed')
    
    # convert vae to diffusers format
    if vae_info is not None:
        vae_dump_path = os.path.join(dump_path, 'vae')
        os.makedirs(vae_dump_path, exist_ok=True)
        vae_pt_to_vae_diffuser(vae_path, vae_dump_path)
        print('\n==> VAE conversion completed')

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version_id", default=None, type=str, required=True, help="Unique version id for the model.")
    parser.add_argument("--controlnet", action="store_true", default=None, help="Set flag if this is a controlnet checkpoint.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()

    # convert model format
    if SAFE_MODE:
        try:
            convert_single(args)
        except Exception as err:
            print('\n[Conversion Error]:', err)
    else:
        convert_single(args)
