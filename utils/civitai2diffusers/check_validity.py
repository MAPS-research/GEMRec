
import os
import json

file_list = os.listdir('./meta')

for file in sorted(file_list):
    print(f'Checking {file:9s}:', end=' ')
    path = os.path.join('./meta', file)
    with open(path, 'r') as f:
        meta_data = json.load(f)
    vid = str(meta_data['id'])
    vae_info, model_info = None, None
    for file_info in meta_data['files']:
        if file_info['primary']:
            model_info = file_info
        if file_info['type'] == 'VAE':
            vae_info = file_info
    model_path = os.path.join('./download', vid, model_info['name'])
    assert os.path.exists(model_path) and (int(os.stat(model_path).st_size / 1024) == int(model_info['sizeKB'])), (vid, model_path)
    if vae_info is not None:
        vae_path = os.path.join('./download', vid, vae_info['name'])
        assert os.path.exists(vae_path) and (int(os.stat(vae_path).st_size / 1024) == int(vae_info['sizeKB'])), (vid, model_path)
    print('done')
