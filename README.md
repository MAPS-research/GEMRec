# ModelCoffer

Download & convert models from civitai to diffusers, and generate images with them

## Usage
### Clone repo
Clone ModelCoffer repo
```
git clone https://github.com/MAPS-research/ModelCoffer.git
```

Clone PIG-misc under this `ModelCoffer` folder
```
git clone https://github.com/MAPS-research/PIG-misc.git
```

### Generate images
Fetch model from civitai or read local models, and generate images with metadatas in promptsets
```
python3 download_and_generate.py --gb --lr --fn --s train
```

### Upload to huggingface
Evaluate images and upload datasets to huggingface
```
python3 evaluate_and_upload.py
```
or run with sbatch
```
sbatch evaluate_and_upload.slurm
```

## Linked hugginface repo
### Datasets
[ModelCofferRoster](https://huggingface.co/datasets/NYUSHPRP/ModelCofferRoster) \
[ModelCofferPromptbook](https://huggingface.co/datasets/NYUSHPRP/ModelCofferPromptBook) \
[ModelCofferMetadata](https://huggingface.co/datasets/NYUSHPRP/ModelCofferMetadata)

### Space
[ModelCofferGallery](https://huggingface.co/spaces/NYUSHPRP/ModelCofferGallery)


## Files
`roster.csv`: records models downloaded from civitai. Contains tag,model_name,model_id,modelVersion_name,modelVersion_id,modelVersion_url,modelVersion_trainedWords,model_download_count

## Directories
### Model data
`./meta`: Meta data for the checkpoints. \
`./output`: Converted models in [Diffusers](https://huggingface.co/docs/diffusers/index) format. \
`./download`: Checkpoint cache downloaded from [Civitai](https://civitai.com/).

### Prompts
`./promptsets`: Prompts and other metadata for image generation

### Generated images
`./generated/train`: Images generated with all downloaded models using the latest `promptset_v*.csv`. \
`./generated/val`: Images generated with experimental prompts using `promptset_e*.csv`. \
Technically speaking, these are not real train and validation splits. Here I use this file structure just for convenience when loading with huggingface datasets

