# ModelCoffer

Download & convert models from [civitai](https://civitai.com/) to [diffusers](https://huggingface.co/docs/diffusers/index), and generate images with them


## Linked hugginface repo
### Datasets
[ModelCofferRoster](https://huggingface.co/datasets/NYUSHPRP/ModelCofferRoster) - The dataset of 200 model checkpoints fetched from Civitai\
[ModelCofferPromptbook](https://huggingface.co/datasets/NYUSHPRP/ModelCofferPromptBook) - The full version of our GemRec-18k dataset, including images and their corresponding metadata\
[ModelCofferMetadata](https://huggingface.co/datasets/NYUSHPRP/ModelCofferMetadata) - The same as ModelCofferPromptbook, but without images

### Space
[ModelCofferGallery](https://huggingface.co/spaces/NYUSHPRP/ModelCofferGallery) [**under construction yet**] - Our web application that allows you to browse and compare images generated by different models. We will collect user prefernece data for personalization 


## Basic Usage
### STEP1: Clone repo
Clone this ModelCoffer repo
```
git clone https://github.com/MAPS-research/ModelCoffer.git
```

### STEP2: Replicate our GemRec-18k dataset presented in our paper and our public dataset
All models we used are registered in `./roster.csv`, and our prompts are stored in `./promptsets/promptset_v6.csv`. To replicate our GemRec-18k dataset, run
```
python3 download_and_generate.py --sd --lr
```
where --sd means generating with three default stable diffusion models, and --lr means generating with the remaining 197 models in `./roster.csv`. The generated images will be stored in `./generated/train`. It will take awail to download all models, and might break halfway due to civitai server error. Please be patient, and if the script breaks, just rerun it, it will skip the downloaded models and continue to download the rest.

## Advanced Usage
**Note that there will be some verbose printing in the terminal, which is for debugging purpose.**

### Use the conversion script only
We provide a script for converting models from civitai to diffusers, enhanced from the [official diffusers conversion script](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py) It handles different base model versions, and will skip those models with file error. The converted models will be stored in `./output`. 
```
cd utils/civitai2diffusers
python3 convert.py --model_version_id 00000
```
Replace `00000` with the modelVersion_id you want to convert. You can find the modelVersion_id on [Civitai](https://civitai.com/) or from its [api](https://github.com/civitai/civitai/wiki/REST-API-Reference)\

**Note: Remeber to put `v1-inference.yaml`, `v2-inference.yaml`, `v2-inference-v.yaml` under the same working directory as `convert.py` script**

### Download latest models from civitai and generate images
Fetch model from civitai or read local models, and generate images with metadatas in promptsets. It will continue to run untill all models are downloaded from civitai. The generated images will be stored in `./generated/train`. 
```
python3 download_and_generate.py --fn
```

### Download a subset of models that have almost the same population distribution as the fullset of models on civitai
Change to the directory of `./everything` and run `get_models.py` to download metadata of all models from civitai. 
```
cd everything
python3 get_models.py
```
Change to the directory of `./utils/popularity`
```
cd ../utils/popularity
```
Plot histogram for old subset & find bin sizes
```
python3 hist.py
```
Distribute all models into the bins
```
python3 grouping.py
```
Deal with subset size, cache models, and candidate models. Cache models are those already inside the roster from each bin, candidate models are those not in the roster but in each bin.
```
python3 sampling.py
```
Change to the root directory of this repo
```
cd ../..
```
Download and generate images with the subset of models. This script will autometically download the right number of models from each bin to match the population distribution of the fullset of models on civitai. The generated images will be stored in `./generated/train`. 
```
python3 download_and_generate.py -p all
```

## Files
`roster.csv`: records models downloaded from civitai. Contains tag,model_name,model_id,modelVersion_name,modelVersion_id,modelVersion_url,modelVersion_trainedWords,model_download_count

## Directories
### Model data
`./meta`: Meta data for the checkpoints. \
`./output`: Converted models in [Diffusers](https://huggingface.co/docs/diffusers/index) format. \
`./download`: Checkpoint cache downloaded from [Civitai](https://civitai.com/). \
`./everything`: This folder contains a script for downloading all model metadata from [Civitai](https://civitai.com/) using its [api](https://github.com/civitai/civitai/wiki/REST-API-Reference), with naming as `{model id}_{latest modelVersion id}.json`. Files will be stored under `./everything/models`

### Prompts
`./promptsets`: Prompts and other metadata for image generation, `promptset_v6.py` is used for generating GemRec-18k dataset.

### Generated images
`./generated/train`: Images generated with all downloaded models using the latest `promptset_v*.csv`.


## TODO
- [ ] Add more custom control for downloading
- [ ] Better formatted logging output