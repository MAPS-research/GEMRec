# Civitai to Diffusers

Download & convert the checkpoints on [Civitai](https://civitai.com/) to [Diffusers](https://huggingface.co/docs/diffusers/index) format.

## Sample Usage
```shell
git clone https://github.com/MAPS-research/PIG-misc && cd civitai2diffusers
python convert.py --model_version_id 29460
```

## Dir Explanations
`./meta`: Meta data for the checkpoints. \
`./output`: Converted models in [Diffusers](https://huggingface.co/docs/diffusers/index) format. \
`./download`: Checkpoint cache downloaded from [Civitai](https://civitai.com/).
