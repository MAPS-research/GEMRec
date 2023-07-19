from datasets import Dataset, load_dataset
import pandas as pd
import random

def enhance_from_v5():
    df = pd.read_csv('promptset_v5.csv')

    # add a column called note
    df['note'] = None

    # set cfgscale to 7
    df['cfgScale'] = 7

    # set sampler to 'Eular a'
    df['sampler'] = 'Eular a'

    # change the prompt_id of prompt tagged civitai
    for idx in range(len(df)):
        if df.loc[idx, 'tag'] == 'civitai':
            df.loc[idx, 'note'] == df.loc[idx, 'prompt_id']
            df.loc[idx, 'prompt_id'] = idx + 1
    
    # copy the first 9 prompts and add to the end, with note 'extended'
    for idx in range(9):
        df.loc[idx, 'note'] = 'original'
        df.loc[len(df)] = [len(df)+1, df.loc[idx, 'tag'], df.loc[idx, 'size'], df.loc[idx, 'seed'], df.loc[idx, 'prompt'], df.loc[idx, 'negativePrompt'], df.loc[idx, 'sampler'], df.loc[idx, 'cfgScale'], 'extended']

    df.to_csv('promptset_v6.csv', index = False)


def sort_v6():
    df = pd.read_csv('promptset_v6.csv')
    df.sort_values(by=['tag', 'size', 'prompt'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # rewrite prompt_id
    df['prompt_id'] = df.index + 1

    df.to_csv('promptset_v6.csv', index = False)


def move_modifier():
    df = pd.read_csv('promptset_v6.csv')
    for idx in range(len(df)):
        if df.loc[idx, 'prompt'].startswith("masterpiece, best quality, best shadow, intricate, "):
            df.loc[idx, 'prompt'] = df.loc[idx, 'prompt'][len("masterpiece, best quality, best shadow, intricate, "):] + ", masterpiece, best quality, best shadow, intricate"

    df.to_csv('promptset_v6.csv', index = False)

def resize_prompts():
    df = pd.read_csv('promptset_v6.csv')
    for idx in range(len(df)):
        width, height = df.loc[idx, 'size'].split('x')
        if width < height:
            df.loc[idx, 'size'] = '512x768'
        elif width > height:
            df.loc[idx, 'size'] = '768x512'
        else:
            df.loc[idx, 'size'] = '512x512'
    
    df.to_csv('promptset_v6.csv', index = False)
        

if __name__ == "__main__":
    # enhance_from_v5()
    # move_modifier()
    resize_prompts()
    sort_v6()

