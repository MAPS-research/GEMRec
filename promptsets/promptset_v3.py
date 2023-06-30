from datasets import Dataset, load_dataset
import pandas as pd
import random

def enhance_from_v2():
    df = pd.read_csv('promptset_v2.csv')
    
    for idx in range(len(df)):
        tag = df.loc[idx, 'tag']
        prompt = df.loc[idx, 'prompt']
        
        # don't add any additional prompts to abstract
        if tag == 'abstract':
            df.loc[idx, 'prompt'] = prompt.replace('masterpiece, best quality, best shadow, intricate, ', '')
            df.loc[idx, 'negativePrompt'] = ' '

        # add basic negative prompts to other tags
        else:
            df.loc[idx, 'negativePrompt'] = 'disfigured, blurry, bad art, lowres, low quality, weird colors, duplicate, out of frame, symmetric, NSFW'

        # add additional negative prompts to people
        if tag == 'people':
            df.loc[idx, 'negativePrompt'] = 'extra fingers, fewer fingers, bad hands, extra hands, bad anatomy, ' + df.loc[idx, 'negativePrompt']

    df.to_csv('promptset_v3.csv', index=False)

    return df

def add_prompts_from_civitai():
    df_c = pd.read_csv('promptset_c1.csv')
    df_v = pd.read_csv('promptset_v3.csv')

    num = 0
    # remove lora and nsfw in civitai prompts
    for idx in range(len(df_c)):
        if num == 11:
            break
    
        prompt = df_c.loc[idx, 'prompt']

        prompt = prompt.replace('nsfw', '').replace('nude', '').replace('naked', '').replace('pussy', '').replace('Nude', '')
        # remove str in the shape of <lora: * >
        prompt = ', '.join([p.strip() for p in prompt.split(',') if not p.strip().startswith('<lora:') and p.strip() != ''])
        
        if prompt == '':
            pass
        else:
            df_c.loc[idx, 'prompt'] = prompt
            df_c.loc[idx, 'negativePrompt'] = 'nsfw' + df_c.loc[idx, 'negativePrompt']
            # add civitai prompts to v3
            df_v.loc[len(df_v)] = df_c.loc[idx]

            num += 1
    
    df_v.to_csv('promptset_v3.csv', index=False)


if __name__ == "__main__":
    enhance_from_v2()
    add_prompts_from_civitai()