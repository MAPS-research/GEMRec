from datasets import Dataset, load_dataset
import pandas as pd
import random

def load_parti_prompts(sampel_num=5):
    parti_prompts = load_dataset('diffusers-parti-prompts/sd-v1-5', split='train')

    # turn Dataset into pandas
    df = pd.DataFrame(parti_prompts)

    test_prompts = df[df['Challenge']!='Basic'][['Prompt', 'Category', 'Challenge']]
    # change category format to match our tags
    test_prompts.loc[:, 'Category'] = test_prompts['Category'].apply(lambda x: x.lower().strip('s'))

    selected_prompts = test_prompts.groupby('Category').apply(lambda x: x.sample(sampel_num, random_state = 2023)).reset_index(drop=True)

    category_replace_map = {'food & beverage': 'food', 'indoor scene': 'architecture', 'outdoor scene': 'scenery'}
    for key in category_replace_map:
        selected_prompts.loc[selected_prompts['Category'] == key, 'Category'] = category_replace_map[key]


    return selected_prompts

def enhance_from_v1(prompts):
    df = pd.read_csv('promptset_v1.csv')
    # print(df)
    
    prompt_id = len(df) + 1
    random.seed(73513)

    for idx in range(len(prompts)):
        meta = prompts.iloc[idx]

        tag = meta['Category']
        size = '512x768' if tag in ['people', 'art', 'animal'] else '768x512'
        seed = random.randint(100000000, 10000000000)
        prompt = 'masterpiece, best quality, best shadow, intricate, ' + meta['Prompt']
        negativePrompt = "extra fingers, fewer fingers, extra hands, bad hands, worst quality, bad quality, lowres, watermark, nsfw"
        sampler = None
        cfgScale = 7
        df.loc[len(df.index)] = [prompt_id, tag, size, seed, prompt, negativePrompt, sampler, cfgScale]

        prompt_id += 1

    df.to_csv('promptset_v2.csv', index = False)
    return df


if __name__ == "__main__":
    extra_prompts = load_parti_prompts(sampel_num=5)
    promptset_v2 = enhance_from_v1(prompts = extra_prompts)

