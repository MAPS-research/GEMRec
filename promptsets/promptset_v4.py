from datasets import Dataset, load_dataset
import pandas as pd
import random


def enhance_from_v3():
    df = pd.read_csv('promptset_v3.csv')

    # drop prompts with tag 'civitai'
    df = df[df['tag'] != 'civitai']
    df.reset_index(drop=True, inplace=True)

    # add one prompt
    df.loc[len(df)] = [len(df)+1, 'world knowledge', '512x768', random.randint(100000000, 10000000000), 'masterpiece, best quality, best shadow, intricate, kpop, korea star, idol producer, black shirt, grey long pants, white suspenders, perfect figure, full body, playing basketball, white background, short hair, middle parting hair style, gray hair, slim', 'worst quality, low quality, normal quality, lowres, monochrome, grayscale', None, None]

    df.to_csv('promptset_v4.csv', index = False)


if __name__ == "__main__":
    enhance_from_v3()