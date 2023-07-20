from datasets import Dataset, load_dataset
import pandas as pd
import random
import numpy as np

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

def change_note_datatype():
    df = pd.read_csv('promptset_v6.csv')

    # check datatypes
    datatypes = []
    for idx in range(len(df)):
        data = df.loc[idx, 'note']
        datatype = type(data)
        if datatype not in datatypes:
            datatypes.append(datatype)
    
    print(datatypes)

    # pick all rows with 'note' as str type, while it is digit, and change the datatype to numpy.int64, and save to pandas
    df.loc[df['note'].apply(lambda x: type(x) == str and x.isdigit()), 'note'] = df.loc[df['note'].apply(lambda x: type(x) == str and x.isdigit()), 'note'].astype("int64")
    
    # check datatypes
    datatypes = []
    for idx in range(len(df)):
        data = df.loc[idx, 'note']
        datatype = type(data)
        if datatype not in datatypes:
            datatypes.append(datatype)
    
    print(datatypes)

    df.to_csv('promptset_v6.csv', index = False)

    df2 = pd.read_csv('promptset_v6.csv')
    datatypes2 = []
    for idx in range(len(df2)):
        data = df2.loc[idx, 'note']
        datatype = type(data)
        if datatype not in datatypes2:
            datatypes2.append(datatype)
    
    print(datatypes2)
        

if __name__ == "__main__":
    # enhance_from_v5()
    # move_modifier()
    # resize_prompts()
    # sort_v6()
    change_note_datatype()

