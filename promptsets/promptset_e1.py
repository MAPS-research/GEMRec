from datasets import Dataset, load_dataset
import pandas as pd
import random

def alter_from_v2():
    df = pd.read_csv('promptset_v2.csv')

    for idx in range(len(df)):
        # print(df.loc[idx, 'prompt'])
        df.loc[idx, 'prompt'] = df.loc[idx, 'prompt'].replace('masterpiece, best quality, best shadow, intricate, ', '')
        df.loc[idx, 'negativePrompt'] = ' '
    
    df.to_csv('promptset_e1.csv', index=False)
    return df

def test_load():
    df = pd.read_csv('promptset_v2.csv')
    print(type(df.loc[0, 'negativePrompt']))

if __name__ == "__main__":
    alter_from_v2()
    # test_load()