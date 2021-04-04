import pandas as pd
import numpy as np
import json
from pathlib import Path

def get_data_iter(path):
    with open(path, 'r') as f:
        for l in f:
            yield l

def make_dataframe(path, type='train'):
    tmp = []
    columns = ['id', 'submitter', 'authors', 'title', 'comments', 'journal-ref',
            'doi', 'report-no', 'categories', 'license', 'abstract', 'versions',
            'update_date', 'authors_parsed', 'doi_cites']    
    for line in get_data_iter(path):
        dict=json.loads(line)
        tmp.append(list(dict.values()))
    if type == 'train':
        return pd.DataFrame(tmp, columns=columns + ['cites'])
    else:
        return pd.DataFrame(tmp, columns=columns)

def main():
    DATA_PATH = Path("./data/")

    df_train = make_dataframe(DATA_PATH / 'train_data.json', 'train')
    df_train['doi_cites'] = df_train['doi_cites'].astype('int')
    df_train['doi_cites'] = np.log1p(df_train['doi_cites'], dtype=np.float32)
    df_train.to_pickle(DATA_PATH / 'train/raw.pickle')

    df_test = make_dataframe(DATA_PATH / 'test_data.json', 'test')
    df_test['doi_cites'] = df_test['doi_cites'].astype('int')
    df_test['doi_cites'] = np.log1p(df_test['doi_cites'], dtype=np.float32)
    df_test.to_pickle(DATA_PATH / 'test/raw.pickle')

if __name__ == "__main__":
    main()