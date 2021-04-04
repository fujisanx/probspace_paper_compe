import pandas as pd
from pathlib import Path

"""doiの特徴量
- doiのid
- 著者初回名（doiのグループ化で利用）
Todo:
   TODOリストを記載
    * 全量の著者名は利用していない、
"""
def make_submitter_feature(df, df_pub):
    return df_merge

def main():
    DATA_PATH = Path("./data/")
    df_train = pd.read_pickle(DATA_PATH / 'train/raw.pickle')
    df_test = pd.read_pickle(DATA_PATH / 'test/raw.pickle')
    df_train_test = pd.concat([df_train, df_test])
    print('df_train', df_train.shape, 'df_test', df_test.shape, 'df_train_test', df_train_test.shape)

if __name__ == "__main__":
    main()