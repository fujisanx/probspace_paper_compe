import pandas as pd
from pathlib import Path

"""doiの特徴量
- doiのid
- 著者初回名（doiのグループ化で利用）
Todo:
   TODOリストを記載
    * 全量の著者名は利用していない、
"""
def make_doi_feature(df, df_pub):
    df['doi_id'] = df['doi'].str.split('/', 1, expand=True)[0]
    df_merge = pd.merge(df[['id', 'doi_id']], df_pub, how='left', on='doi_id')
    return df_merge


def main():
    print('start doi encoder feature')

    DATA_PATH = Path("./data/")
    df_train = pd.read_pickle(DATA_PATH / 'train/raw.pickle')
    df_test = pd.read_pickle(DATA_PATH / 'test/raw.pickle')
    print('df_train', df_train.shape, 'df_test', df_test.shape)
    
    df_pub = (
        pd.read_csv('./data/doi_publishers.csv', dtype={'prefix':'object'})
            .rename(columns={
                'prefix': 'doi_id',
                'publisher': 'pub_publisher',
                'journals': 'pub_journals',
                'dois': 'pub_dois',
            })
    )

    df_train_result = make_doi_feature(df_train, df_pub)
    df_train_result.to_pickle(DATA_PATH / 'train/02_doi_feature.pickle')

    df_test_result = make_doi_feature(df_test, df_pub)
    df_test_result.to_pickle(DATA_PATH / 'test/02_doi_feature.pickle')
    print('df_train_result', df_train_result.shape, 'df_test_result', df_test_result.shape)


if __name__ == "__main__":
    main()