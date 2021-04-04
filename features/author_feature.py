import pandas as pd
from pathlib import Path

"""著者情報の特徴量
- 著者数の数（おそらく一人や100を超える大規模なのは引用低いのでは）
- 著者初回名（doiのグループ化で利用）
Todo:
   TODOリストを記載
    * 全量の著者名は利用していない、
"""
def make_authos_feature(df):
    df_temp = df.copy()
    df_temp['author_num'] = [len(x) for x in df.authors_parsed.tolist()]
    df_temp['author_first'] = [x[0][0] for x in df.authors_parsed.tolist()]
    return df_temp[['id', 'author_first', 'author_num']]

def main():
    print('start author feature')
    DATA_PATH = Path("./data/")
    df_train = pd.read_pickle(DATA_PATH / 'train/raw.pickle')
    df_test = pd.read_pickle(DATA_PATH / 'test/raw.pickle')
    print('df_train', df_train.shape, 'df_test', df_test.shape)

    df_train_result = make_authos_feature(df_train)
    df_train_result.to_pickle(DATA_PATH / 'train/04_author_feature.pickle')

    df_test_result = make_authos_feature(df_test)
    df_test_result.to_pickle(DATA_PATH / 'test/04_author_feature.pickle')
    print('df_train_result', df_train_result.shape, 'df_test_result', df_test_result.shape)


if __name__ == "__main__":
    main()