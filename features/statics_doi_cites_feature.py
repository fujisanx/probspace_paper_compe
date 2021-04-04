import pandas as pd
from pathlib import Path

"""doi_citesの特徴量
- doi_citesの各特長からの算出　平均, min max std 
Todo:
    * 
"""


def make_statics_feature(df, key_col, target_col):
    print('start statics feature', key_col, target_col)
    df_result = df[[key_col, target_col]].groupby(key_col).mean().rename(columns={target_col:f'{target_col}_mean_{key_col}'}).reset_index()
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).count().rename(columns={target_col: f'{target_col}_count_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).sum().rename(columns={target_col: f'{target_col}_sum_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).min().rename(columns={target_col: f'{target_col}_min_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).max().rename(columns={target_col: f'{target_col}_max_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).median().rename(columns={target_col:f'{target_col}_median_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).std().rename(columns={target_col: f'{target_col}_std_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).quantile(.10).rename(columns={target_col: f'{target_col}_q10_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).quantile(.25).rename(columns={target_col: f'{target_col}_q25_{key_col}'}).reset_index(), on=key_col, how='left')
    df_result = pd.merge(df_result, df[[key_col, target_col]].groupby(key_col).quantile(.75).rename(columns={target_col: f'{target_col}_q75_{key_col}'}).reset_index(), on=key_col, how='left')
    return df_result


def make_statics_id_feature(df_origin, df_feature, key_col):
    df_result = df_origin.copy()[['id', key_col]]
    df_result = pd.merge(df_result, df_feature, on=key_col, how='left')
    df_result = df_result.drop(key_col, axis=1)
    return df_result

def make_train_test(df_train, df_test, df_feature, key, target_col):
    DATA_PATH = Path("./data/")
    print(key, 'train', df_train.shape, 'test', df_test.shape, 'feature', df_feature.shape, df_train[key].nunique(), df_feature[key].nunique())

    df_train_result = make_statics_id_feature(df_train, df_feature, key)
    df_test_result = make_statics_id_feature(df_test, df_feature, key)
    df_train_result.to_pickle(DATA_PATH / f'train/07_statics_{key}_{target_col}_feature.pickle')
    df_test_result.to_pickle(DATA_PATH / f'test/07_statics_{key}_{target_col}_feature.pickle')
    print(key, 'train', df_train_result.shape, 'test', df_test_result.shape)

def main():
    DATA_PATH = Path("./data/")
    df_train = pd.read_pickle(DATA_PATH / 'train/full.pickle')
    df_test = pd.read_pickle(DATA_PATH / 'test/full.pickle')
    df_train_test = pd.concat([df_train, df_test])
    print('df_train', df_train.shape, 'df_test', df_test.shape, 'df_train_test', df_train_test.shape)
    
    target_col = 'doi_cites'
    key = 'submitter_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'doi_id_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'author_first_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)
    
    key = 'pub_publisher_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'update_ym'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'first_created_ym'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'license_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_main_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_main_detail_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_name_parent_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_name_parent_main_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_name_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)






    
    target_col = 'pred_doi_cites'
    key = 'submitter_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'doi_id_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'author_first_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)
    
    key = 'pub_publisher_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'update_ym'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'first_created_ym'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'license_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_main_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_main_detail_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_name_parent_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_name_parent_main_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)

    key = 'category_name_label'
    df_feature = make_statics_feature(df_train_test, key, target_col)
    make_train_test(df_train, df_test, df_feature, key, target_col)



if __name__ == "__main__":
    main()