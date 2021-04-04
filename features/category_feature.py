import pandas as pd
from pathlib import Path
import collections

"""categoryの特徴量
- categoryをワンホットエンコード
- category parent
- category child
- 子も作成
"""

# id, category, parent, parent_main, child
def make_id_category(df_train_test, path):
    # カテゴリを分割して横に
    df_sp = df_train_test.copy()[['id', 'categories']]
    df_sp = pd.concat([df_sp[['id']], df_sp['categories'].str.split(' ',expand=True)], axis=1)
    print('split category', df_sp.shape)
    (
        pd.concat([df_sp['id'], df_sp[0].str.split('.',expand=True)[0]], axis=1)
            .rename(columns={0:'category_main'})
            .to_pickle(path / 'id_category_main_train_test.pickle')
    )
    (
        pd.concat([df_sp['id'], df_sp[0].str.split('.',expand=True)[0]], axis=1)
            .rename(columns={0:'category_main_detail'})
            .to_pickle(path / 'id_category_main_detail_train_test.pickle')
    )


    # id category
    idx = 0
    temp = df_sp[df_sp[idx].isnull() == False][['id', idx]].rename(columns={idx:0})
    df_uni = pd.concat([df_sp[['id', 0]], temp])
    for idx in range(1, len(df_sp.columns) - 1):
        temp = df_sp[df_sp[idx].isnull() == False][['id', idx]].rename(columns={idx:0})
        df_uni = pd.concat([df_uni, temp])
    df_uni = df_uni.rename(columns={0: 'category_name'})
    print('to union', df_uni.shape)

    # id, category, parent, child
    df_uni['category_name'] = df_uni['category_name'].str.lower()
    df_cat = pd.concat([df_uni[['id', 'category_name']], df_uni['category_name'].str.split('.',expand=True).rename(columns={0:'category_name_parent', 1:'category_name_child'})], axis=1)
    print('category', df_cat.shape)
    df_cat.head()

    # id, category, parent, parent_main, child
    df_cat2 = pd.concat([df_cat[['id', 'category_name', 'category_name_parent', 'category_name_child']], df_cat['category_name_parent'].str.split('-', expand=True).rename(columns={0:'category_name_parent_main', 1:'category_name_parent_sub'})], axis=1)
    print('category_main_split', df_cat.shape)

    df_cat2.to_pickle(path / 'id_category_train_test.pickle')
    return df_cat2

# onehot encode
def make_one_hot_encode(df, target_col):
    # to 0, 1 endoe
    df_dummy = pd.get_dummies(df[target_col])
    df_dummy_with_id = pd.concat([df[['id']], df_dummy], axis=1)
    df_sum = df_dummy_with_id.groupby('id').sum()

    for col in df_sum.columns:
        df_sum[col] = df_sum[col] > 0
        df_sum[col] = df_sum[col].astype('int')

    df_sum = df_sum.reset_index()
    return df_sum


# train test pickle
def make_join_id(df_train, df_test, df_feature, key):
    DATA_PATH = Path("./data/")
    df_train_result = pd.merge(df_train[['id']], df_feature, on='id', how='left')
    df_train_result.to_pickle(DATA_PATH / f'train/05_{key}_feature.pickle')

    df_test_result = pd.merge(df_test[['id']], df_feature, on='id', how='left')
    df_test_result.to_pickle(DATA_PATH / f'test/05_{key}_feature.pickle')
    print(key, 'train', df_train_result.shape, 'test', df_test_result.shape)


def make_unique_category(df_ids, path):
   # 重複除去
    target_col1 = pd.read_pickle(path / '05_category_name_parent_feature.pickle').columns[1:]
    target_col2 = pd.read_pickle(path / '05_category_name_parent_main_feature.pickle').columns[1:]
    target_col3 = pd.read_pickle(path / '05_category_name_feature.pickle').columns[1:]

    target = []
    target.extend(target_col1)
    target.extend(target_col2)

    ignore_col = []
    for k, v in collections.Counter(target).items():
        if v >= 2:
            ignore_col.append(k)
    ignore_col

    df_concat = pd.read_pickle(path / '05_category_name_parent_feature.pickle')
    df_concat = df_concat.drop(ignore_col, axis=1)
    df_concat = pd.merge(df_concat, pd.read_pickle(path / '05_category_name_parent_main_feature.pickle'), on='id', how='left')

    df_train_result = pd.merge(df_ids, df_concat, on='id', how='left')
    df_train_result.to_pickle(path / '05_category_unique_parent_feature.pickle')
    print(path, df_train_result.shape)

    # 小区分の方
    target_col1 = pd.read_pickle(path / '05_category_unique_parent_feature.pickle').columns[1:]
    target_col2 = pd.read_pickle(path / '05_category_name_feature.pickle').columns[1:]

    target = []
    target.extend(target_col1)
    target.extend(target_col2)

    ignore_col = []
    for k, v in collections.Counter(target).items():
        if v >= 2:
            ignore_col.append(k)
    ignore_col

    df_concat = pd.read_pickle(path / '05_category_name_feature.pickle')
    df_concat = df_concat.drop(ignore_col, axis=1)

    df_train_result = pd.merge(df_ids, df_concat, on='id', how='left')
    df_train_result.to_pickle(path / '05_category_unique_feature.pickle')
    print(path, df_train_result.shape)

def make_id_text(df_train, df_test, df_cat, target):
    df_temp = df_cat.copy()
    df_temp = df_cat[~df_cat.duplicated()].sort_values(target).groupby('id').agg({target: ' '.join}).rename(columns={target:f'{target}_unique'})
    DATA_PATH = Path("./data/")
    df_train_result = pd.merge(df_train[['id']], df_temp, on='id', how='left')
    df_train_result.to_pickle(DATA_PATH / f'train/05_{target}_unique_name_feature.pickle')

    df_test_result = pd.merge(df_test[['id']], df_temp, on='id', how='left')
    df_test_result.to_pickle(DATA_PATH / f'test/05_{target}_unique_name_feature.pickle')
    print(target, 'train', df_train_result.shape, 'test', df_test_result.shape)


def main():
    DATA_PATH = Path("./data/")
    df_train = pd.read_pickle(DATA_PATH / 'train/raw.pickle')
    df_test = pd.read_pickle(DATA_PATH / 'test/raw.pickle')

    df_train_test = pd.concat([df_train, df_test])
    print('df_train', df_train.shape, 'df_test', df_test.shape, 'df_train_test', df_train_test.shape)

    df_cat = make_id_category(df_train_test, DATA_PATH)
    df_feature = pd.read_pickle(DATA_PATH / 'id_category_main_train_test.pickle')
    key = 'category_main'
    make_join_id(df_train, df_test, df_feature, key)

    key = 'category_main_detail'
    df_feature = pd.read_pickle(DATA_PATH / 'id_category_main_detail_train_test.pickle')
    make_join_id(df_train, df_test, df_feature, key)

    # category_name, category_name_parent, category_name_child, category_name_parent_main, category_name_parent_sub

    key = 'category_name_parent'
    df_feature = make_one_hot_encode(df_cat, key)
    make_join_id(df_train, df_test, df_feature, key)
    make_id_text(df_train, df_test, df_cat, key)

    key = 'category_name_parent_main'
    df_feature = make_one_hot_encode(df_cat, key)
    make_join_id(df_train, df_test, df_feature, key)
    make_id_text(df_train, df_test, df_cat, key)
    
    key = 'category_name'
    df_feature = make_one_hot_encode(df_cat, key)
    make_join_id(df_train, df_test, df_feature, key)
    make_id_text(df_train, df_test, df_cat, key)
    '''
    key = 'category_name_child'
    df_feature = make_one_hot_encode(df_cat, key)
    make_join_id(df_train, df_test, df_feature, key)
    make_id_text(df_train, df_test, df_cat, key)

    key = 'category_name_parent_sub'
    df_feature = make_one_hot_encode(df_cat, key)
    make_join_id(df_train, df_test, df_feature, key)
    make_id_text(df_train, df_test, df_cat, key)
    '''

    make_unique_category(df_train[['id']], Path("./data/train/"))
    make_unique_category(df_test[['id']], Path("./data/test/"))



if __name__ == "__main__":
    main()