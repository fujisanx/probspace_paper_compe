import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def make_encoder_feature(df, col):
    le = LabelEncoder()
    return le.fit_transform(df[col].fillna('null0').values)


def main():
    print('start label encoder feature')
    DATA_PATH = Path("./data/")
    df_train = pd.read_pickle(DATA_PATH / 'train/full.pickle')
    df_test = pd.read_pickle(DATA_PATH / 'test/full.pickle')
    df_train_test = pd.concat([df_train, df_test])
    print('df_train', df_train.shape, 'df_test', df_test.shape, 'df_train_test', df_train_test.shape)
    
    df_result = df_train_test.copy()[['id']]
    df_result['submitter_label'] = make_encoder_feature(df_train_test, 'submitter')
    df_result['doi_id_label'] = make_encoder_feature(df_train_test, 'doi_id')
    df_result['author_first_label'] = make_encoder_feature(df_train_test, 'author_first')
    df_result['pub_publisher_label'] = make_encoder_feature(df_train_test, 'pub_publisher')
    df_result['license_label'] = make_encoder_feature(df_train_test, 'license')
    df_result['category_main_label'] = make_encoder_feature(df_train_test, 'category_main')
    df_result['category_main_detail_label'] = make_encoder_feature(df_train_test, 'category_main_detail')
    df_result['category_name_parent_label'] = make_encoder_feature(df_train_test, 'category_name_parent_unique')
    df_result['category_name_parent_main_label'] = make_encoder_feature(df_train_test, 'category_name_parent_main_unique')
    df_result['category_name_label'] = make_encoder_feature(df_train_test, 'category_name_unique')

    df_train_result = pd.merge(df_train[['id']], df_result, on='id', how='left')
    df_train_result.to_pickle(DATA_PATH / 'train/06_label_encoder.pickle')
    df_test_result = pd.merge(df_test[['id']], df_result, on='id', how='left')
    df_test_result.to_pickle(DATA_PATH / 'test/06_label_encoder.pickle')
    print('df_train_result', df_train_result.shape, 'df_test_result', df_test_result.shape)

if __name__ == "__main__":
    main()