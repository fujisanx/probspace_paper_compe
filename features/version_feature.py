import pandas as pd
from pathlib import Path


"""versionの特徴量
- update_date
- version
    - 初回作成日
    - 最終更新日
    - 差分
- 年
- 月
- 年月
- 差分時間
- 差分日付
Todo:
   TODOリストを記載
    * 全量の著者名は利用していない、
"""
def make_version_feature(input_df):
    output_df = input_df.copy()
    output_df['update_date'] = pd.to_datetime(input_df.update_date)
    output_df['first_created_date'] = pd.to_datetime(input_df.versions.apply(lambda p: p[0]['created']))
    output_df['last_created_date'] = pd.to_datetime(input_df.versions.apply(lambda p: p[-1]['created']))
    output_df = output_df[['id', 'update_date', 'first_created_date', 'last_created_date']]

    output_df['update_year'] = output_df['update_date'].dt.year
    output_df['first_created_year'] = output_df['first_created_date'].dt.year
    output_df['last_created_year'] = output_df['last_created_date'].dt.year

    output_df['update_month'] = output_df['update_date'].dt.month
    output_df['first_created_month'] = output_df['first_created_date'].dt.month
    output_df['last_created_month'] = output_df['last_created_date'].dt.month

    output_df['update_ym'] = output_df['update_date'].dt.year * 100 + output_df['update_date'].dt.month
    output_df['first_created_ym'] = output_df['first_created_date'].dt.year * 100 + output_df['first_created_date'].dt.month
    output_df['last_created_ym'] = output_df['last_created_date'].dt.year * 100 + output_df['last_created_date'].dt.month

    output_df['update_day'] = output_df['update_date'].dt.day
    output_df['first_created_day'] = output_df['first_created_date'].dt.day
    output_df['last_created_day'] = output_df['last_created_date'].dt.day

    output_df['update_date_unixtime'] = output_df['update_date'].astype(int) / 1000000000
    output_df['first_created_unixtime'] = output_df['first_created_date'].astype(int) / 1000000000
    output_df['last_created_unixtime'] = output_df['last_created_date'].astype(int) / 1000000000
    output_df['diff_update_date_unixtime'] = output_df['update_date_unixtime'] - output_df['first_created_unixtime']
    output_df['diff_update_date_unixtime'] = output_df['diff_update_date_unixtime'].astype('int')
    output_df['diff_created_unixtime'] = output_df['last_created_unixtime'] - output_df['first_created_unixtime']

    output_df['num_created'] = input_df.versions.apply(lambda p: len(p))
    
    output_df['update_date_days'] = output_df['update_date_unixtime'] / (60 * 60  * 24) 
    output_df['update_date_days'] = output_df['update_date_days'].astype('int')
    output_df['first_created_days'] = output_df['first_created_unixtime'] / (60 * 60  * 24)
    output_df['first_created_days'] = output_df['first_created_days'].astype('int')
    output_df['last_created_days'] = output_df['last_created_unixtime'] / (60 * 60  * 24)
    output_df['last_created_days'] = output_df['last_created_days'].astype('int')
    output_df['diff_created_days'] = output_df['last_created_days'] - output_df['first_created_days']
    output_df['rate_created_days'] = (output_df['diff_created_days'] + 1) / (output_df['num_created'] + 1)
    output_df.drop(['update_date', 'first_created_date', 'last_created_date'], axis=1)

    return output_df

def main():
    DATA_PATH = Path('./data/')
    df_train = pd.read_pickle(DATA_PATH / 'train/raw.pickle')
    df_test = pd.read_pickle(DATA_PATH / 'test/raw.pickle')

    make_version_feature(df_train).to_pickle(DATA_PATH / 'train/03_version_feature.pickle')
    make_version_feature(df_test).to_pickle(DATA_PATH / 'test/03_version_feature.pickle')


if __name__ == '__main__':
    main()