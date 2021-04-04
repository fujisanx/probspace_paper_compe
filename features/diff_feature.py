import pandas as pd
from pathlib import Path
import itertools

"""著者情報の特徴量
- 著者数の数（おそらく一人や100を超える大規模なのは引用低いのでは）
- 著者初回名（doiのグループ化で利用）
Todo:
   TODOリストを記載
    * 全量の著者名は利用していない、
"""
def make_diff_feature(df, df_ids,colA, colB):
    df[f'diff_{colA}_{colB}'] = df[colA] -df[colB]
    df[f'diff_rate_{colA}_{colB}'] = (df[colA] + 1) / (df[colB] + 1)
    return pd.concat([df_ids, df[[f'diff_{colA}_{colB}']], df[f'diff_rate_{colA}_{colB}']], axis=1)

def main():
    print('start diff feature')
    DATA_PATH = Path("./data/")

    colA = 'doi_cites'
    df = pd.read_pickle(DATA_PATH / 'train/full.pickle')

    keys = [
        'doi_cites', 
        'doi_cites_mean_submitter_label', 'doi_cites_mean_doi_id_label', 'doi_cites_mean_author_first_label', 'doi_cites_mean_pub_publisher_label',
        'doi_cites_mean_update_ym', 'doi_cites_mean_first_created_ym', 'doi_cites_mean_license_label',
        'doi_cites_mean_category_main_label', 'doi_cites_mean_category_main_detail_label', 'doi_cites_mean_category_name_parent_label', 
        'doi_cites_mean_category_name_parent_main_label', 'doi_cites_mean_category_name_label',

        'pred_doi_cites', 
        'pred_doi_cites_mean_submitter_label', 'pred_doi_cites_mean_doi_id_label', 'pred_doi_cites_mean_author_first_label', 'pred_doi_cites_mean_pub_publisher_label',
        'pred_doi_cites_mean_update_ym', 'pred_doi_cites_mean_first_created_ym', 'pred_doi_cites_mean_license_label',
        'pred_doi_cites_mean_category_main_label', 'pred_doi_cites_mean_category_main_detail_label', 'pred_doi_cites_mean_category_name_parent_label', 
        'pred_doi_cites_mean_category_name_parent_main_label', 'pred_doi_cites_mean_category_name_label'
    ]

    df_result = df[['id']].copy()
    for a, b in list(itertools.combinations(keys,2)):
        df_result = pd.merge(df_result, make_diff_feature(df, df[['id']], a, b), on='id', how='left')
    df_result.to_pickle(DATA_PATH / 'train/08_diff_feature.pickle')
    print('diff feature_train_result', df_result.shape)

    df = pd.read_pickle(DATA_PATH / 'test/full.pickle')
    df_result = df[['id']].copy()
    for a, b in list(itertools.combinations(keys,2)):
        df_result = pd.merge(df_result, make_diff_feature(df, df[['id']], a, b), on='id', how='left')
    df_result.to_pickle(DATA_PATH / 'test/08_diff_feature.pickle')

    print('diff feature test_result', df_result.shape)


if __name__ == "__main__":
    main()