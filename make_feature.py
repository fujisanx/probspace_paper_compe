"""特徴量のマージとtrain test用のファイルアウトプット
Todo:
    * バージョン管理する？面倒か？
"""
import pandas as pd
import numpy as np
from pathlib import Path
from features.json_to_pickle import main as make_raw
from features.doi_feature import main as doi_feature
from features.version_feature import main as version_feature
from features.author_feature import main as author_feature
from features.category_feature import main as category_feature
from features.label_encoder import main as label_encoder
from features.statics_doi_cites_feature import main as statics_doi_cites_feature
from features.diff_feature import main as diff_feature


from functools import partial
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)


def _fall_within_range(dtype_min_value, dtype_max_value, min_value, max_value):
    """データ型の表現できる範囲に収まっているか調べる関数"""
    if min_value < dtype_min_value:
        # 下限が越えている
        return False

    if max_value > dtype_max_value:
        # 上限が越えている
        return False

    # 範囲内に収まっている
    return True


def _cast(df, col_name, cast_candidates):
    # カラムに含まれる最小値と最大値を取り出す
    min_value, max_value = df[col_name].min(), df[col_name].max()

    for cast_type, (dtype_min_value, dtype_max_value) in cast_candidates.items():
        if df[col_name].dtype == cast_type:
            # 同じ型まで到達した時点で、キャストする意味はなくなる
            return

        if _fall_within_range(dtype_min_value, dtype_max_value, min_value, max_value):
            # キャストしたことをログに残す
            LOGGER.info(f'column {col_name} casted: {df[col_name].dtype.type} to {cast_type}')
            # 最も小さなビット数で表現できる型にキャストできたので終了
            df[col_name] = df[col_name].astype(cast_type)
            return


def _cast_func(df, col_name):
    col_type = df[col_name].dtype.type

    if issubclass(col_type, np.integer):
        # 整数型
        cast_candidates = {
            cast_type: (np.iinfo(cast_type).min, np.iinfo(cast_type).max)
            for cast_type in [np.int8, np.int16, np.int32]
        }
        return partial(_cast, cast_candidates=cast_candidates)

    if issubclass(col_type, np.floating):
        # 浮動小数点型
        cast_candidates = {
            cast_type: (np.finfo(cast_type).min, np.finfo(cast_type).max)
            for cast_type in [np.float16, np.float32]
        }
        return partial(_cast, cast_candidates=cast_candidates)

    # その他は未対応
    return None


def _memory_usage(df):
    """データフレームのサイズと接頭辞を返す"""
    units = ['B', 'kB', 'MB', 'GB']
    usage = float(df.memory_usage().sum())

    for unit in units:
        if usage < 1024:
            return usage, unit
        usage /= 1024

    return usage, unit


def shrink(df):
    # 元のサイズをログに記録しておく
    usage, unit = _memory_usage(df)
    LOGGER.info(f'original dataframe size: {usage:.0f}{unit}')

    for col_name in tqdm(df.columns):
        # 各カラムごとにより小さなビット数で表現できるか調べていく
        func = _cast_func(df, col_name)
        if func is None:
            continue
        func(df, col_name)

    # 事後のサイズをログに記録する
    usage, unit = _memory_usage(df)
    LOGGER.info(f'shrinked dataframe size: {usage:.0f}{unit}')



def main():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_fmt,
                        level=logging.DEBUG)


    make_raw()
    doi_feature()
    version_feature()
    author_feature()
    category_feature()
    for ver in ['train', 'test']: # , 'test'
        DATA_PATH = Path(f"./data/{ver}")
        df = pd.read_pickle(DATA_PATH / 'raw.pickle')
        print('start... full merge', ver, df.shape)
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '02_doi_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '03_version_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '04_author_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / 'pred_doi_cites.pickle'), on='id', how='left')

        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_main_detail_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_main_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_name_parent_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_name_parent_main_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_name_parent_main_unique_name_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_name_parent_unique_name_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_name_unique_name_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_unique_parent_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '05_category_unique_feature.pickle'), on='id', how='left')

        # 型補正
        df.to_pickle(DATA_PATH / 'full.pickle')
        print('done... full feature shape', df.shape)

    label_encoder()
    for ver in ['train', 'test']:
        DATA_PATH = Path(f"./data/{ver}")
        df = pd.read_pickle(DATA_PATH / 'full.pickle')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '06_label_encoder.pickle'), on='id', how='left')
        df.to_pickle(DATA_PATH / 'full.pickle')

    statics_doi_cites_feature()
    for ver in ['train', 'test']:
        DATA_PATH = Path(f"./data/{ver}")
        df = pd.read_pickle(DATA_PATH / 'full.pickle')
        # 上記のfull pickleが必要な処理後
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_author_first_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_doi_id_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_pub_publisher_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_submitter_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_update_ym_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_first_created_ym_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_license_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_main_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_main_detail_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_name_parent_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_name_parent_main_label_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_name_label_doi_cites_feature.pickle'), on='id', how='left')

        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_author_first_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_doi_id_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_pub_publisher_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_submitter_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_update_ym_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_first_created_ym_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_license_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_main_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_main_detail_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_name_parent_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_name_parent_main_label_pred_doi_cites_feature.pickle'), on='id', how='left')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '07_statics_category_name_label_pred_doi_cites_feature.pickle'), on='id', how='left')

        df.to_pickle(DATA_PATH / 'full.pickle')


    diff_feature()
    for ver in ['train', 'test']:
        DATA_PATH = Path(f"./data/{ver}")
        df = pd.read_pickle(DATA_PATH / 'full.pickle')
        df = pd.merge(df, pd.read_pickle(DATA_PATH / '08_diff_feature.pickle'), on='id', how='left')

        df['is_null_comments'] = df['comments'].isnull() * 1
        df['is_null_report-no'] = df['report-no'].isnull() * 1
        df['is_null_journal-ref'] = df['journal-ref'].isnull() * 1

        df['pub_journals'] = np.log1p(df['pub_journals'])

        df['submitter_label'] = df['submitter_label'].astype('category')
        df['doi_id_label'] = df['doi_id_label'].astype('category')
        df['author_first_label'] = df['author_first_label'].astype('category')
        df['pub_publisher_label'] = df['pub_publisher_label'].astype('category')
        df['license_label'] = df['license_label'].astype('category')
        df['category_main_label'] = df['category_main_label'].astype('category')
        df['category_name_parent_label'] = df['category_name_parent_label'].astype('category')
        df['category_name_parent_main_label'] = df['category_name_parent_main_label'].astype('category')
        df['category_name_label'] = df['category_name_label'].astype('category')

        shrink(df)
        print(df.dtypes)

        df = pd.merge(df, pd.read_pickle(DATA_PATH / '10_roberta_raw.pickle'), on='id', how='left')
        df.to_pickle(DATA_PATH / 'full_comp.pickle')

        print('done... full merge', ver, df.shape)


if __name__ == "__main__":
    main()
