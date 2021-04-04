import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import collections
pd.set_option('display.max_columns', 500)

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import transformers

import nltk
import re


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from transformers import *

plt.style.use('seaborn')

MOLDEL_NAME = 'roberta-base'
MAX_LEN = 312
"""title abstractの特徴量
- 複数のベクトル化が考えられる
Todo:
    * BERT　from hagging face pretrained
    * doc2vec
"""


def encode_texts(texts, maxlen=312):
    tokenizer = AutoTokenizer.from_pretrained(MOLDEL_NAME)
    enc_di = tokenizer.batch_encode_plus(
        list(texts), 
        truncation=True,
        padding='longest',
        max_length=maxlen
    )
    return enc_di['input_ids'], enc_di['attention_mask']

def tokenize(df):
    num = 50000
    ids, att = encode_texts(df['text'].values[:num])
    for i in range(num, len(df), num):
        print(i)
        end_num = i + num
        if num + i > len(df):
            end_num = len(df)
        ids_temp, att_temp = encode_texts(df['text'].values[i:end_num])
        ids = np.vstack([ids, ids_temp])
        att = np.vstack([att, att_temp])
    return ids, att

def create_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32, name="ids")
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32, name="atttention_mask")

    bert_model = TFRobertaModel.from_pretrained(MOLDEL_NAME)
    x = bert_model(ids, attention_mask=att)[0]
    x = tf.squeeze(x[:, -1:, :], axis=1)
    outputs = tf.keras.layers.Dense(1, name='outputs')(x)

    model = tf.keras.Model(inputs=[ids, att], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='mse', metrics='mse')

    return model


def save_df(df, bert_vec, path):
    col_dict = {}
    for i in range(768):
        col_dict[i] = f'roberta_vec_{i}'
    df_result = pd.concat([df[['id']], pd.DataFrame(bert_vec).rename(columns=col_dict)], axis=1)
    df_result.to_pickle(path / '10_roberta_raw.pickle')


def main():
    print('start bert')
    model = create_model()
    print(model.summary())
    layer_name = 'tf_op_layer_Squeeze'
    hidden_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


    DATA_PATH = Path("./data/test/")
    df_test = pd.read_pickle(DATA_PATH / 'raw.pickle')
    df_test['text'] = df_test['title'] + df_test['abstract']
    ids, att = tokenize(df_test)
    bert_vec = hidden_layer_model.predict([ids, att], batch_size=256, verbose=1)
    save_df(df_test, bert_vec, DATA_PATH)

    DATA_PATH = Path("./data/train/")
    df_train = pd.read_pickle(DATA_PATH / 'raw.pickle')
    df_train['text'] = df_train['title'] + df_train['abstract']
    df_train = df_train[['id', 'text', 'doi_cites']]
    ids, att = tokenize(df_train)
    print(len(ids), print(att))

    bert_vec = hidden_layer_model.predict([ids, att], batch_size=256, verbose=1)
    save_df(df_train, bert_vec, DATA_PATH)
    



if __name__ == "__main__":
    main()