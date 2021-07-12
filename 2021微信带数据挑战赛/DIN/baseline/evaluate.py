from tensorflow.keras import utils
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing import sequence
import my_model
import numpy as np
import pandas as pd
import tensorflow as tf


#%%

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

#%%

FEED_INFO = "../my-data/my_info_gene.csv"
I_HISTORY_DATA = "../pre-data/pre_history_i.csv"
DIS_HISTORY_DATA = "../pre-data/pre_history_dis.csv"
RECOM_DATA = "../pre-data/test_a.csv"
RECOM_TARGET = None
GENE_FEED_EMB = "../my-data/general_feed_emd.npy"

feed_info = pd.read_csv(FEED_INFO, header=0, index_col=None).values
all_feed_id = feed_info[:, 0].astype(np.int64)
i_history = pd.read_csv(I_HISTORY_DATA, header=0, index_col=None).groupby(by=['userid'])
dis_history = pd.read_csv(DIS_HISTORY_DATA, header=0, index_col=None).groupby(by=['userid'])
recom_data = pd.read_csv(RECOM_DATA, header=0, index_col=None).values


tset_recom = recom_data

#%%

i_maxlen = max(i_history.size().values)
dis_maxlen = max(dis_history.size().values)

new_feed_au_so_si_p = np.array([-2])
new_feed_tag = np.expand_dims(np.concatenate([np.ones(shape=(1, )), np.zeros(shape=(350, ))]), axis=0)
new_feed_emb = np.load("../my-data/general_feed_emd.npy")

non_ausosip = -2

non_tag = np.concatenate([np.array([1]), np.zeros(shape=(350, ))])

non_i_action = np.zeros(shape=(4, ))

#%%

def convert_emb(x):
    return np.array(x.replace('[', '').replace(']', '').split(','), dtype=np.float)


def convert_tag(x):
    return np.array(x.replace('[', '').replace(']', '').replace('\n', '').split('. '), dtype=np.float)


import numba as nb
@nb.njit(parallel=True)
def isin_nb(matrix, index_to_remove):
    # matrix and index_to_remove have to be numpy arrays
    # if index_to_remove is a list with different dtypes this
    # function will fail

    out = np.empty(matrix.shape[0], dtype=nb.boolean)
    index_to_remove_set = set(index_to_remove)

    for i in nb.prange(matrix.shape[0]):
        if matrix[i] in index_to_remove_set:
            out[i] = True
        else:
            out[i] = False
    return out

# 生成器
def generator(the_data, batch_size=8):
    i = 0
    while True:

        if i + batch_size > the_data.shape[0]:
            data = the_data[i:recom_data.shape[0]]
            i = 0
        else:
            data = the_data[i:i + batch_size]
            i += batch_size

        i_author = []
        i_song = []
        i_singer = []
        i_psec = []
        i_tag = []
        i_emb = []
        i_action = []

        dis_author = []
        dis_song = []
        dis_singer = []
        dis_psec = []
        dis_tag = []
        dis_emb = []

        recom_author = []
        recom_song = []
        recom_singer = []
        recom_psec = []
        recom_tag = []
        recom_emb = []

        device = utils.to_categorical(data[:, -1] - 1, num_classes=2)

        # 对于batch_size中的用户依次生成数据
        # 用try是因为，可能有未知视频，或者互动、不互动序列为空
        for u in range(data.shape[0]):
            user = data[u][0]
            recom_feed = data[u][1]

            try:
                recom_info = feed_info[all_feed_id == recom_feed][0]
                recom_author.append([recom_info[1]])
                recom_psec.append([recom_info[2]])
                recom_singer.append([recom_info[3]])
                recom_song.append([recom_info[4]])
                recom_emb.append([convert_emb(recom_info[5])])
                recom_tag.append([convert_tag(recom_info[6])])
            except:
                recom_author.append([non_ausosip])
                recom_psec.append([non_ausosip])
                recom_singer.append([non_ausosip])
                recom_song.append([non_ausosip])
                recom_emb.append([new_feed_emb])
                recom_tag.append([non_tag])

            try:
                dis_feed = dis_history.get_group(user).values[:, 1]
                temp = feed_info[isin_nb(all_feed_id, dis_feed)]
                dis_author.append(temp[:, 1])
                dis_psec.append(temp[:, 2])
                dis_singer.append(temp[:, 3])
                dis_song.append(temp[:, 4])
                dis_emb.append(list(map(convert_emb, temp[:, 5])))
                dis_tag.append(list(map(convert_tag, temp[:, 6])))
            except:
                dis_author.append([non_ausosip])
                dis_psec.append([non_ausosip])
                dis_singer.append([non_ausosip])
                dis_song.append([non_ausosip])
                dis_emb.append([new_feed_emb])
                dis_tag.append([non_tag])

            try:
                i_feed = i_history.get_group(user).values[:, 1]
                i_action.append(i_history.get_group(user).values[:, 2:])
                temp = feed_info[isin_nb(all_feed_id, i_feed)]
                i_author.append(temp[:, 1])
                i_psec.append(temp[:, 2])
                i_singer.append(temp[:, 3])
                i_song.append(temp[:, 4])
                i_emb.append(list(map(convert_emb, temp[:, 5])))
                i_tag.append(list(map(convert_tag, temp[:, 6])))
            except:
                i_author.append([non_ausosip])
                i_psec.append([non_ausosip])
                i_singer.append([non_ausosip])
                i_song.append([non_ausosip])
                i_emb.append([new_feed_emb])
                i_tag.append([non_tag])
                i_action.append([non_i_action])

        # 最后将所有不长序列填充为定长序列，需要词典索引后emb的用-1，不需要的直接0
        yield [sequence.pad_sequences(i_author, value=-1, padding='post', maxlen=i_maxlen),
               sequence.pad_sequences(dis_author, value=-1, padding='post', maxlen=dis_maxlen),
               np.array(recom_author),
               sequence.pad_sequences(i_song, value=-1, padding='post', maxlen=i_maxlen),
               sequence.pad_sequences(dis_song, value=-1, padding='post', maxlen=dis_maxlen),
               np.array(recom_song),
               sequence.pad_sequences(i_singer, value=-1, padding='post', maxlen=i_maxlen),
               sequence.pad_sequences(dis_singer, value=-1, padding='post', maxlen=dis_maxlen),
               np.array(recom_singer),
               sequence.pad_sequences(i_psec, value=-1, padding='post', maxlen=i_maxlen),
               sequence.pad_sequences(dis_psec, value=-1, padding='post', maxlen=dis_maxlen),
               np.array(recom_psec),
               sequence.pad_sequences(i_tag, value=0, padding='post', maxlen=i_maxlen),
               sequence.pad_sequences(dis_tag, value=0, padding='post', maxlen=dis_maxlen),
               np.array(recom_tag),
               sequence.pad_sequences(i_emb, value=0, padding='post', maxlen=i_maxlen),
               sequence.pad_sequences(dis_emb, value=0, padding='post', maxlen=dis_maxlen),
               np.array(recom_emb),
               sequence.pad_sequences(i_action, value=0, padding='post', maxlen=i_maxlen),
               device]


model_bn_true = tf.keras.models.load_model("./model/my_model_2")

#%%

print(model_bn_true.summary())
BATCH_SIZE = 64
tset_gene = generator(recom_data, batch_size=BATCH_SIZE)

#%%

test_stes = int(recom_data.shape[0] / BATCH_SIZE) + 1
test_target = model_bn_true.predict(tset_gene, steps=test_stes, verbose=1,
                                       max_queue_size=20)

#%%

temp_1 = pd.DataFrame(recom_data[:, :2], columns=['userid', 'feedid'])
temp_2 = pd.DataFrame(test_target, columns=['read_comment', 'like', 'click_avatar', 'forward'])

#%%

submit = pd.concat([temp_1.reset_index(drop=True), temp_2.reset_index(drop=True)], axis=1)
submit.to_csv("./submit/bn_all_submit.csv", index=None)