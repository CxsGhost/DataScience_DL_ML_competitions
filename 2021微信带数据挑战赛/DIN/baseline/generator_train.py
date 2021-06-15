#%%


from tensorflow.keras import utils
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing import sequence
import my_model
import numpy as np
import pandas as pd

#%%

FEED_INFO = "./data_14/my_info_gene.csv"
I_HISTORY_DATA = "./data_14/train_history_i.csv"
DIS_HISTORY_DATA = "./data_14/train_history_dis.csv"
RECOM_DATA = "./data_14/train_recommend_2.csv"
RECOM_TARGET = "./data_14/train_target_2.csv"
GENE_FEED_EMB = "./data_14/general_feed_emd.npy"

# 因为每条推荐视频都要生成关于此用户的互动和不互动序列，太大，所以用生成器
feed_info = pd.read_csv(FEED_INFO, header=0, index_col=None).groupby(by=['feedid'])
i_history = pd.read_csv(I_HISTORY_DATA, header=0, index_col=None).groupby(by=['userid'])
dis_history = pd.read_csv(DIS_HISTORY_DATA, header=0, index_col=None).groupby(by=['userid'])
recom_data = pd.read_csv(RECOM_DATA, header=0, index_col=None).values
recom_target = pd.read_csv(RECOM_TARGET, header=0, index_col=None).values

# test_quantity = int(len(recom_data) * 0.2)
# test_recom = recom_data[-test_quantity:]
# test_target = recom_target[-test_quantity:]

# train_recom = recom_data[:test_quantity]
# train_target = recom_target[:test_quantity]
train_recom = recom_data
train_target = recom_target
#%%

# 互动和不互动序列的填充长度按照最长的进行
i_maxlen = max(i_history.size().values)
dis_maxlen = max(dis_history.size().values)

# 新视频未知视频的填充数据，主要是为了提交阶段，其中feed_embedding用全部的均值来代替（我自己瞎想的）
new_feed_au_so_si_p = np.array([-2])
new_feed_tag = np.expand_dims(np.concatenate([np.ones(shape=(1, )), np.zeros(shape=(350, ))]), axis=0)
new_feed_emb = np.load("../my_data/train_14/general_feed_emd.npy")

# 假如某个用户互动或不互动序列为空时，填充
non_ausosip = -2

non_tag = np.concatenate([np.array([1]), np.zeros(shape=(350, ))])

non_i_action = np.zeros(shape=(4, ))

#%%

# csv读出来的东西不能直接用，需要转化一下
def convert_emb(x):
    return np.array(x.replace('[', '').replace(']', '').split(','), dtype=np.float)


def convert_tag(x):
    return np.array(x.replace('[', '').replace(']', '').replace('\n', '').split('. '), dtype=np.float)


# 生成器
def generator(the_data, the_target, batch_size=8):
    i = 0
    while True:
        i_author = [[] for _ in range(batch_size)]
        i_song = [[] for _ in range(batch_size)]
        i_singer = [[] for _ in range(batch_size)]
        i_psec = [[] for _ in range(batch_size)]
        i_tag = [[] for _ in range(batch_size)]
        i_emb = [[] for _ in range(batch_size)]
        i_action = []

        dis_author = [[] for _ in range(batch_size)]
        dis_song = [[] for _ in range(batch_size)]
        dis_singer = [[] for _ in range(batch_size)]
        dis_psec = [[] for _ in range(batch_size)]
        dis_tag = [[] for _ in range(batch_size)]
        dis_emb = [[] for _ in range(batch_size)]

        recom_author = []
        recom_song = []
        recom_singer = []
        recom_psec = []
        recom_tag = []
        recom_emb = []

        # 检查是否已经一个epoch
        if i + batch_size > the_data.shape[0]:
            data = the_data[i:recom_data.shape[0]]
            target = the_target[i:recom_data.shape[0]]
            i = 0
        else:
            data = the_data[i:i+batch_size]
            target = the_target[i:i+batch_size]
            i += batch_size

        # 设备信息noe-hot放入
        device = utils.to_categorical(data[:, -1] - 1, num_classes=2)

        # 对于batch_size中的用户依次生成数据
        # 用try是因为，可能有未知视频，或者互动、不互动序列为空
        for u in range(batch_size):
            user = data[u][0]
            recom_feed = data[u][1]

            try:
                recom_info = feed_info.get_group(recom_feed).values[0]
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
                for f in dis_feed:
                    temp = feed_info.get_group(f).values[0]
                    dis_author[u].append(temp[1])
                    dis_psec[u].append(temp[2])
                    dis_singer[u].append(temp[3])
                    dis_song[u].append(temp[4])
                    dis_emb[u].append(convert_emb(temp[5]))
                    dis_tag[u].append(convert_tag(temp[6]))
            except:
                dis_author[u].append(non_ausosip)
                dis_psec[u].append(non_ausosip)
                dis_singer[u].append(non_ausosip)
                dis_song[u].append(non_ausosip)
                dis_emb[u].append(new_feed_emb)
                dis_tag[u].append(non_tag)

            try:
                i_feed = i_history.get_group(user).values[:, 1]
                i_action.append(i_history.get_group(user).values[:, 2:])
                for f in i_feed:
                    temp = feed_info.get_group(f).values[0]
                    i_author[u].append(temp[1])
                    i_psec[u].append(temp[2])
                    i_singer[u].append(temp[3])
                    i_song[u].append(temp[4])
                    i_emb[u].append(convert_emb(temp[5]))
                    i_tag[u].append(convert_tag(temp[6]))
            except:
                i_author[u].append(non_ausosip)
                i_psec[u].append(non_ausosip)
                i_singer[u].append(non_ausosip)
                i_song[u].append(non_ausosip)
                i_emb[u].append(new_feed_emb)
                i_tag[u].append(non_tag)
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
               device], target

#%%

# 批量大小
BATCH_SIZE = 32
test_BATCH_SIZE = 32

# 生成器
train_gene = generator(train_recom, train_target, batch_size=BATCH_SIZE)
# test_gene = generator(test_recom, test_target, batch_size=test_BATCH_SIZE)

#%%

# 模型
the_model = my_model.get_model(FEED_INFO)

#%%

# 生成器参数
steps_per_epoch = int(train_recom.shape[0] / BATCH_SIZE)
# valid_steps = int(test_recom.shape[0] / test_BATCH_SIZE)

# 拟合
the_model.fit(train_gene,
              epochs=1,
              steps_per_epoch=steps_per_epoch)
              # validation_data=test_gene,
              # validation_steps=valid_steps,
              # validation_freq=1)
