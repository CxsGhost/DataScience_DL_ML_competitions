from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import utils
import tensorflow as tf
import numpy as np
import pandas as pd


FEED_INFO_PATH = "my_data/my_info.csv"

my_feed_info = pd.read_csv(FEED_INFO_PATH, header=0, index_col=None)
feed_feature = my_feed_info.columns


author_data = my_feed_info['authorid'].values
videoplaysec_data = my_feed_info['videoplayseconds'].values
song_data = my_feed_info['bgm_song_id'].values
singer_data = my_feed_info['bgm_singer_id'].values
tag_data = my_feed_info['manual_tag_list'].values


i_author_input = layers.Input(shape=(None, ), dtype='int32', name='i_author_input')
dis_author_input = layers.Input(shape=(None, ), dtype='int32', name='dis_author_input')
recom_author_input = layers.Input(shape=(1, ), dtype='int32', name='recom_author_input')

author_encoder = preprocessing.IntegerLookup(oov_value=-1, mask_value=-2, name='author_lookup')
author_encoder.adapt(author_data.astype(np.int32))

i_author_enco = author_encoder(i_author_input)
dis_author_enco = author_encoder(dis_author_input)
recom_author_enco = author_encoder(recom_author_input)

au_so_si_pl_mask = layers.Masking(mask_value=0, name='au_so_si_pl_mask')

i_author_enco = au_so_si_pl_mask(i_author_enco)
dis_author_enco = au_so_si_pl_mask(dis_author_enco)

author_emb_layer = layers.Embedding(input_dim=len(author_encoder.get_vocabulary()),
                                    output_dim=32, name='author_embedding_layer')

i_author_emb = author_emb_layer(i_author_enco)
dis_author_emb = author_emb_layer(dis_author_enco)
recom_authot_emb = author_emb_layer(recom_author_enco)

Author_Embedding = models.Model(inputs=[i_author_input, dis_author_input, recom_author_input],
                                outputs=[i_author_emb, dis_author_emb, recom_authot_emb],
                                name='Author_Embedding')


i_song_input = layers.Input(shape=(None, ), dtype='int32', name='i_song_input')
dis_song_input = layers.Input(shape=(None, ), dtype='int32', name='dis_song_input')
recom_song_input = layers.Input(shape=(1, ), dtype='int32', name='recom_song_input')

song_encoder = preprocessing.IntegerLookup(oov_value=-1, mask_value=-2, name='song_lookup')
song_encoder.adapt(song_data.astype(np.int32))

i_song_enco = song_encoder(i_song_input)
dis_song_enco = song_encoder(dis_song_input)
recom_song_enco = song_encoder(recom_song_input)

i_song_enco = au_so_si_pl_mask(i_song_enco)
dis_song_enco = au_so_si_pl_mask(dis_song_enco)

song_emb_layer = layers.Embedding(input_dim=len(song_encoder.get_vocabulary()),
                                  output_dim=32, name='song_embedding_layer')

i_song_emb = song_emb_layer(i_song_enco)
dis_song_emb = song_emb_layer(dis_song_enco)
recom_song_emb = song_emb_layer(recom_song_enco)

Song_Embedding = models.Model(inputs=[i_song_input, dis_song_input, recom_song_input],
                              outputs=[i_song_emb, dis_song_emb, recom_song_emb],
                              name='Song_Embedding')


i_singer_input = layers.Input(shape=(None, ), dtype='int32', name='i_singer')
dis_singer_input = layers.Input(shape=(None, ), dtype='int32', name='dis_singer')
recom_singer_input = layers.Input(shape=(1, ), dtype='int32', name='recom_singer')

singer_encoder = preprocessing.IntegerLookup(oov_value=-1, mask_value=-2, name='singer_encoder')
singer_encoder.adapt(singer_data.astype(np.int32))

i_singer_enco = singer_encoder(i_singer_input)
dis_singer_enco = singer_encoder(dis_singer_input)
recom_singer_enco = singer_encoder(recom_singer_input)

i_singer_enco = au_so_si_pl_mask(i_singer_enco)
dis_singer_enco = au_so_si_pl_mask(dis_singer_enco)

singer_emb_layer = layers.Embedding(input_dim=len(singer_encoder.get_vocabulary()),
                                    output_dim=32, name='singer_embedding_layer')

i_singer_emb = singer_emb_layer(i_singer_enco)
dis_singer_emb = singer_emb_layer(dis_singer_enco)
recom_singer_emb = singer_emb_layer(recom_singer_enco)

Singer_Embedding = models.Model(inputs=[i_singer_input, dis_singer_input, recom_singer_input],
                                outputs=[i_singer_emb, dis_singer_emb, recom_singer_emb],
                                name='Singer_Embedding')


i_psec_input = layers.Input(shape=(None, ), dtype='int32', name='i_psec_input')
dis_psec_input = layers.Input(shape=(None, ), dtype='int32', name='dis_psec_input')
recom_psec_input = layers.Input(shape=(1, ), dtype='int32', name='recom_psec_input')

psec_encoder = preprocessing.IntegerLookup(oov_value=-1, mask_value=0, name='psec_encoder')
psec_encoder.adapt(videoplaysec_data.astype(np.int32))
print(psec_encoder.get_vocabulary())
print(len(psec_encoder.get_vocabulary()))

i_psec_enco = psec_encoder(i_psec_input)
dis_psec_enco = psec_encoder(dis_psec_input)
recom_psec_enco = psec_encoder(recom_psec_input)

i_psec_enco = au_so_si_pl_mask(i_psec_enco)
dis_psec_enco = au_so_si_pl_mask(dis_psec_enco)

psec_emb_layer =layers.Embedding(input_dim=len(psec_encoder.get_vocabulary()),
                                 output_dim=4, name='psec_embedding_layer')

i_psec_emb = psec_emb_layer(i_psec_enco)
dis_psec_emb = psec_emb_layer(dis_psec_enco)
recom_psec_emb = psec_emb_layer(recom_psec_enco)

Psec_Embedding = models.Model(inputs=[i_psec_input, dis_psec_input, recom_psec_input],
                              outputs=[i_psec_emb, dis_psec_emb, recom_psec_emb],
                              name='Psec_Embedding')
print(Psec_Embedding.summary())
print(Psec_Embedding.output_shape)
utils.plot_model(Psec_Embedding)


tag_token = preprocessing.TextVectorization(output_mode='binary')
tag_token.adapt(tag_data)
tag_vocabulary = tag_token.get_vocabulary()


i_tag_input = layers.Input(shape=(None, len(tag_vocabulary), ), dtype='int16', name='i_tag_input')
dis_tag_input = layers.Input(shape=(None, len(tag_vocabulary), ), dtype='int16', name='dis_tag_input')
recom_tag_input = layers.Input(shape=(1, len(tag_vocabulary), ), dtype='int16', name='recom_tag_input')

tag_mask = layers.Masking(mask_value=0, name='tag_mask')

i_tag_mask = tag_mask(i_tag_input)
dis_tag_mask = tag_mask(dis_tag_input)

tag_encoder = layers.TimeDistributed(layers.Dense(units=8, use_bias=False, name='tag_dense'), name='tag_encoder_layer')

i_tag_enco = tag_encoder(i_tag_mask)
dis_tag_enco = tag_encoder(dis_tag_mask)
recom_tag_enco = tag_encoder(recom_tag_input)

Tag_Encoder = models.Model(inputs=[i_tag_input, dis_tag_input, recom_tag_input],
                           outputs=[i_tag_enco, dis_tag_enco, recom_tag_enco],
                           name='Tag_Encoder')
print(Tag_Encoder.summary())
print(Tag_Encoder.output_shape)
utils.plot_model(Tag_Encoder)


ifeed_emb_input = layers.Input(shape=(None, 512, ), dtype='float32', name='ifeed_emb_input')
disfeed_emb_input = layers.Input(shape=(None, 512, ), dtype='float32', name='disfeed_emb_input')
recomfeed_emb_input = layers.Input(shape=(None, 512, ), dtype='float32', name='recomfeed_emb_input')


feature_concat_layer = layers.Concatenate(name='feature_concat_layer')

i_feed_vec = feature_concat_layer([Author_Embedding.outputs[0],
                                   Song_Embedding.outputs[0],
                                   Singer_Embedding.outputs[0],
                                   Psec_Embedding.outputs[0],
                                   Tag_Encoder.outputs[0],
                                   ifeed_emb_input])
dis_feed_vec = feature_concat_layer([Author_Embedding.outputs[1],
                                     Song_Embedding.outputs[1],
                                     Singer_Embedding.outputs[1],
                                     Psec_Embedding.outputs[1],
                                     Tag_Encoder.outputs[1],
                                     disfeed_emb_input])
recom_feed_vec = feature_concat_layer([Author_Embedding.outputs[2],
                                       Song_Embedding.outputs[2],
                                       Singer_Embedding.outputs[2],
                                       Psec_Embedding.outputs[2],
                                       Tag_Encoder.outputs[2],
                                       recomfeed_emb_input])
