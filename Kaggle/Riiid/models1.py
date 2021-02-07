import numpy as np
import pandas as pd
import psutil

'''
Riiid Competition Submission Ver 1.0.1 Alpha
(C) Copyright By Author 2020 - Now
All rights reserved
'''
import sys

sys.path.append('/kaggle/input/riiid-dataset/')
# 路径
question_metadata_dir = r'/kaggle/input/riiid-dataset/question_metadata.csv'
lesson_metadata_dir = r'/kaggle/input/riiid-dataset/lesson_metadata.csv'
pickle_dir = r'/kaggle/input/riiid-dataset/stage.pickle'
model_dir = r'/kaggle/input/riiid-dataset/classifier.model'

import datetime

print("{} 提交启动".format(str(datetime.datetime.now())))
# 加载库
try:
    import pandas as pd
    import pickle
    import trueskill
    import math
    import lightgbm as lgb
    import riiideducation
    import time
    from sklearn.metrics import roc_auc_score


except ImportError as e:
    print("{} 导入错误，错误信息：{}".format(str(datetime.datetime.now()), e))

print("{} 包导入完成".format(str(datetime.datetime.now())))
env = trueskill.TrueSkill(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164, draw_probability=0)
env.make_as_global()


def win_probability(team1, team2):
    '''
    根据两个TrueSkill对象，计算获胜概率
    :param team1:用户TrueSkill对象
    :param team2:问题Trueskill对象
    :return: 获胜概率
    '''
    delta_mu = team1.mu - team2.mu
    sum_sigma = sum([team1.sigma ** 2, team2.sigma ** 2])
    size = 2
    denom = math.sqrt(size * (0.05 * 0.05) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


class user:
    '''
    用户 类
    '''

    def __init__(self):
        '''
        初始化user类
        :param None
        :return: None
        '''
        # 直接可输出的特征

        # 数量类
        self.question_answered_num = 0  # 用户回答问题的总数量
        self.question_answered_num_agg_field = [0] * 7  # 用户总回答的问题(按TOEIC学科领域统计)

        # 正确率类
        self.question_answered_mean_accuracy = 0  # 用户回答的问题的平均正确率
        self.question_answered_mean_accuracy_agg_field = [0] * 7  # 用户总回答的问题的平均正确率
        self.question_answered_mean_difficulty_weighted_accuracy = 0  # 用户总回答的问题的平均难度加权正确率
        self.question_answered_mean_difficulty_weighted_accuracy_agg_field = [0] * 7  # 用户总回答的问题的平均难度加权正确率(按TOEIC学科领域统计)

        # 极值类
        self.max_solved_difficulty = 1  # 用户解答的最难问题
        self.max_solved_difficulty_agg_field = [1] * 7  # 用户解答的最难问题(按TOEIC学科领域统计)
        self.min_wrong_difficulty = 0  # 用户做错的最简单问题
        self.min_wrong_difficulty_agg_field = [0] * 7  # 用户做错的最简单问题

        # 课程学习类
        self.lessons_overall = 0  # 用户总共学了多少课
        self.lessons_overall_agg_field = [0] * 7  # 用户总共学了多少课（按TOEIC学科领域统计）

        # 交互时间信息类
        self.session_time = 0  # 用户本Session的分钟数
        self.since_last_session_time = 0  # 距离上次Session的小时数

        # 需要进一步处理的特征
        self._mmr_object = trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
                                           draw_probability=0).Rating()  # MMR模块
        self._mmr_object_agg_field = [trueskill.setup(mu=0.3, sigma=0.164486, beta=0.05, tau=0.00164,
                                                      draw_probability=0).Rating()] * 7  # MMR模块（按TOEIC学科领域统计）
        self._most_liked_guess = [0] * 4  # 用户做错时最喜欢的选项
        self._last_session_start_time = 0  # 本Session开始的时间
        self._first_action_time = 0  # 首次交互的时间
        self._question_num_dict = {}  # 用户回答问题的记录
        self._first_processed_flag = False  # 是否处理的表示

    def update_user(self, data: pd.DataFrame):
        '''
        处理一帧测试集
        :param data: pandas DataFrame
        :return: None
        '''
        _temp = None

        # 判断用户是否正在观看课程
        if data['content_type_id'] == 0:
            # Content Type 为 0，即用户正在回答问题

            # 处理回答计数部分
            self.question_answered_num = self.question_answered_num + 1
            question_field = int(data['content_field'])
            self.question_answered_num_agg_field[question_field - 1] = int(self.question_answered_num_agg_field[
                                                                               question_field - 1]) + 1

            # 处理正确率部分
            if data['answered_correctly'] == 1:
                self.question_answered_mean_accuracy = \
                    (self.question_answered_mean_accuracy * (
                            self.question_answered_num - 1) + 1) / self.question_answered_num

                self.question_answered_mean_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1) + 1) \
                    / self.question_answered_num_agg_field[question_field - 1]

                self.question_answered_mean_difficulty_weighted_accuracy = \
                    (self.question_answered_mean_difficulty_weighted_accuracy * (self.question_answered_num - 1) + (
                            1 - data['mean_question_accuracy']) * 3) \
                    / self.question_answered_num

                self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1) + (
                             1 - data['mean_question_accuracy']) * 3) \
                    / self.question_answered_num_agg_field[question_field - 1]


            else:
                self.question_answered_mean_accuracy = \
                    (self.question_answered_mean_accuracy * (
                            self.question_answered_num - 1)) / self.question_answered_num

                self.question_answered_mean_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1)) / \
                    self.question_answered_num_agg_field[question_field - 1]

                self.question_answered_mean_difficulty_weighted_accuracy = \
                    (self.question_answered_mean_difficulty_weighted_accuracy * (self.question_answered_num - 1)) \
                    / self.question_answered_num

                self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] = \
                    (self.question_answered_mean_difficulty_weighted_accuracy_agg_field[question_field - 1] * (
                            self.question_answered_num_agg_field[question_field - 1] - 1)) \
                    / self.question_answered_num_agg_field[question_field - 1]

            # 处理最大/最小正确率部分

            if data['answered_correctly'] == 1:
                if data['mean_question_accuracy'] < self.max_solved_difficulty:
                    self.max_solved_difficulty = data['mean_question_accuracy']
                if data['mean_question_accuracy'] < self.max_solved_difficulty_agg_field[question_field - 1]:
                    self.max_solved_difficulty_agg_field[question_field - 1] = data['mean_question_accuracy']
            else:
                if data['mean_question_accuracy'] > self.min_wrong_difficulty:
                    self.min_wrong_difficulty = data['mean_question_accuracy']
                if data['mean_question_accuracy'] > self.min_wrong_difficulty_agg_field[question_field - 1]:
                    self.min_wrong_difficulty_agg_field[question_field - 1] = data['mean_question_accuracy']

            # 处理猜测部分
            if data['answered_correctly'] == 0:
                self._most_liked_guess[int(data['user_answer'])] = self._most_liked_guess[
                                                                       int(data['user_answer'])] + 1

            # 处理时间部分
            if self._first_action_time == 0:
                self._first_action_time = data['timestamp']
                self._last_session_start_time = data['timestamp']
            else:
                if data['timestamp'] - self._last_session_start_time >= 7200 * 1000:
                    self.since_last_session_time = (data[
                                                        'timestamp'] - self._last_session_start_time) / 1000 / 3600
                    self._last_session_start_time = data['timestamp']
                    self.session_time = 0
                else:
                    self.session_time = (data['timestamp'] - self._last_session_start_time) / 1000 / 60

            # 处理问题记录部分
            if str(data['content_id']) in self._question_num_dict:
                self._question_num_dict[str(data['content_id'])] = self._question_num_dict[str(data['content_id'])] + 1
            else:
                self._question_num_dict[str(data['content_id'])] = 1

            # 处理TrueSkill部分
            if data['answered_correctly'] == 1:
                self._mmr_object, _temp = \
                    trueskill.rate_1vs1(self._mmr_object,
                                        trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05, tau=0.00164, draw_probability=0).Rating())
                self._mmr_object_agg_field[question_field - 1], _temp = \
                    trueskill.rate_1vs1(self._mmr_object_agg_field[question_field - 1],
                                        trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05,
                                                        tau=0.00164, draw_probability=0).Rating())
            else:
                _temp, self._mmr_object = \
                    trueskill.rate_1vs1(trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05, tau=0.00164, draw_probability=0).Rating(),
                                        self._mmr_object)

                _temp, self._mmr_object_agg_field[question_field - 1] = \
                    trueskill.rate_1vs1(trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486,
                                                        beta=0.05,
                                                        tau=0.00164, draw_probability=0).Rating(),
                                        self._mmr_object_agg_field[question_field - 1])



        else:
            # Content Type 不为 0 ，即用户在观看视频

            self.lessons_overall = self.lessons_overall + 1
            lesson_field = int(data['content_field'])
            self.lessons_overall_agg_field[lesson_field - 1] = self.lessons_overall_agg_field[lesson_field - 1] + 1

    def process_output(self, data):
        '''
        根据user现有属性设置输出训练数据
        :param data: 本行数据集
        :return: output_dict 训练数据
        '''
        output_dict = {}

        # 回答数量类
        output_dict['question_answered_num'] = self.question_answered_num
        output_dict['question_answered_num_agg_field'] = self.question_answered_num_agg_field[
            int(data['content_field']) - 1]

        # 回答正确率类
        output_dict['question_answered_mean_accuracy'] = self.question_answered_mean_accuracy

        output_dict['question_answered_mean_accuracy_agg_field'] = self.question_answered_mean_accuracy_agg_field[
            int(data['content_field']) - 1]
        output_dict[
            'question_answered_mean_difficulty_weighted_accuracy'] = self.question_answered_mean_difficulty_weighted_accuracy
        output_dict['question_answered_mean_difficulty_weighted_accuracy_agg_field'] = \
            self.question_answered_mean_difficulty_weighted_accuracy_agg_field[int(data['content_field']) - 1]

        # 极值类

        output_dict['max_solved_difficulty'] = self.max_solved_difficulty
        output_dict['max_solved_difficulty_agg_field'] = self.max_solved_difficulty_agg_field[
            int(data['content_field']) - 1]
        output_dict['min_wrong_difficulty'] = self.min_wrong_difficulty
        output_dict['min_wrong_difficulty_agg_field'] = self.min_wrong_difficulty_agg_field[
            int(data['content_field']) - 1]

        # 课程学习类
        output_dict['lessons_overall'] = self.lessons_overall
        output_dict['lessons_overall_agg_field'] = self.lessons_overall_agg_field[int(data['content_field']) - 1]
        if output_dict['lessons_overall_agg_field'] > 0:
            output_dict['field_learnt'] = 1
        else:
            output_dict['field_learnt'] = 0
        # 交互时间类
        output_dict['session_time'] = self.session_time
        output_dict['time_to_last_session'] = self.since_last_session_time

        output_dict['task_id'] = data['task_container_id']
        output_dict['prior_time'] = data['prior_question_elapsed_time']
        # 问题信息类
        output_dict['mean_question_accuracy'] = data['mean_question_accuracy']
        output_dict['std_question_accuracy'] = data['std_accuracy']
        output_dict['question_id'] = data['content_id']
        # TrueSkill 信息类
        output_dict['mmr_overall'] = self._mmr_object.mu
        output_dict['mmr_overall_agg_field'] = self._mmr_object_agg_field[int(data['content_field']) - 1].mu
        output_dict['mmr_confidence'] = self._mmr_object.sigma

        output_dict['mmr_overall_agg_field'] = self._mmr_object_agg_field[int(data['content_field']) - 1].sigma
        output_dict['mmr_win_prob'] = win_probability(self._mmr_object,
                                                      trueskill.setup(mu=1 - data['mean_question_accuracy'],
                                                                      sigma=0.164486,
                                                                      beta=0.05, tau=0.00164,
                                                                      draw_probability=0).Rating())
        output_dict['mmr_win_prob_agg_field'] = win_probability(
            self._mmr_object_agg_field[int(data['content_field']) - 1],
            trueskill.setup(mu=1 - data['mean_question_accuracy'], sigma=0.164486, beta=0.05,
                            tau=0.00164, draw_probability=0).Rating())
        output_dict['user_id'] = data['user_id']
        output_dict['tag_1'] = data['tag_1']
        output_dict['tag_2'] = data['tag_2']

        output_dict['tags_encoded'] = data['tags_encoded']
        # 特殊特征类

        if not pd.isna(['prior_question_had_explanation']):
            output_dict['previous_explained'] = data['prior_question_had_explanation']
        else:
            output_dict['previous_explained'] = False

        if str(data['content_id']) in self._question_num_dict:
            output_dict['question_seen'] = 1
        else:
            output_dict['question_seen'] = 0

        # 猜测类
        max_choice = 0
        max_choice_num = 0
        i = 0
        for item in self._most_liked_guess:
            if item > max_choice_num:
                max_choice_num = item
                max_choice = i
            i = i + 1

        if output_dict['mmr_win_prob'] <= 0.4:
            if max_choice == data['correct_answer']:
                output_dict['most_liked_guess_correct'] = True
            else:
                output_dict['most_liked_guess_correct'] = False
        else:
            output_dict['most_liked_guess_correct'] = True

        # 训练目标
        # output_dict['answered_correctly'] = data['answered_correctly']

        return output_dict


# 导入Metadata
question_metadata = pd.read_csv(question_metadata_dir)
lesson_metadata = pd.read_csv(lesson_metadata_dir)
print("{} Metadata 文件导入完成".format(str(datetime.datetime.now())))

# 设置Metadata索引
question_metadata = question_metadata.set_index(keys=['content_id'])
lesson_metadata = lesson_metadata.set_index(keys=['content_id'])
print("{} Metadata 索引设置完成".format(str(datetime.datetime.now())))

# 导入Pickle状态
with open(pickle_dir, 'rb') as fo:
    user_pickle = pickle.load(fo)

print("{} Pickle 导入完成".format(str(datetime.datetime.now())))

# 重置Trueskill状态
for user_id, user_info in user_pickle.items():
    user_pickle[user_id]._mmr_object = trueskill.setup(mu=user_pickle[user_id]._mmr_object[0],
                                                       sigma=user_pickle[user_id]._mmr_object[1],
                                                       beta=0.05, tau=0.00164,
                                                       draw_probability=0).Rating()
    for i in range(0, 7):
        # 1+1
        user_pickle[user_id]._mmr_object_agg_field[i] = trueskill.setup(
            mu=user_pickle[user_id]._mmr_object_agg_field[i][0],
            sigma=user_pickle[user_id]._mmr_object_agg_field[i][1],
            beta=0.05, tau=0.00164,
            draw_probability=0).Rating()

print("{} Pickle Trueskill状态重置完成".format(str(datetime.datetime.now())))

# 导入模型
model = lgb.Booster(model_file=model_dir)
print("{} 模型导入完成".format(str(datetime.datetime.now())))

MAX_SEQ = 100
import torch, joblib
import psutil
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=MAX_SEQ, embed_dim=128):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill + 1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.3)

        self.dropout = nn.Dropout(0.3)
        self.layer_normal = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight


skills = joblib.load("/kaggle/input/riiid-sakt-model-dataset-public/skills.pkl.zip")
n_skill = len(skills)
group = joblib.load("/kaggle/input/riiid-sakt-model-dataset-public/group.pkl.zip")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_SAKT = SAKTModel(n_skill, embed_dim=128)
try:
    model_SAKT.load_state_dict(torch.load("/kaggle/input/riiid-sakt-model-dataset-public/sakt_model.pt"))
except:
    model_SAKT.load_state_dict(
        torch.load("/kaggle/input/riiid-sakt-model-dataset-public/sakt_model.pt", map_location='cpu'))
model_SAKT.to(device)
model_SAKT.eval()


class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, max_seq=MAX_SEQ):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_ = self.samples[user_id]

            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill

        questions = np.append(q[2:], [target_id])

        return x, questions


prev_test_df = None

env = riiideducation.make_env()
iter_test = env.iter_test()
prev_test_df = None

print("{} 比赛环境设置完成".format(str(datetime.datetime.now())))

# 初始换变量
rows_accum = 0  # 行计数器
first_submission = True  # 是否第一组标记
model_prd = [0]
true_value = []
last_df = pd.DataFrame()
print("{} 比赛变量设置完成".format(str(datetime.datetime.now())))

for (test_df, sample_prediction_df) in iter_test:

    cur = (test_df, sample_prediction_df)

    if (prev_test_df is not None) & (psutil.virtual_memory().percent < 90):
        prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
        prev_test_df = prev_test_df[prev_test_df.content_type_id == False]

        prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))
        for prev_user_id in prev_group.index:
            if prev_user_id in group.index:
                group[prev_user_id] = (
                    np.append(group[prev_user_id][0], prev_group[prev_user_id][0])[-MAX_SEQ:],
                    np.append(group[prev_user_id][1], prev_group[prev_user_id][1])[-MAX_SEQ:]
                )

            else:
                group[prev_user_id] = (
                    prev_group[prev_user_id][0],
                    prev_group[prev_user_id][1]
                )

    prev_test_df = test_df.copy()

    test_df = test_df[test_df.content_type_id == False]
    TEST = test_df[test_df.content_type_id == False]
    test_dataset = TestDataset(group, test_df, skills)
    test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)

    outs = []

    for item in test_dataloader:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()

        with torch.no_grad():
            output, att_weight = model_SAKT(x, target_id)
        outs.extend(torch.sigmoid(output)[:, -1].view(-1).data.cpu().numpy())

    TEST['answered_correctly'] = outs
    TEST = TEST.loc[TEST['content_type_id'] == 0, ['row_id', 'answered_correctly']]

    (test_df, sample_prediction_df) = cur

    if first_submission == False:
        last_df['answered_correctly'] = eval(test_df.iloc[0]['prior_group_answers_correct'])
        last_df['user_answer'] = eval(test_df.iloc[0]['prior_group_responses'])
        true_value.extend(eval(test_df.iloc[0]['prior_group_answers_correct']))
        for index, row in last_df.iterrows():
            user_pickle[row['user_id']].update_user(row)
    rows_accum = rows_accum + test_df.shape[0]
    if first_submission == False:
        1 + 1
        # print("{} 当前正在处理第 {} 行 , 截至目前AUC为 {}".format(str(datetime.datetime.now()),rows_accum,roc_auc_score(true_value,model_prd)))
    test_df['answered_correctly'] = 0.6524
    st = float(time.time())
    # 完成Merge 与 Concat 工作
    try:
        sub_1 = test_df[test_df['content_type_id'] == False]
        sub_2 = test_df[test_df['content_type_id'] == True]
        del test_df
        sub_1 = sub_1.merge(question_metadata, on="content_id", how="left")
        sub_2 = sub_2.merge(lesson_metadata, on="content_id", how="left")
        test_df = pd.DataFrame()
        test_df = pd.concat([sub_1, sub_2])
    except Exception:
        pass

    for index, row in test_df.iterrows():
        # print(row.row_id)
        try:
            if row['user_id'] not in user_pickle:
                user_pickle[row['user_id']] = user()
            if row['content_type_id'] == 0:
                predict_dict = user_pickle[row['user_id']].process_output(row)
                l = []
                for i, v in predict_dict.items():
                    l.append(v)
                prd_value = float(model.predict([l])[0])
                test_df.loc[test_df.row_id == row.row_id, 'answered_correctly'] = 0.9 * prd_value + float(
                    TEST[TEST['row_id'] == row.row_id]['answered_correctly']) * 0.1
                model_prd.append(prd_value)


        except Exception as e:
            print(e)
            pass

    time_taken = float(time.time()) - st
    print("{} 基于当前速率，共需要 {} 分钟完成预测".format(
        str(datetime.datetime.now()), int(time_taken / test_df.shape[0] * 2500000 / 60)))
    if first_submission == True:
        first_submission = False
    last_df = test_df
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

float(TEST[TEST['row_id'] == row.row_id]['answered_correctly'] * 0.5)

test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']]

TEST

