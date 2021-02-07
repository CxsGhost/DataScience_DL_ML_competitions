import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import copy


delivery_pos = np.array([[0, 0], [3, 2], [1, 5], [5, 4], [4, 7], [0, 8], [3, 11],
                         [7, 9], [9, 6], [10, 2], [14, 0], [17, 3], [14, 6],
                         [12, 9], [10, 12], [7, 14], [2, 16], [6, 18], [11, 17],
                         [15, 12], [19, 9], [22, 5], [21, 0], [27, 9], [15, 19],
                         [15, 14], [20, 17], [21, 13], [24, 20], [25, 16], [28, 18]])
pos_weight = np.array([8, 8.2, 6, 5.5, 3, 4.5, 7.2, 2.3, 1.4, 6.5, 4.1, 12.7, 5.8, 3.8, 4.6,
                       3.5, 5.8, 7.5, 7.8, 3.4, 6.2, 6.8, 2.4, 7.6, 9.6, 10, 12, 6, 8.1, 4.2])

max_weight = 25
max_time = 360
v = 25

# 构造距离矩阵
distance_mat = np.zeros(shape=(delivery_pos.shape[0], delivery_pos.shape[0]))
for i in range(delivery_pos.shape[0]):
    for j in range(i, delivery_pos.shape[0]):
        if i == j:
            distance_mat[i, j] = np.inf
        else:
            distance_mat[i, j] = np.sum(np.absolute(delivery_pos[i][0] - delivery_pos[j][0]) +
                                        np.absolute(delivery_pos[i][1] - delivery_pos[j][1]))
            distance_mat[j, i] = distance_mat[i, j]
distance_mat = np.array(distance_mat)

# 计算路径货物总重量是否超标
def sum_weight(seq_path_,zero_pos_):
    weight_list_ = []
    excess_list_ = []
    seq_path_ = np.array(seq_path_, dtype=np.int)
    zero_pos_ = np.array(zero_pos_, dtype=np.int)
    path_ = seq_path_[0: zero_pos_[0]]
    if len(path_):
        weight_list_.append(np.sum(pos_weight[path_ - 1]))
        sum_ = 0
        for j_ in range(len(path_)):
            sum_ += pos_weight[path_[j_] - 1]
            if sum_ > max_weight:
                excess_list_.append([path_[j_ - 1], path_[j_]])
                sum_ = pos_weight[path_[j_] - 1]
    for i_ in range(len(zero_pos_) - 1):
        path_ = seq_path_[zero_pos_[i_]: zero_pos_[i_ + 1]]
        if not len(path_):
            continue
        sum_ = 0
        for j_ in range(len(path_)):
            sum_ += pos_weight[path_[j_] - 1]
            if sum_ > max_weight:
                excess_list_.append([path_[j_ - 1], path_[j_]])
                sum_= pos_weight[path_[j_] - 1]
        weight_list_.append(np.sum(pos_weight[path_ - 1]))
    path_ = seq_path_[zero_pos_[-1]:]
    if len(path_):
        sum_ = 0
        weight_list_.append(np.sum(pos_weight[path_ - 1]))
        for j_ in range(len(path_)):
            sum_ += pos_weight[path_[j_] - 1]
            if sum_ > max_weight:
                excess_list_.append([path_[j_ - 1], path_[j_]])
                sum_= pos_weight[path_[j_] - 1]
    return [1, weight_list_, excess_list_]
    # if np.max(weight_list_) > max_weight:
    #     return [0]
    # else:
    #     return [1, weight_list_]


# 计算路径工作时长是否合格
def sum_time(seq_path_, zero_pos_):
    seq_path_ = np.array(seq_path_, dtype=np.int)
    zero_pos_ = np.array(zero_pos_, dtype=np.int)
    time_list_ = []
    path_ = seq_path_[0: zero_pos_[0]]
    if len(path_):
        if len(path_) == 1:
            time_ = (distance_mat[0][path_[0]] * 2) / 25 * 60 + len(path_) * 8
            time_list_.append(time_)
        else:
            way_ = 0
            for i_ in range(len(path_) - 1):
                way_ += distance_mat[i_][i_ + 1]
            way_ += distance_mat[0][path_[0]] + distance_mat[path_[-1]][0]
            time_ = way_ / v * 60 + len(path_) * 8
            time_list_.append(time_)
    for i_ in range(len(zero_pos_) - 1):
        way_ = 0
        path_ = seq_path_[zero_pos_[i_]: zero_pos_[i_ + 1]]
        if len(path_):
            if len(path_) == 1:
                time_ = (distance_mat[0][path_[0]] * 2) / 25 * 60 + len(path_) * 8
                time_list_.append(time_)
            else:
                for j_ in range(len(path_) - 1):
                    way_ += distance_mat[i_][i_ + 1]
                way_ += distance_mat[0][path_[0]] + distance_mat[path_[-1]][0]
                time_ = way_ / v * 60 + len(path_) * 8
                time_list_.append(time_)
    path_ = seq_path_[zero_pos_[-1]:]
    if len(path_):
        if len(path_) == 1:
            time_ = (distance_mat[0][path_[0]] * 2) / 25 * 60 + len(path_) * 8
            time_list_.append(time_)
        else:
            way_ = 0
            for i_ in range(len(path_) - 1):
                way_ += distance_mat[i_][i_ + 1]
            way_ += distance_mat[0][path_[0]] + distance_mat[path_[-1]][0]
            time_ = way_ / v * 60 + len(path_) * 8
            time_list_.append(time_)
    if np.max(time_list_) > max_time:
        return [0]
    else:
        return [1, time_list_]


# 二位交换法
def exchange_pos(seq_path_):
    pos_list = range(len(seq_path_))
    pos_ = random.sample(pos_list, 2)
    seq_path_[pos_[0]], seq_path_[pos_[1]] = seq_path_[pos_[1]], seq_path_[pos_[0]]
    return seq_path_

# 倒置法
def reverse_pos(seq_path_):
    pos_list = range(len(seq_path_))
    pos_ = random.sample(pos_list, 2)
    pos_ = np.sort(pos_, kind='quicksort')
    seq_path_[pos_[0]: pos_[1] + 1] = seq_path_[pos_[0]: pos_[1] + 1][::-1]
    return seq_path_

# 移位法
def shift_pos(seq_path_):
    pos_list = range(len(seq_path_))
    pos_ = random.sample(pos_list, 4)
    pos_ = np.sort(pos_, kind='quicksort')
    new_path_ = seq_path_[: pos_[0]]
    new_path_ = np.append(new_path_, seq_path_[pos_[2]: pos_[3] + 1])
    new_path_ = np.append(new_path_, seq_path_[pos_[1] + 1: pos_[2]])
    new_path_ = np.append(new_path_, seq_path_[pos_[0]: pos_[1] + 1])
    new_path_ = np.append(new_path_, seq_path_[pos_[3] + 1:])
    return new_path_

# 模拟退火算法
beta_k = lambda tk_, te_, t0_: (t0 - te) / (tk - te)

t0 = 100
te = 1
tk = t0
ze = np.inf
alpha = 0.99
number = 0
position = range(1, 31)
zero_position = range(1, 30)

best_path = random.sample(position, 30)
best_zero = random.sample(zero_position, 7)
best_zero = np.sort(best_zero, kind='quicksort')
best_ze = np.inf
best_path = np.insert(best_path, best_zero, [0 for _ in range(7)])

middle_path = []
middle_zero_pos = []
middle_zes = []

better_ze = []

p1 = 0.8
p2 = 0.1


def longway(path_):
    path_.insert(0, 0)
    path_.append(0)
    long = 0
    for i in range(len(path_) - 1):
        long += distance_mat[path_[i]][path_[i + 1]]
    return long

def divide_path(best_path_):
    divide_ways = []
    zero_pos_ = []
    zero_n_ = 0
    for n in best_path_:
        if n:
            zero_n_ += 1
        else:
            zero_pos_.append(zero_n_)

    seq_path_ = []
    for i in range(len(best_path_)):
        if best_path_[i]:
            seq_path_.append(best_path_[i])

    path = seq_path_[: zero_pos_[0] + 1]
    if len(path):
        divide_ways.append(copy.deepcopy(path))
    for w in range(len(zero_pos_) - 1):
        path = seq_path_[zero_pos_[w] + 1: zero_pos_[w + 1] + 1]
        if len(path):
            divide_ways.append(copy.deepcopy(path))
    path = seq_path_[zero_pos_[-1] + 1:]
    if len(path):
        divide_ways.append(copy.deepcopy(path))

    # 确定最终路径
    final_way = []
    weight = 0
    for w in divide_ways:
        cut_p = []
        for p in range(len(w)):
            weight += pos_weight[w[p] - 1]
            if weight > max_weight:
                cut_p.append(p)
                weight = pos_weight[w[p] - 1]
        weight = 0
        if not len(cut_p):
            final_way.append(w)
        else:
            final_way.append(w[: cut_p[0]])
            for i in range(len(cut_p) - 1):
                final_way.append(w[cut_p[i]: cut_p[i + 1]])
            final_way.append(w[cut_p[-1]:])
    return final_way


while tk > te:

    # 产生新解
    p = np.random.rand(1)[0]
    if p <= p1:
        new_path = exchange_pos(best_path)
    elif p > p1 + p2:
        new_path = reverse_pos(best_path)
    else:
        new_path = shift_pos(best_path)

    # 对新解进行拆分
    zero_pos = []
    zero_n = 0
    for n in new_path:
        if n:
            zero_n += 1
        else:
            zero_pos.append(zero_n)

    seq_path = []
    for i in range(len(new_path)):
        if new_path[i]:
            seq_path.append(new_path[i])

    # 计算是否符合限制条件，不符合则重新生成
    weight_info = sum_weight(seq_path, zero_pos)
    if not weight_info[0]:
        continue
    time_info = sum_time(seq_path, zero_pos)
    if not time_info[0]:
        continue

    # 产生可行解则加1
    number += 1

    final = divide_path(new_path)
    cur_ze = 0
    for p in final:
        cur_ze += longway(p)
    print(cur_ze)
    # excess_list = weight_info[2]
    # for item in excess_list:
    #     cur_ze += distance_mat[0][item[0]] + distance_mat[0][item[1]]
    #     cur_ze -= distance_mat[item[0]][item[1]]

    # 记录中间两次的情况
    if number == 1 or number == 200:
        middle_path.append(new_path)
        middle_zero_pos.append(zero_pos)
        middle_zes.append(cur_ze)

    # 判断是否是更优解，是否接受该解
    if cur_ze <= best_ze:
        better_ze.append(cur_ze)
        tk = np.power(alpha, number) * t0
        best_ze = cur_ze
        best_path = new_path
        best_zero = zero_pos
    else:
        if np.random.rand(1)[0] < np.exp(-beta_k(tk, te, t0)):
            better_ze.append(cur_ze)
            best_ze = cur_ze
            best_path = new_path
            best_zero = zero_pos
        else:
            better_ze.append(best_ze)


# 画出迭代情况图
ax_1 = plt.figure(figsize=(6.8, 6.2)).gca()
ax_1.plot(range(len(better_ze)), better_ze, marker='o', markersize=4, linestyle='--', label='总路程')
ax_1.set_xlabel('迭代次数', fontdict={'size': 13})
ax_1.set_ylabel('总路程（km）', fontdict={'size': 13})
plt.legend(fontsize='x-large')
plt.savefig('第一问迭代图.png')
plt.show()


# 切分路径
def divide_path(best_path_):
    divide_ways = []
    zero_pos_ = []
    zero_n_ = 0
    for n in best_path_:
        if n:
            zero_n_ += 1
        else:
            zero_pos_.append(zero_n_)

    seq_path_ = []
    for i in range(len(best_path_)):
        if best_path_[i]:
            seq_path_.append(best_path_[i])

    path = seq_path_[: zero_pos_[0] + 1]
    if len(path):
        divide_ways.append(copy.deepcopy(path))
    for w in range(len(zero_pos_) - 1):
        path = seq_path_[zero_pos_[w] + 1: zero_pos_[w + 1] + 1]
        if len(path):
            divide_ways.append(copy.deepcopy(path))
    path = seq_path_[zero_pos_[-1] + 1:]
    if len(path):
        divide_ways.append(copy.deepcopy(path))

    # 确定最终路径
    final_way = []
    weight = 0
    for w in divide_ways:
        cut_p = []
        for p in range(len(w)):
            weight += pos_weight[w[p] - 1]
            if weight > max_weight:
                cut_p.append(p)
                weight = pos_weight[w[p] - 1]
        weight = 0
        if not len(cut_p):
            final_way.append(w)
        else:
            final_way.append(w[: cut_p[0]])
            for i in range(len(cut_p) - 1):
                final_way.append(w[cut_p[i]: cut_p[i + 1]])
            final_way.append(w[cut_p[-1]:])
    return final_way


# 计算分离最短路径
print('最终路径：(每次初始值的不同，以及迭代过程新解产生的随机性\n最终结果会有略微差异，但最短总路径基本稳定)')
final_path = divide_path(best_path)
for p in final_path:
    print(p)
print('最短总路程为:', best_ze, 'KM')


# 绘制路径示意图
def draw_path(all_path):
    ax = plt.figure(figsize=(8, 7)).gca()
    ax.scatter(delivery_pos[:, 0], delivery_pos[:, 1], color='black', marker='o', s=100)
    colors = ['red', 'gold', 'forestgreen', 'darkorange',
              'dodgerblue', 'blue', 'darkviolet', 'violet',
              'pink', 'gray', 'brown', 'sienna', 'olive',
              'lawngreen', 'springgreen', 'cyan', 'hotpink']
    for p_ in range(len(all_path)):
        path_ = all_path[p_]
        path_ = np.insert(path_, 0, 0)
        path_ = np.insert(path_, len(path_), 0)
        for pos in range(len(path_) - 1):
            ax.arrow(delivery_pos[path_[pos]][0], delivery_pos[path_[pos]][1],
                     delivery_pos[path_[pos + 1]][0] - delivery_pos[path_[pos]][0],
                     0, width=0.1, color=colors[p_])
            ax.arrow(delivery_pos[path_[pos + 1]][0], delivery_pos[path_[pos]][1],
                     0, (delivery_pos[path_[pos + 1]][1] - delivery_pos[path_[pos]][1]) * 0.95,
                     width=0.1, color=colors[p_])
    ax.set_title('规划路径图（水平竖直路径）', fontdict={'size': 15})
    plt.xlim(0, 30)
    plt.ylim(0, 25)
    plt.show()

    ax = plt.figure(figsize=(8, 7)).gca()
    ax.scatter(delivery_pos[:, 0], delivery_pos[:, 1], color='black', marker='o', s=100)
    for p_ in range(len(all_path)):
        path_ = all_path[p_]
        path_ = np.insert(path_, 0, 0)
        path_ = np.insert(path_, len(path_), 0)
        for pos in range(len(path_) - 1):
            ax.arrow(delivery_pos[path_[pos]][0], delivery_pos[path_[pos]][1],
                     delivery_pos[path_[pos + 1]][0] - delivery_pos[path_[pos]][0],
                     delivery_pos[path_[pos + 1]][1] - delivery_pos[path_[pos]][1],
                     width=0.1, color=colors[p_])
    ax.set_title('规划路径图（斜线路径示意）', fontdict={'size': 15})
    plt.xlim(0, 30)
    plt.ylim(0, 25)
    plt.show()


draw_path(final_path)
