#%%
import numpy as np

delivery_pos = np.array([[0, 0], [3, 2], [1, 5], [5, 4], [4, 7], [0, 8], [3, 11],
                         [7, 9], [9, 6], [10, 2], [14, 0], [17, 3], [14, 6],
                         [12, 9], [10, 12], [7, 14], [2, 16], [6, 18], [11, 17],
                         [15, 12], [19, 9], [22, 5], [21, 0], [27, 9], [15, 19],
                         [15, 14], [20, 17], [21, 13], [24, 20], [25, 16], [28, 18]])
pos_weight = np.array([8, 8.2, 6, 5.5, 3, 4.5, 7.2, 2.3, 1.4, 6.5, 4.1, 12.7, 5.8, 3.8, 4.6,
                       3.5, 5.8, 7.5, 7.8, 3.4, 6.2, 6.8, 2.4, 7.6, 9.6, 10, 12, 6, 8.1, 4.2])
#%%
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

def longway(path_):
    path_.insert(0, 0)
    path_.append(0)
    long = 0
    for i in range(len(path_) - 1):
        long += distance_mat[path_[i]][path_[i + 1]]
    print(long)
    # long -= distance_mat[path_[-2]][path_[-1]]

#%%

path = np.array([[1, 13, 22],
[12, 6, 7],
[5],
[14, 17, 3, 2],
[16, 15, 24, 19],
[8, 9, 10, 25],
[18, 26],
[4],
[11, 27],
[29, 30, 28, 20, 23],
[21]])
for p in path:
    longway(p)