import numpy as np
import math
import random as ran
from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt



def sparse_x(N, s):
    # np.random.seed(0)
    range1 = list(np.random.uniform(-10, -1, (s,1)))
    range2 = list(np.random.uniform(1, 10, (s,1)))
    combined_range = range1 + range2
    # print(combined_range)

    # ran.seed(0)
    non_zeros = np.array(ran.sample(combined_range, s))

    zeros = list(np.zeros(N-s))

    x = np.array([np.append(non_zeros, zeros)], dtype=float).T

    np.random.shuffle(x)
    return x

# x = sparse_x(20,5)
# print(x)
# print(y)



def OMP(A_norm,y,s):
    (M, N) = A_norm.shape
    res = y
    index_vector = np.zeros(N, dtype=int)
    x_est = np.zeros((N, 1))
    for i in range(s):
        weight = np.fabs(np.dot(A_norm.T, res))
        wmax_pos = np.argmax(weight)
        index_vector[wmax_pos] = 1
        A_newinv = np.linalg.pinv(A_norm[:, index_vector>0])
        weight_reduce = np.dot(A_newinv, y)
        # print(weight_reduce)
        res = y - np.dot(A_norm[:, index_vector > 0], weight_reduce)
        error = np.linalg.norm(res)

        if error < 10**(-5):
            break

    x_est[index_vector>0] = weight_reduce
    return x_est


N = 20

s_max = []
Pro = []
ERROR = []
for M in range(2, N):
    success_vector = []
    avg_err_vector = []
    for num_s in range(1, M):
        # print(type(num_s))

        err_all = []
        X = []
        success = 0
        for i in range(800):

            A = np.random.normal(0, 1, size=(M, N))
            A_norm = normalize(A, axis=0, norm='max')
            x = sparse_x(N, num_s)
            X.append(x)
            y = np.dot(A_norm, X[0])

            x_est = OMP(A_norm, y, num_s)
            err = np.linalg.norm(X[0] - x_est) / np.linalg.norm(X[0])
            if err < 10**(-5):
                success += 1


            err_all.append(err)
        avg_err = sum(err_all) / len(err_all)
        avg_err_vector.append(avg_err)

        success_rate = success / 800
        if success_rate <= 0.05:
            # smax = num_s - 1
            # s_max.append(smax)

            break
        success_vector.append(success_rate)
        # print(success_vector)
        # success_vector = [0] * (18 - len(success_vector)) + success_vector
    # success_vector = success_vector[::-1]
    # success_vector = [0] * (18 - len(success_vector)) + success_vector

    avg_err_vector = avg_err_vector[::-1]
    avg_err_vector = [0] * (18 - len(avg_err_vector)) + avg_err_vector
    ERROR.append(avg_err_vector[::-1])





ERROR_matrix = np.array([np.array(i) for i in ERROR])
ERROR_matrix = np.vstack(ERROR_matrix)

# Probability = np.array([np.array(i) for i in Pro])
# Probability = np.vstack(Probability)
# M = np.linspace(1, 18, 18, dtype=int)
# S = np.linspace(1, 18, 18, dtype=int)
# fig1, ax1 = plt.subplots()
# im1 = ax1.imshow(Probability)
#
# ax1.set_xticks(np.arange(18), minor=False)
# ax1.set_yticks(np.arange(18), minor=False)
# ax1.set_xticklabels(M, minor=False)
# ax1.set_yticklabels(S, minor=False)
#
# plt.setp(ax1.get_xticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor", size=5)
# plt.setp(ax1.get_yticklabels(), rotation=90, ha="right",
#          rotation_mode="anchor", size=5)

M = np.linspace(1, 18, 18, dtype=int)
S = np.linspace(1, 18, 18, dtype=int)
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(ERROR_matrix)

ax2.set_xticks(np.arange(18), minor=False)
ax2.set_yticks(np.arange(18), minor=False)
ax2.set_xticklabels(M, minor=False)
ax2.set_yticklabels(S, minor=False)

plt.setp(ax2.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor", size=6)
plt.setp(ax2.get_yticklabels(), rotation=90, ha="right",
         rotation_mode="anchor", size=6)

plt.colorbar(im2)
# plt.colorbar(im1)

# ax1.set_title("Recovery noiseless phase transition N=20")
ax2.set_title("Normalized Error transition plot N=20")
# ax1.set_xlabel('Sparsity')
# ax1.set_ylabel('M')
ax2.set_xlabel('Sparsity')
ax2.set_ylabel('M')

plt.show()