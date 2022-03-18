import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('Y_Y2_Y3_and_A1_A2_A3.mat')
A1 = mat['A1']
A2 = mat['A2']
A3 = mat['A3']
y1 = mat['y1']
y2 = mat['y2']
y3 = mat['y3']
# print(A1.shape)

# def OMP(A_norm,y,s):
#     (M, N) = A_norm.shape
#     res = y
#     index_vector = np.zeros(N, dtype=int)
#     x_est = np.zeros((N, 1))
#     for i in range(s):
#         weight = np.fabs(np.dot(A_norm.T, res))
#         wmax_pos = np.argmax(weight)
#         index_vector[wmax_pos] = 1
#         A_newinv = np.linalg.pinv(A_norm[:, index_vector>0])
#         weight_reduce = np.dot(A_newinv, y)
#         res = y - np.dot(A_norm[:, index_vector > 0], weight_reduce)
#         error = np.linalg.norm(res)
#
#         if error < 10**(-5):
#             break
#
#     x_est[index_vector>0] = weight_reduce
#     return x_est







As = [A1, A2, A3]
Ys = [y1, y2, y3]
image = []
for i in range(len(As)):
    img = np.dot(np.linalg.pinv(As[i]), Ys[i])
    img = img.T
    img = img.reshape(160, 90)
    img = img.T
    image.append(img)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(image[0])
axs[0].set_title('MSE image for A1')
axs[1].imshow(image[1])
axs[1].set_title('MSE image for A2')
axs[2].imshow(image[2])
axs[2].set_title('MSE image for A3')

plt.show()