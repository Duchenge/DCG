import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#生成随机三维数据集
np.random.seed(4)
m =50
w1, w2 = 0.1, 0.3
noise = 0.2
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
#绘制三维图像
fig = plt.figure(figsize=(6, 3.8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:,0],X[:,1],X[:,2],"k.")
plt.show()

#svd分解
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

#构造对角矩阵
m, n = X.shape
S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)

#得到降维后的数据
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
#绘制降维后的二维图像
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.plot(X2D[:, 0], X2D[:, 1], "k.")
plt.show()


