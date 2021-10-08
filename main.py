import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = scipy.io.loadmat('ex4data1.mat')
x = data['X']
y = data['y']
n = x.shape[1] + 1
m = x.shape[0]
# normalized_x = x / 255
x0 = np.ones((m, 1))
x = np.hstack((x0, x))

weights = scipy.io.loadmat('ex4weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

learning_rate = 0.0005




#plottins pixels
# X = []
# for i in range(0, 5000):
#     X.append(np.reshape(x[i][:400], (20, 20)))
# # print(np.array(X).shape)
#
# pixel_plot = plt.figure()
# pixel_plot.add_axes()
# plt.title("pixel_plot")
# pixel_plot = plt.imshow(X[700], cmap='twilight', interpolation='nearest')
# plt.colorbar(pixel_plot)
# plt.savefig('pixel_plot.png')
# plt.gray()
# plt.show()