import numpy as np


x = np.array([.1, .31, 0, .004, .3, 0, .41, .2, .01, 0])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])


def MSE(A, B): return np.square(np.subtract(B, A)).mean()
# 0.05843160000000001
print(MSE(x,y))

def derMSE(A, B): return (-2*np.subtract(B, A)).mean()
# -0.0668
print(derMSE(x,y))