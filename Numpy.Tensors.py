import numpy as np

X = np.array(12) # Scalar / 0-Tensor
print(X.ndim)

Y = np.array([1,2,3,4]) # Vector / 1-Tensor
print(Y.ndim)

Z = np.array([[1,2,3,4], # Matrix / 2-Tensor
              [5,6,7,8]])

print(Z.ndim)
