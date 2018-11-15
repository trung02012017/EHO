import numpy as np

a = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5]])

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i, j] > 2:
            a[i, j] = 2

print(a)