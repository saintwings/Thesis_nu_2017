import numpy as np
a = np.ones((100,100,100))
b = np.ones((100,100,100))
c = np.concatenate((a[...,np.newaxis],b[...,np.newaxis]),axis=3)

print(c)

print(c.shape)

c[1][2][3][1] = 5
print(c[1][2][3][1])
