import numpy as np

a = np.zeros([12, 200])
a[1][1] = 1
print(np.all(a == 0))
