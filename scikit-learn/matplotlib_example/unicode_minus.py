# unknown

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(19680801)
matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
# number of points = 100
ax.plot(10*np.random.rand(100), 10*np.random.rand(100), 'o')
ax.set_title('Using hyphen instead of Unicode minus')
plt.show()