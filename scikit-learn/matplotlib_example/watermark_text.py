# using alpha to control the transparency
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19820210)
fig, ax = plt.subplots()
ax.plot(np.random.rand(5), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
ax.grid()

fig.text(0.05, .005, 'Property of MPL', fontsize=50, color='gray',
         ha='left', va='bottom', alpha=0.5)
plt.show()