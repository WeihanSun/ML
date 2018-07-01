# Bezier curve using for vector image to zoom in with limited calculation
# path.CURVE3: quad path.CURVE4: tertiary

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Path = mpath.Path

fig, ax = plt.subplots()

#pp1 = mpatches.PathPatch(
#    Path([(0, 0), (1, 0), (1, 1), (0, 0)],
#         [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
#    fc="none", transform=ax.transData)

vertices = [(0, 0), (1, 0), (1, 1), (2, 3), (0, 0)]
codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]

pp1 = mpatches.PathPatch(Path(vertices, codes), fc="none", transform=ax.transData)


ax.add_patch(pp1)
ax.plot([0.75], [0.25], "ro")
ax.set_title('The red point should be on the path')

plt.show()