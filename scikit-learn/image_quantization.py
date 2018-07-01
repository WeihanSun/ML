# using K-means to cluster pixels described in RGB
# shuffle 1000 pixel for clustering
# use the centroid (RGB) of each cluster
# as the new color for pixels in the same cluster

from skimage import data, img_as_float64
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

ncluster = 32
original_img = np.array(data.astronaut(), dtype=np.float64) / 255
original_dim = tuple(original_img.shape)
width, height, depth = tuple(original_img.shape)
img_flattened = np.reshape(original_img, (width * height, depth))
img_array_sample = shuffle(img_flattened, random_state=0)[:1000]
estimater = KMeans(n_clusters=ncluster, random_state=0)
estimater.fit(img_array_sample)
cluster_assignment = estimater.predict(img_flattened)
compressed_palette = estimater.cluster_centers_
compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_idx = 0
for i in range(width):
    for j in range(height):
        compressed_img[i][j] = compressed_palette[cluster_assignment[label_idx]]
        label_idx += 1
plt.subplot(121)
plt.title('original image')
plt.imshow(original_img)
plt.axis('off')
plt.subplot(122)
plt.title('compressed image (%d colors)' % ncluster)
plt.imshow(compressed_img)
plt.axis('off')
plt.show()

