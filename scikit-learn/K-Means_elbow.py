# find best K for K-Means
# metric: (1) average distortion & (2) silhouette coefficient
# (1) average distortion: decrease with increase of K,
# but the improvement declines
# Use elbow method to find best K
# (2) silhouette coefficient: increase for more compact clusters
# mean of instances' silhouette coefficients(SC)
# SC of 1 instance is (-1, 1), = mean dist to instances in the cluster /
# mean dist to instances in the other clusters

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T

K = range(1, 10)
diff = []
sil_coef = []
mean_distortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # distortion
    dist = sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
    # silhouette coefficient
    if k == 1:
        sc = 0
    else:
        sc = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
    if k > 2:
        diff.append(mean_distortions[k - 2] - dist)
    mean_distortions.append(dist)
    sil_coef.append(sc)


elbow = np.argmax(diff)+1
max_index = np.argmax(sil_coef)
print('(distortion) max index = %d' % (elbow), np.round(diff, 3))
print('(silhouette coefficient) max index = %d' % max_index, np.round(sil_coef, 3))
# draw K selection
fig0 = plt.figure(0)
sub_plot1 = plt.subplot(121)
sub_plot2 = plt.subplot(122)
sub_plot1.plot(K, mean_distortions, 'bx-')
sub_plot1.annotate('elbow', xy=(K[elbow]+.01, mean_distortions[elbow]+.01),
            xytext=(K[elbow]+.3, mean_distortions[elbow]+.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
sub_plot1.set_xlabel('k')
sub_plot1.set_ylabel('Average distortion')
sub_plot1.set_title('(1) average distortion')

sub_plot2.plot(K, sil_coef, 'rx-')
sub_plot2.annotate('maximum', xy=(K[max_index]+.01, sil_coef[max_index]+.01),
            xytext=(K[max_index]+1.5, sil_coef[max_index]),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
sub_plot2.set_xlabel('k')
sub_plot2.set_ylabel('Silhouette coefficient')
sub_plot2.set_title('(2) Silhouette coefficient')
# draw best clustering
fig1 = plt.figure(1)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
markers = ['x', '^', 'o', '*', '+']
kmeans = KMeans(n_clusters=K[max_index]).fit(X)
for i, l in enumerate(kmeans.labels_):
    plt.plot(X[i][0], X[i][1], color=colors[l], marker=markers[l])
plt.show()
