# multi-label for 1 instance
# problem transformation: (1) increase classes (2) one binary classifier for each class
# performance metrics

import numpy as np
from sklearn.metrics import hamming_loss, jaccard_similarity_score
predicted = np.array([[0, 1], [1, 1]])
true = np.array([[0, 1], [1, 1]])
print('haming_loss = %f' % hamming_loss(predicted, true))
print('jaccard_sim = %f' % jaccard_similarity_score(predicted, true))
predicted = np.array([[0, 1], [1, 0]])
print('haming_loss = %f' % hamming_loss(predicted, true))
print('jaccard_sim = %f' % jaccard_similarity_score(predicted, true))
predicted = np.array([[1, 1], [1, 0]])
print('haming_loss = %f' % hamming_loss(predicted, true))
print('jaccard_sim = %f' % jaccard_similarity_score(predicted, true))