from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus = ['The dog ate a sandwich and I ate a sandwich',
          'The wizard transfigured a sandwich']

vectorizer1 = TfidfVectorizer(stop_words='english')
print(vectorizer1.fit_transform(corpus))
print(vectorizer1.fit_transform(corpus).todense())
feature_vector1 = vectorizer1.fit_transform(corpus).todense()
print('distance between two documents: %.2f' %
      euclidean_distances(feature_vector1[0], feature_vector1[1]))

print("========hashing trick=========")
from sklearn.feature_extraction.text import HashingVectorizer
#corpus = ['the', 'ate', 'bacon', 'cat']

vectorizer2 = HashingVectorizer(n_features=6)
print(vectorizer2.fit_transform(corpus))
print(vectorizer2.fit_transform(corpus).todense())
feature_vector2 = vectorizer2.fit_transform(corpus).todense()
print('distance between two documents: %.2f' %
      euclidean_distances(feature_vector2[0], feature_vector2[1]))