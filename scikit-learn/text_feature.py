# test features
# (1) stopword (removing a, the, some, in, beneath...)
# (2) stem & lemma (using Natural Language Tool Kit (NLTK))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus = ['UNC player Duke in basketball',
          'Duke lost the basketball game',
          'I ate a sandwitch']

vectorizer = CountVectorizer(stop_words='english')
features = vectorizer.fit_transform(corpus).todense()
print(features)
print(vectorizer.vocabulary_)

print('Dist between 1st and 2nd: %.3f' % euclidean_distances(features[0], features[1]))
print('Dist between 1st and 3rd: %.3f' % euclidean_distances(features[0], features[2]))
print('Dist between 2nd and 3rd: %.3f' % euclidean_distances(features[1], features[2]))

print("======stem & lemma=======")
import nltk
#nltk.download() # only for the 1st time
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering', 'v'))
print(lemmatizer.lemmatize('gathering', 'n'))

stemmer = PorterStemmer()
print(stemmer.stem('ate'))
print(lemmatizer.lemmatize('ate', 'v'))

from nltk import word_tokenize
print("------------word token-------------")
for document in corpus:
    for token in word_tokenize(document):
        print(token)
