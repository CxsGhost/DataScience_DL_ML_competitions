from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# initialize and fit bag of words model
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, '
    'and one and one is two'])
bag = count.fit_transform(docs)

print(count.vocabulary_)
print(bag.toarray())









