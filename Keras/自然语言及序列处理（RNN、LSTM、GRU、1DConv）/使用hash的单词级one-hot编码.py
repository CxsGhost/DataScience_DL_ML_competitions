import numpy as np

samples = ['the cat sat on the mat', 'the dog ate my homework']
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in enumerate(sample.split(' ')[: 50]):
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1

print(results)
print(results.shape)
print(hash('dog'))
