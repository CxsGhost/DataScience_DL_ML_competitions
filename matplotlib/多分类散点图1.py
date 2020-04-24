from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500,
                           n_features=10,
                           n_classes=5,
                           n_informative=4,
                           random_state=0)
print(y)
