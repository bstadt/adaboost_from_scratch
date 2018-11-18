import numpy as np

def columnize(x):
    return np.array(x).reshape((-1, 1))

class AdaBoost:
    def __init__(self):
        self._w = None
        return

    def fit(X, Y, iterations=5000):

        #uniformly initialize weights
        w = [1/X.shape[0]\
            for _ in range(X.shape[0])]
        self._w = columnize(w)

        #run training loop
        for iteration in range(iterations):
            stump = Stump()
            stump.fit(X, Y)
        return

    def predict(X, Y):
        return
