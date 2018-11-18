import numpy as np
from tqdm import tqdm
from .stump import Stump

#I <3 column vectors
def columnize(x):
    return np.array(x).reshape((-1, 1))

class AdaBoost:
    def __init__(self):
        self._stumps = []
        self._stump_weights = []
        return


    def _reweight_data(self, data_weights, Y, Y_hat, stump_weight):
        for idx in range(data_weights.shape[0]):

            #drop weight on correct prediction
            if Y[idx] != Y_hat[idx]:
                data_weights[idx] = data_weights[idx]*np.e**(stump_weight)

        data_weights /= np.sum(data_weights)

        return data_weights


    def fit(self, X, Y, stumps = 100):

        #uniformly initialize weights
        data_weights = [1/X.shape[0]\
                       for _ in range(X.shape[0])]
        data_weights = columnize(data_weights)

        for _ in tqdm(range(stumps)):

            #run training loop
            stump = Stump()
            stump.fit(X, Y, data_weights)
            Y_hat = stump.predict(X)

            #update stump weights
            error = np.sum(data_weights[np.where(Y_hat != Y)])
            stump_weight = np.log((1 - error)/error)
            self._stump_weights.append(stump_weight)

            #get new data weights
            data_weights = self._reweight_data(data_weights, Y ,Y_hat, stump_weight)

            #update stumps
            self._stumps.append(stump)


    def predict(self, X):
        n = X.shape[0]

        #prediction matrix has shape (num examples, num stumps)
        prediction_matrix = np.stack([stump.predict(X).flatten()\
                                      for stump in self._stumps]).T


        #preallocate final predictions
        final_predictions = np.ones((n, ))

        #take max weight prediction for every example
        for example_idx in range(n):
            cur_predictions = prediction_matrix[example_idx, :]
            class_weights = {prediction:0 for prediction in np.unique(cur_predictions)}
            for i, prediction in enumerate(cur_predictions):
                class_weights[prediction] += self._stump_weights[i]

            max_weight_class = max(class_weights, key=class_weights.get)
            final_predictions[example_idx] = max_weight_class

        return columnize(final_predictions)
