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


    def _reweight_data(self, data_weights, Y, Y_hat, error):
        for idx in range(data_weights.shape[0]):

            #drop weight on correct prediction
            if Y[idx] == Y_hat[idx]:
                data_weights[idx] = data_weights[idx]/(2*(1-error))

            #increase weight on incorrect prediction
            else:
                data_weights[idx] = data_weights[idx]/(2*error)

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

            #get new data weights
            Y_hat = stump.predict(X)
            error = np.sum(data_weights[np.where(Y_hat != Y)])
            data_weights = self._reweight_data(data_weights, Y ,Y_hat, error)

            #update stump weights
            stump_weight = .5 * np.log((1 - error)/error)
            self._stump_weights.append(stump_weight)

            #update stumps
            self._stumps.append(stump)


    def predict(self, X):
        n = X.shape[0]

        #prediction matrix has shape (num examples, num stumps)
        prediction_matrix = np.stack([stump.predict(X).flatten()\
                                      for stump in self._stumps])

        #take max weight prediction for every example
        final_predictions = [0 for _ in range(n)]
        for example_idx in range(n):
            cur_predictions = prediction_matrix[example_idx].flatten()
            accumulator = {y_hat:0 for y_hat in np.unique(cur_predictions)}
            for i, y_hat in enumerate(cur_predictions):
                accumulator[y_hat] += self._stump_weights[i]

            #get the max weighted class
            #apparently the fastest way
            #https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
            weights = list(accumulator.values())
            classes = list(accumulator.keys())
            prediction = classes[weights.index(max(weights))]
            final_predictions[example_idx] = prediction

        return columnize(final_predictions)
