import numpy as np
from scipy.stats import mode

def columnize(x):
    return np.array(x).reshape((-1, 1))

class Stump:

    def __init__(self,
                 dimselect_method='random',
                 splitselect_method='all'):

        #ensure a valid dimselect_method is selected
        self._dimselect_methods = {'random': self._dimselect_random}
        if dimselect_method not in self._dimselect_methods.keys():
            raise NotImplementedError('dimselect_method must be in: '+str(self._dimselect_methods.keys()))
        self._dimselect_method = self._dimselect_methods[dimselect_method]


        #ensure a valid splitselect_method is selected
        self._splitselect_methods= {'all': self._splitselect_all}
        if splitselect_method not in  self._splitselect_methods.keys():
            raise NotImplementedError('splitselect_method must be in: '+str(self._splitselect_methods.keys()))
        self._splitselect_method = self._splitselect_methods[splitselect_method]

        self._k = None
        self._split = None
        self._less_eq_class = None
        self._greater_class = None
        return


    def _dimselect_random(self, X):
        return np.random.randint(0, X.shape[1])


    def _splitselect_all(self, features, Y):
        #get all possible split values
        feature_min = np.min(features)
        feature_max = np.max(features)
        splits = np.unique(features)

        #get parent entropy
        parent_entropy = self._get_entropy(Y)

        #minus 2 since we dont consider min and max
        info_gains = [0. for _ in splits]

        #calculate info gain for all splits
        for i, split in enumerate(splits):

            #no information gain in min or max split
            if np.allclose(split, feature_min) or np.allclose(split, feature_max):
                info_gains[i] = 0.

            #get class splits
            less_eq_Y = columnize([y\
                                  for i, y in enumerate(Y)\
                                  if features[i] <= split])

            greater_Y = columnize([y\
                                   for i, y in enumerate(Y)\
                                   if features[i] > split])

            #calculate information gain:
            gain = self._get_info_gain(parent_entropy,
                                       less_eq_Y,
                                       greater_Y)
            info_gains[i] = gain

        #return split with best gain
        argmax_gain = np.argmax(info_gains)
        split = splits[argmax_gain]

        #get class probabilities at this split
        #TODO: would it be faster to save splits?
        #NOTE: scipy mode returnes "mode result", so have to index and flatten
        less_eq_class = mode([y\
                              for i, y in enumerate(Y)\
                              if features[i] <= split])[0].flatten()

        greater_class = mode([y\
                              for i, y in enumerate(Y)\
                              if features[i] > split])[0].flatten()


        return split, less_eq_class, greater_class


    def _get_info_gain(self,
                       parent_entropy,
                       le_child,
                       g_child):

        #get split probabilities
        n_le_child = le_child.shape[0]
        n_g_child = g_child.shape[0]
        n = n_le_child + n_g_child

        p_le = n_le_child/n
        p_g = n_g_child/n

        entropy_le = self._get_entropy(le_child)
        entropy_g = self._get_entropy(g_child)

        conditional_entropy = -1 * p_le * entropy_le + p_g * entropy_g

        return parent_entropy + conditional_entropy


    def _get_entropy(self, X):

        entropy = 0

        for k in np.unique(X):
            n = X.shape[0]
            p_k = np.sum(X == k)/n
            class_entropy = p_k * np.log(p_k)
            entropy -= class_entropy

        return entropy


    def fit(self, X, Y):

        #select a dimension
        k = self._dimselect_method(X)

        #get features
        features = X[:,k]

        #get split
        split, less_eq_class, greater_class = self._splitselect_method(features, Y)

        #save decision params
        self._k = k
        self._split = split
        self._less_eq_class = less_eq_class
        self._greater_class = greater_class
        return


    def predict(self, X):

        split = self._split
        less_eq_class = self._less_eq_class
        greater_class = self._greater_class
        features = X[:, self._k]
        predictions = [0 for _ in range(features.shape[0])]

        #classify each datapoint
        for i in range(features.shape[0]):
            feature = features[i]
            if feature <= split:
                predictions[i] = less_eq_class
            else:
                predictions[i] = greater_class

        return columnize(predictions)


    def split(self, X):
        split = self._split
        features = X[:, self._k]

        #split examples
        less_eq = [X[i]\
                   for i in X.shape[0]\
                   if features[i] <= split]

        greater = [X[i]\
                   for i in X.shape[0]\
                   if features[i] > split]

        return np.stack(less_eq), np.stack(greater)
