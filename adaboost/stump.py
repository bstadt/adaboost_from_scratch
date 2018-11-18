import numpy as np
np.seterr(all='raise')
from scipy.stats import mode

#I <3 column vectors
def columnize(x):
    return np.array(x).reshape((-1, 1))

class Stump:

    def __init__(self,
                 dimselect_method='all',
                 splitselect_method='all'):

        #ensure a valid dimselect_method is selected
        self._dimselect_methods = {'all': self._dimselect_all}
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


    def _dimselect_all(self, X):
        return list(range(X.shape[1]))


    def _splitselect_all(self, features, Y, weights):

        #get parent entropy
        parent_entropy = self._get_weighted_entropy(Y, weights)

        #sort features, weights, labels
        sorted_args = np.argsort(features.flatten())
        sorted_features = columnize(features.flatten()[sorted_args])
        sorted_weights = columnize(weights.flatten()[sorted_args])
        sorted_Y = columnize(Y.flatten()[sorted_args])

        #precalculate splits
        n = sorted_Y.shape[0]
        split_vals = np.unique([np.mean(sorted_features[i-1:i+1])\
                                for i in range(1, n)])
        split_vals = sorted(split_vals)

        #dont want to split on the first or lats values
        if split_vals[-1] == sorted_features[-1]:
            split_vals = split_vals[:-1]

        if split_vals[0] == sorted_features[0]:
            split_vals = split_vals[1:]

        #calculate weighted info gain for all splits
        best_split_idx = None
        best_info_gain = None

        feature_split_idx = 0
        for split_idx, split_val in enumerate(split_vals):

            #update feature split idx
            while sorted_features[feature_split_idx] <= split_val:
                feature_split_idx +=1

            #get class split
            less_eq_Y = sorted_Y[:feature_split_idx]
            greater_Y = sorted_Y[feature_split_idx:]

            #get weight split
            less_eq_w = sorted_weights[:feature_split_idx]
            greater_w = sorted_weights[feature_split_idx:]

            #calculate information gain:
            info_gain = self._get_weighted_info_gain(parent_entropy,
                                                     less_eq_Y,
                                                     less_eq_w,
                                                     greater_Y,
                                                     greater_w)

            #update best stump params
            if best_info_gain is None or info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_idx = split_idx

        #return split with best gain
        #NOTE: scipy mode returns a "mode result" so you need to index in and flatten
        less_eq_class = mode(Y[:feature_split_idx])[0].flatten()
        greater_class = mode(Y[feature_split_idx:])[0].flatten()

        split = split_vals[best_split_idx]

        return split, less_eq_class, greater_class, best_info_gain


    def _get_weighted_info_gain(self,
                                parent_entropy,
                                le_child,
                                le_weight,
                                g_child,
                                g_weight):

        #get split weights
        w_le = np.sum(le_weight)
        w_g = np.sum(g_weight)

        entropy_le = self._get_weighted_entropy(le_child, le_weight)
        entropy_g = self._get_weighted_entropy(g_child, g_weight)

        conditional_entropy = w_le * entropy_le + w_g * entropy_g

        return parent_entropy - conditional_entropy


    def _get_weighted_entropy(self, X, w):

        entropy = 0
        w_tot = np.sum(w)

        for k in np.unique(X):
            locs = np.where(X == k)
            w_k = np.sum(w[locs])/w_tot
            class_entropy = w_k * np.log(w_k)
            entropy -= class_entropy

        return entropy


    def fit(self, X, Y, weights):

        #select a dimension
        dims = self._dimselect_method(X)

        #within all selected dimensions, find best split
        best_k = None
        best_gain = None
        best_split = None
        bset_less_eq_class = None
        best_greater_class = None

        for k in dims:
            #get features
            features = X[:,k]

            #get split
            split, less_eq_class, greater_class, info_gain = self._splitselect_method(features, Y, weights)

            #save best results
            if best_gain is None or info_gain > best_gain:
                best_gain = info_gain
                best_k = k
                best_split = split
                best_less_eq_class = less_eq_class
                best_greater_class = greater_class

        #save decision params
        self._k = best_k
        self._split = best_split
        self._less_eq_class = best_less_eq_class
        self._greater_class = best_greater_class
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
