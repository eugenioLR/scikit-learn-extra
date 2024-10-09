import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neighbors._base import KNeighborsMixin, NeighborsBase
# from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets


class LaesaClassifer(BaseEstimator, KNeighborsMixin, ClassifierMixin):
    def __init__(
        self,
        *,
        n_prototypes=5,
        weights="uniform",
        p=2,
        metric="minkouski",
        metric_params=None
    ):
        self.n_prototypes = n_prototypes
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self._fit_X = X.copy() 
        self._fit_y = y.copy()
        self.n_features_in = X.shape[1]
        self.n_samples_fit_ = X.shape[0]

        # Select prototypes
        prototype_idx = np.random.permutation(self.n_samples_fit_)[:self.n_prototypes]
        self._prototypes = self._fit_X[prototype_idx]
        self._fit_dist_to_prototypes = sp.spatial.distance_matrix(self._fit_X, self._prototypes)

                            
    def predict(self, X):
        dist_to_prototypes = sp.spatial.distance_matrix(X, self._prototypes)
        
        triangle_inequality = np.abs(dist_to_prototypes.reshape((X.shape[0], 1, self.n_prototypes)) - self._fit_dist_to_prototypes)
        max_dist_bound = np.max(triangle_inequality, axis=1)
                
        order_dist_to_prototypes = np.argsort(dist_to_prototypes, axis=1).squeeze()
        ordered_dist_to_prototypes = np.sort(dist_to_prototypes, axis=1).squeeze()
                
        # Ignore distances that exceed the maximum distance bound
        stop_condition = ordered_dist_to_prototypes > max_dist_bound
        # Convert boolean vector to indices where the first True is found
        stop_condition_idx = np.where(stop_condition.any(axis=1), stop_condition.argmax(axis=1), self.n_prototypes)
                
        # Fill distance matrix after stop condition indices with infintes
        columns = np.tile(np.arange(dist_to_prototypes.shape[1]), dist_to_prototypes.shape[0]).reshape(dist_to_prototypes.shape)
        mask = columns > stop_condition_idx[:, None]
        dist_to_prototypes_masked = dist_to_prototypes.copy()
        dist_to_prototypes_masked[mask] = np.inf

        # Choose the closest prototype to each data point and the closest point to it in the dataset
        prototype_chosen = np.argmin(dist_to_prototypes_masked, axis=1).flatten()                 
        nearest_neighbors = np.argmin(triangle_inequality[np.arange(triangle_inequality.shape[0]), :, prototype_chosen], axis=1)
                
        return self._fit_y[nearest_neighbors]
