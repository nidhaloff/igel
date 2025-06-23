"""K-Medians Clustering is an alternative to the popular K-Means Clustering, whereing the goal is to determine the median of the points in a cluster instead of the mean, to represent a cluster centre.

This algorithm differs from the K-Medoids algorithm in that the K-Medoids algorithm requires the cluster centres to be actual data samples, whereas the K-Medians and K-Means algorithms generate synthetic samples not necessarily present in the actual data.

K-Medians seeks to minimize the 1-norm distance from each point to its nearest cluster center, as opposed to K-Means which uses the Euclidean or 2-norm distance."""

import random
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin


class KMedians(BaseEstimator, ClusterMixin):
    """ Initialization of parameters
    Parameters
        ----------
        n_clusters : int, optional, default: 4
            The number of clusters to form as well as the number of medians to
            generate.

        metric : string, or callable, optional, default: 'euclidean'
            What distance metric to use. See :func:metrics.pairwise_distances
            metric can be 'precomputed', the user must then feed the fit method
            with a precomputed kernel matrix and not the design matrix X.

        method : string, optional, default: 'per-axis'
            Specify the method of computing the median for multidimensional data. 
            'per-axis' takes the median for each attribute and combines them to make the cluster centre. 

            More options can be implemented from the following:

            A. Juan and E. Vidal. Fast k-means-like clustering in metric spaces. Pattern Recognition Letters, 15(1):19–25, 1994.
            A. Juan and E. Vidal. Fast Median Search in Metric Spaces. In A. Amin, D. Dori, P. Pudil, and H. Freeman, editors, Advances in Pattern Recognition, volume 1451, pages 905–912. Springer-Verlag, 1998
            Whelan, C., Harrell, G., & Wang, J. (2015). Understanding the K-Medians Problem.

        init : {'random'}, optional, default: 'random'
            Specify median initialization method. 'random' selects n_clusters
            elements from the dataset. 

            More options can be implemented from the following:

            Alfons Juan and Enrique Vidal. 2000. Comparison of Four Initialization Techniques for the K -Medians Clustering Algorithm. In Proceedings of the Joint IAPR International Workshops on Advances in Pattern Recognition. Springer-Verlag, Berlin, Heidelberg, 842–852.
 
        max_iter : int, optional, default : 300
            Specify the maximum number of iterations when fitting. It can be zero in
            which case only the initialization is computed which may be suitable for
            large datasets when the initialization is sufficiently efficient

        tol : float, optional, default : 0.0001
            Specify the tolerance of change in cluster centers. If change in cluster centers is less than tolerance, algorithm stops.

        random_state : int, RandomState instance or None, optional
            Specify random state for the random number generator. Used to
            initialise medians when init='random'.

        Attributes
        ----------
        cluster_centers_ : array, shape = (n_clusters, n_features)
                or None if metric == 'precomputed'
            Cluster centers, i.e. medians (elements from the original dataset)

        medoid_indices_ : array, shape = (n_clusters,)
            The indices of the median rows in X

        labels_ : array, shape = (n_samples,)
            Labels of each point

        inertia_ : float
            Sum of distances of samples to their closest cluster center.
            
        score_ : float
            Negative of the inertia. The more negative the score, the higher the variation in cluster points, the worse the clustering. 
    """

    def __init__(self, n_clusters = 4, metric = 'manhattan', method = 'per-axis', init = 'random', max_iter = 300, tol = 0.0001, random_state = None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _get_random_state(self, seed = None):
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        elif isinstance(seed, (int, np.integer)):
            return np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            return seed 

    def _is_nonnegative(self, value, variable, strict = True):
        """Checks if the value passed is a non-negative integer which may or may not include zero"""

        if not isinstance(value,(int,np.integer)):
            raise ValueError("%s should be an integer" % (variable))

        if strict:
            isnegative = value > 0
        else:
            isnegative = value >= 0

        if not isnegative:
            raise ValueError("%s should be non-negative" % (variable))
        return isnegative

    def _check_arguments(self):
        """Checks if arguments are as per specification"""

        if self._is_nonnegative(self.n_clusters,"n_clusters") and self._is_nonnegative(self.max_iter, "max_iter") and isinstance(self.tol, (float, np.floating)):
            pass
        else:
            raise ValueError("Tolerance is not specified as floating-point value")

        if (isinstance(self.random_state, (int, np.integer, None, np.random.RandomState))  or self.random_state is np.random):
            pass
        else:
            raise ValueError("Random state is not valid")

        distance_metrics = ['euclidean', 'manhattan', 'cosine', 'cityblock', 'l1', 'l2']
        median_computation_methods = ['per-axis']
        init_methods = ['random']

        if self.metric not in distance_metrics:
            raise ValueError('%s not a supported distance metric' % (self.metric))

        if self.method not in median_computation_methods:
            raise ValueError('%s not a supported median computation method' % (self.metric))        
        
        if self.init not in init_methods:
            raise ValueError('%s not a supported initialization method' % (self.init))

    def _initialize_centers(self, X, n_clusters, random_state_object):
        """ Implementation of random initialization"""

        if self.init == 'random':
            """Randomly chooses K points within set of samples"""
            return random_state_object.choice(len(X), n_clusters)

    def _compute_inertia(self, distances, labels):
        """Compute inertia of new samples. Inertia is defined as the sum of the
        sample distances to closest cluster centers.

        Parameters
        ----------
        distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
            Distances to cluster centers.

        labels : {array-like}, shape = {n_samples}

        Returns
        -------
        Sum of sample distances to closest cluster centers.
        """

        # Define inertia as the sum of the sample-distances
        # to closest cluster centers
        inertia = 0
        for i in range(self.n_clusters):
          indices = np.argwhere(labels == i)
          inertia += np.sum(distances[i, indices])
        return inertia    

    def fit(self, X, Y = None):
        """Fits the model to the data"""

        self._check_arguments()
        random_state_object = self._get_random_state(self.random_state)

        if Y:
            raise Exception ("Clustering fit takes only one parameter")

        X = check_array(X, accept_sparse = ['csc','csr'])

        n_samples, n_features = X.shape[0], X.shape[1]

        if self.n_clusters > n_samples:
            raise ValueError('Number of clusters %s cannot be greater than number of samples %s' % (self.n_clusters, n_samples))

        centers = X[self._initialize_centers(X, self.n_clusters, random_state_object)]
        # print(centers)
        distances = pairwise_distances(centers, X, metric = self.metric)
        # print("Distances:", distances)

        medians = [None]* self.n_clusters
        labels = None

        if self.method == 'per-axis':
            for i in range(self.max_iter):
                old_centers = np.copy(centers)
                labels = np.argmin(distances, axis = 0)

                for item in range(self.n_clusters):
                    indices = np.argwhere(labels == item)
                    medians[item] = np.median(X[indices], axis = 0)

                centers = np.squeeze(np.asarray(medians))
                distances = pairwise_distances(centers, X, metric = self.metric)

                if np.all(np.abs(old_centers - centers) < self.tol):
                    break
                elif i == self.max_iter - 1:
                        warnings.warn(
                            "Maximum number of iteration reached before "
                            "convergence. Consider increasing max_iter to "
                            "improve the fit.",
                            ConvergenceWarning,
                        )
        self.cluster_centers_ = centers
        self.labels_ = np.argmin(distances, axis = 0)
        self.inertia_ = self._compute_inertia(distances, self.labels_)
        self.score_ = - self.inertia_

    def transform(self, X):
        """Transforms given data into cluster space of size {n_samples, n_clusters}"""
        X = check_array(X, accept_sparse=["csr", "csc"])
        check_is_fitted(self, "cluster_centers_")

        Y = self.cluster_centers_
        return pairwise_distances(X, Y=Y, metric=self.metric)

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
        return pairwise_distances_argmin(X, Y = self.cluster_centers_, metric=self.metric)
        
    def score(self, X):
        """Returns score"""
        return self.score_


        


            


    

