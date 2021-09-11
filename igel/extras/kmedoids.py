"""K-medoids clustering

K-Medoids, unlike K-Means, use existing points in the dataset as cluster centres. These data points have the smallest overall intra-cluster dissimilarity/distance, and are meant to represent a cluster more robustly without resorting to outlier penalties like K-Means uses. They can use any kind of distance metric to compute pairwise distance.

Note: This algorithm is not to be confused with K-Medians, which is a version of K-Means that uses Manhattan distance and calculates the median of the cluster instead of the mean. Additionally, in K-Medians, the resulting median data point may not be part of the original sample set, whereas in K-Medoids, it is necessary for the result to be an existing sample."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class KMedoids(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters=4,
        metric="euclidean",
        init="random",
        max_iter=300,
        random_state=None,
    ):
        """Parameters
        ----------
        n_clusters : int, optional, default: 4
            The number of clusters to form as well as the number of medoids to
            generate.

        metric : string, or callable, optional, default: 'euclidean'
            What distance metric to use. See :func:metrics.pairwise_distances
            metric can be 'precomputed', the user must then feed the fit method
            with a precomputed kernel matrix and not the design matrix X.

        init : {'random', 'heuristic'}, optional, default: 'random'
            Specify medoid initialization method. 'random' selects n_clusters
            elements from the dataset. 'heuristic' picks the n_clusters points
            with the smallest sum distance to every other point.

        max_iter : int, optional, default : 300
            Specify the maximum number of iterations when fitting. It can be zero in
            which case only the initialization is computed which may be suitable for
            large datasets when the initialization is sufficiently efficient

        random_state : int, RandomState instance or None, optional
            Specify random state for the random number generator. Used to
            initialise medoids when init='random'.

        Attributes
        ----------
        cluster_centers_ : array, shape = (n_clusters, n_features)
                or None if metric == 'precomputed'
            Cluster centers, i.e. medoids (elements from the original dataset)

        medoid_indices_ : array, shape = (n_clusters,)
            The indices of the medoid rows in X

        labels_ : array, shape = (n_samples,)
            Labels of each point

        inertia_ : float
            Sum of distances of samples to their closest cluster center.

        score_ : float
            Negative of the inertia. The more negative the score, the higher the variation in cluster points, the worse the clustering."""

        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

    def _get_random_state(self, seed=None):

        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        elif isinstance(seed, (int, np.integer)):
            return np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            return seed

    def _is_nonnegative(self, value, variable, strict=True):
        """Checks if the value passed is a non-negative integer which may or may not be equal to zero"""

        if not isinstance(value, (int, np.integer)):
            raise ValueError("%s should be an integer" % (variable))

        if strict:
            isnegative = value > 0
        else:
            isnegative = value >= 0

        if not isnegative:
            raise ValueError("%s should be non-negative" % (variable))
        return isnegative

    def _check_arguments(self):
        """Checks if all the arguments are valid"""

        if (
            self._is_nonnegative(self.n_clusters, "n_clusters")
            and self._is_nonnegative(self.max_iter, "max_iter")
            and (
                isinstance(
                    self.random_state,
                    (int, np.integer, None, np.random.RandomState),
                )
                or self.random_state is np.random
            )
        ):
            pass
        else:
            raise ValueError("Random state is not valid")

        distance_metrics = [
            "euclidean",
            "manhattan",
            "cosine",
            "cityblock",
            "l1",
            "l2",
        ]
        init_methods = ["random", "heuristic"]

        if self.metric not in distance_metrics:
            raise ValueError(
                "%s not a supported distance metric" % (self.metric)
            )

        if self.init not in init_methods:
            raise ValueError(
                "%s not a supported initialization method" % (self.init)
            )

    def _initialize_medoids(self, d_matrix, n_clusters, random_state_object):
        """Implementation of two initialization methods."""

        if self.init == "random":
            """Randomly chooses K points from existing set of samples"""
            return random_state_object.choice(len(d_matrix), n_clusters)

        elif self.init == "heuristic":
            """Chooses initial points as the first K points with shortest total distance to all other points  in the set. Recommended."""
            return np.argpartition(np.sum(d_matrix, axis=1), n_clusters - 1)[
                :n_clusters
            ]

    def _compute_inertia(self, distances):
        """Compute inertia of new samples. Inertia is defined as the sum of the
        sample distances to closest cluster centers.

        Parameters
        ----------
        distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
            Distances to cluster centers.

        Returns
        -------
        Sum of sample distances to closest cluster centers.
        """

        # Define inertia as the sum of the sample-distances
        # to closest cluster centers
        inertia = np.sum(np.min(distances, axis=1))
        return inertia

    def _compute_optimal_swap(
        self, D, medoid_idxs, not_medoid_idxs, Djs, Ejs, n_clusters
    ):
        """Compute best cost change for all the possible swaps.

        Parameters
        ----------
        D : {array-like, sparse matrix}, shape=(n_samples, n_samples)
            Distance matrix.

        medoid_idxs : {array-like}, shape = (n_clusters)
                      Indices of medoid points

        not_medoid_idxs : {array-like}, shape = (n_samples - n_clusters)
                          Indices of non-medoid points

        Djs : {array-like}, shape = (n_samples)
              Distances of each point to closest medoid

        Ejs : {array-like}, shape = (n_samples)
              Distances of each point to next-closest medoid

        n_clusters : integer
                     Number of clusters/medoids


        Returns
        -------
        Sum of sample distances to closest cluster centers."""

        # Initialize best cost change and the associated swap couple.
        best_swap = (1, 1, 0.0)
        # print("Number of samples:", len(D))
        cur_cost_change = 0.0
        not_medoid_shape = len(not_medoid_idxs)
        cluster_i_bool, not_cluster_i_bool, second_best_medoid = (
            None,
            None,
            None,
        )
        not_second_best_medoid = None

        i, j, h = 0, 0, 0
        id_i, id_j, id_h = 0, 0, 0
        # Compute the change in cost for each swap.
        for h in range(not_medoid_shape):
            # id of the potential new medoid.
            id_h = not_medoid_idxs[h]
            # print("\n-------------------------\nConsidering the potential data point ", id_h)
            for i in range(n_clusters):
                # id of the medoid we want to replace.
                id_i = medoid_idxs[i]
                # print("\nConsidering the medoid numbered ",id_i)
                cur_cost_change = 0.0
                # compute for all not-selected points the change in cost
                for j in range(not_medoid_shape):
                    id_j = not_medoid_idxs[j]
                    # print("\nFor the not selected point ", id_j)
                    cluster_i_bool = D[id_i, id_j] == Djs[id_j]
                    # print("\nIs the current point ",id_j," in same cluster as ", id_i,"? : ", cluster_i_bool)

                    second_best_medoid = D[id_h, id_j] < Ejs[id_j]
                    # print("\nIs the current point ",id_j," 's distance from the potential medoid swap point'", id_h,"less than the second closest medoid? :", second_best_medoid)

                    if cluster_i_bool & second_best_medoid:
                        cur_cost_change += D[id_j, id_h] - Djs[id_j]
                    elif cluster_i_bool & (not second_best_medoid):
                        cur_cost_change += Ejs[id_j] - Djs[id_j]
                    elif (not cluster_i_bool) & (D[id_j, id_h] < Djs[id_j]):
                        cur_cost_change += D[id_j, id_h] - Djs[id_j]
                    # print("\nCost change:", cur_cost_change)

                # same for i
                second_best_medoid = D[id_h, id_i] < Ejs[id_i]
                if second_best_medoid:
                    # print("\nCost change increases by distance to swap")
                    cur_cost_change += D[id_i, id_h]
                else:
                    # print("\nCost changes increases by distance to next medoid")
                    cur_cost_change += Ejs[id_i]

                if cur_cost_change < best_swap[2]:
                    best_swap = (id_i, id_h, cur_cost_change)
                    # print("Swap current medoid ", id_i, " with potential medoid ", id_h, " at the cost change of ",cur_cost_change)

        # If one of the swap decrease the objective, return that swap.
        if best_swap[2] < 0:
            return best_swap
        else:
            # print("No good swap")
            return None

    def fit(self, X, Y=None):

        """ X is the training data which must be, in general, a 2D array of dimension n_samples * n_features

        Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features), \
                or (n_samples, n_samples) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """

        self._check_arguments()

        random_state_object = self._get_random_state(self.random_state)

        if Y != None:
            raise Exception("Clustering fit takes only one parameter")

        X = check_array(X, accept_sparse=["csc", "csr"])

        n_samples, n_features = X.shape[0], X.shape[1]

        if self.n_clusters > n_samples:
            raise ValueError(
                f"Number of clusters {self.n_clusters} cannot be greater than number of samples {n_samples}"
            )

        distances = pairwise_distances(X, Y, metric=self.metric)
        # print("Distances:", distances.shape)
        medoids = self._initialize_medoids(
            distances, self.n_clusters, random_state_object
        )  # Initialized medoids.
        d_closest_medoid, d_second_closest_medoid = np.sort(
            distances[medoids], axis=0
        )[[0, 1]]
        labels = None

        for i in range(self.max_iter):
            medoids_copy = np.copy(medoids)
            not_medoids = np.delete(np.arange(len(distances)), medoids)
            # Associate each data point with closest medoid
            labels = np.argmin(distances[medoids, :], axis=0)
            # For each medoid m and each observation o, compute cost of swap
            optimal_swap = self._compute_optimal_swap(
                distances,
                medoids,
                not_medoids,
                d_closest_medoid,
                d_second_closest_medoid,
                self.n_clusters,
            )
            # If cost is current best, keep this medoid and o combo
            if optimal_swap:
                current_medoid, potential_medoid, _ = optimal_swap
                medoids[medoids == current_medoid] = potential_medoid
                d_closest_medoid, d_second_closest_medoid = np.sort(
                    distances[medoids], axis=0
                )[[0, 1]]
            # If cost function decreases (total intra-cluster sum), swap. Else terminate.

            if np.all(medoids_copy == medoids):
                break
            elif i == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )

        self.cluster_centers_ = X[medoids]
        self.labels_ = np.argmin(distances[medoids, :], axis=0)
        self.medoid_indices_ = medoids
        self.inertia_ = self._compute_inertia(self.transform(X))
        self.score_ = -self.inertia_
        # Return self to enable method chaining
        return self

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
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
        return pairwise_distances_argmin(
            X, Y=self.cluster_centers_, metric=self.metric
        )

    def score(self, X):
        """Returns score"""
        return self.inertia_
