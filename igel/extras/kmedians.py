"""K-Medians Clustering is an alternative to the popular K-Means Clustering, whereing the goal is to determine the median of the points in a cluster instead of the mean, to represent a cluster centre.

This algorithm differs from the K-Medoids algorithm in that the K-Medoids algorithm requires the cluster centres to be actual data samples, whereas the K-Medians and K-Means algorithms generate synthetic samples not necessarily present in the actual data.

K-Medians seeks to minimize the 1-norm distance from each point to its nearest cluster center, as opposed to K-Means which uses the Euclidean or 2-norm distance."""