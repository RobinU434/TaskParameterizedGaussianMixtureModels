import time
from typing import Any, Dict, Tuple

import numpy as np
from numpy import ndarray
from sklearn import metrics
from sklearn.cluster import KMeans

from tpgmm.utils.learning_modules import ClassificationModule
from tpgmm.utils.arrays import identity_like
from tpgmm.utils.stochastic import multivariate_gauss_cdf


class TPGMM(ClassificationModule):
    """
    This class in an implementation of the task parameterized gaussian mixture model according to Calinon Paper @https://calinon.ch/papers/Calinon-JIST2015.pdf

    It implements also an Expectation Maximization Algorithm with:  \n
    E-Step:
    \f[
        h_{t, i} = \frac{\pi_i \prod_{j=1}^P \mathcal{N}\left(X_t^{(j)} \mid \mu_i^{(j)}, \Sigma_i^{(j)}\right)}{\sum_{k=1}^K \pi_k \prod_{j=1}^P \mathcal{N}\left(X_t^{(j)} \mid \mu_k^{(j)}, \Sigma_k^{(j)}\right)}
    \f]

    M-Step:
    \f[
        \pi_i \leftarrow \frac{\sum_{t=1}^N h_{t, i}}{N}
    \f]
    \f[
        \mu_i^{(j)} \leftarrow \frac{\sum_{t=1}^N h_{t, i} X_t^{(j)}}{\sum_{t=1}^N h_{t, i}}
    \f]
    \f[
        \Sigma_i^{(j)} \leftarrow \frac{\sum_{t=1}^N h_{t, i} \left(X_t^{(j)} - \mu_i^{(j)}\right)\left(X_t^{(j)} - \mu_i^{(j)}\right)^T}{\sum_{t=1}^N h_{t, i}}
    \f]

    The optimization criterion is the log-likelihood implemented with:
    \f[
        LL = \frac{\sum_{t=1}^N \log\left(\sum_{k=1}^K \pi_k \prod_{j=1}^J\mathcal{N}\left(X_t^{(j)} \mid \mu_k^{(j)}, \Sigma_k^{(j)}\right)\right)}{N}
    \f]

    Variable explanation:\n
    \f$N\f$ ... number of components\n
    \f$\pi\f$ ... weights between components\n
    \f$i\f$ ... component index\n
    \f$j\f$ ... frame index (like pick or place frame)\n
    \f$\mu\f$ ... mean\n
    \f$\Sigma\f$ ... variance / covariance matrix\n
    \f$LL\f$ ... log likelihood

    Examples:
    >>> trajectories = load_trajectories()  # trajectory data in local reference frames with shape: (num_reference_frames, num_points, 3)
    >>> tpgmm = TPGMM(n_components=5)
    >>> tpgmm.fit(trajectories)

    Args:
        n_components (int): number of components
        tol (float): threshold to break from EM algorithm. Defaults to 1e-3.
        max_iter (int): maximum number of iterations to perform the expectation maximization algorithm. Defaults to 100.
        min_iter (int): minimum number of iterations to perform the expectation maximization algorithm. Defaults to 5.
        weights_init (ndarray): initial weights between each component. If set replaces initialization from K-Means. Defaults to None.
        means_init (ndarray): initial means between each component. If set replaces initialization from K-Means. Defaults to None.
        reg_factor (float): regularization factor for empirical covariance matrix. Defaults to 1e-5.
        verbose (bool): Triggers print of learning stats. Defaults to False.

    Attributes:
        weights_: ndarray of shape (n_components,) Weights between gaussian components.
        means_: ndarray of shape (num_frames, n_components, num_features) Mean matrix for each frame and component.
        covariances_: ndarray of shape (num_frames, n_components, num_features, num_features). Covariance matrix for each frame and component
    """

    def __init__(
        self,
        n_components: int,
        threshold: float = 1e-7,
        max_iter: int = 100,
        min_iter: int = 5,
        weights_init: ndarray = None,
        means_init: ndarray = None,
        reg_factor: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        """init class

        Args:
            n_components (int): number of gaussian multidimensional distributions to mix. 
            threshold (float): threshold to break from EM algorithm. Defaults to 1e-3.
            max_iter (int): maximum number of iterations to perform the expectation maximization algorithm. Defaults to 100.
            min_iter (int): minimum number of iterations to perform the expectation maximization algorithm. Defaults to 5.
            weights_init (ndarray): initial weights between each component. If set replaces initialization from K-Means. Defaults to None.
            means_init (ndarray): initial means between each component. If set replaces initialization from K-Means. Defaults to None.
            reg_factor (float): regularization factor for empirical covariance matrix. Defaults to 1e-5.
            verbose (bool): Triggers print of learning stats. Defaults to False.
        """
        super().__init__(n_components)
        self._max_iter = max_iter
        self._min_iter = min_iter
        self._threshold = threshold
        self._reg_factor = reg_factor

        self._verbose = verbose

        self._k_means_algo = KMeans(
            n_clusters=self._n_components, init="k-means++", n_init="auto"
        )
        """KMeans: algorithm to initialize unsupervised clustering"""
        self._cov_reg_matrix = None
        """ndarray: to avoid singularities. shape: (num_frames, n_components, num_features, num_features)"""

        self.weights_ = weights_init
        """ndarray: Weights between gaussian mixture models shape: (n_components)"""

        self.means_ = means_init
        """ndarray: mean matrix for each frame and component: (num_frames, n_components, num_features)"""

        self.covariances_ = None
        """ndarray: covariance matrix for each frame and component. Shape: (num_frames, n_components, num_features, num_features)"""

    def fit(self, X: ndarray) -> None:
        """fits X on the task parameterized gaussian mixture model using K-Means clustering as default initialization method and executes the expectation maximization algorithm: \n
        E-Step:
        self._update_h()

        M-Step:
        self._update_weights_()
        self._update_means_()
        self._update_covariances()

        The optimization criterion is the log-likelihood implemented in:
        self._log_likelihood()

        The algorithm stops if \f$LL_{t-1} - LL_t < \textit{self.tol}\f$ with \f$LL_{t}\f$ as the log-likelihood at time t.

        Args:
            X (ndarray): data tensor to fit the the task parameterized gaussian mixture model on. Expected shape: (num_frames, num_points, num_features)
        """
        # perform k-means clustering
        if self._verbose:
            print("Started KMeans clustering")
        self.means_, self.covariances_ = self._k_means(X)
        self._cov_reg_matrix = identity_like(self.covariances_) * 1e-15

        if self._verbose:
            print("finished KMeans clustering")

        # init weights with uniform probability
        if self.weights_ is None:
            self.weights_ = np.ones(self._n_components) / self._n_components

        if self._verbose:
            print("Started expectation maximization")
        probabilities = self.gauss_cdf(X)
        log_likelihood = self._log_likelihood(probabilities)
        for epoch_idx in range(self._max_iter):
            # Expectation
            h = self._update_h(probabilities)

            # Maximization
            self._update_weights(h)
            self._update_mean(X, h)
            self._update_covariances_(X, h)

            # update probabilities and log likelihood
            probabilities = self.gauss_cdf(X)
            updated_log_likelihood = self._log_likelihood(probabilities)

            # Logging
            difference = updated_log_likelihood - log_likelihood
            if np.isnan(difference):
                raise ValueError("improvement is nan")
            if self._verbose:
                print(
                    f"Log likelihood: {updated_log_likelihood} improvement {difference}"
                )

            # break if threshold is reached
            if (
                difference < self._threshold and epoch_idx >= self._min_iter
            ) or epoch_idx > self._max_iter:
                break

            log_likelihood = updated_log_likelihood

    def predict(self, X: ndarray) -> ndarray:
        """predict cluster labels for each data point in X

        Args:
            X (ndarray): data in local reference frames. Shape (num_frames, num_points, num_features)

        Returns:
            ndarray: the label for each data-point. Shape (num_points)
        """
        probabilities = self.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        return labels

    def predict_proba(self, X: ndarray) -> ndarray:
        """predict cluster labels for each data point

        Args:
            X (ndarray): data in local reference frames. Shape (num_frames, num_points, num_features)

        Returns:
            ndarray: cluster probabilities for each data_point. Shape: (num_points, num_components)
        """
        frame_probs = self.gauss_cdf(X)
        probabilities = np.prod(frame_probs, axis=0).T
        return probabilities

    def silhouette_score(self, X: ndarray) -> float:
        """calculated the silhouette score of the model over the given metric and given data x
        TODO(Marco Todescato): please review this function if the merge for the silhouette score is correct
        Args:
            X (ndarray): data in expected shape: (num_frames, num_points, num_features)
            metric (str): _description_. Defaults to "euclidean".
        """
        labels = self.predict(X)
        scores = np.empty(X.shape[0])
        for frame_idx in range(X.shape[0]):
            scores[frame_idx] = metrics.silhouette_score(X[frame_idx], labels)
        weights = np.tile(self.weights_[:, None], (1, X.shape[0]))
        weighted_sum = (weights @ scores) / (self.weights_ * X.shape[0])
        return weighted_sum.mean()

    def inertia(self, X: ndarray) -> float:
        """Sum of squared distances of samples to their closest cluster center.
        In case of multiple frames we take the mean squared distance to their closest cluster center over all frames.

        TODO(Marco Todescato): please review if this is correct
        Args:
            X (ndarray): data in local reference frames. Shape (num_frames, num_points, num_features)

        Returns:
            float: average inertia score over all frames
        """
        probabilities = self.gauss_cdf(X)
        closest_cluster = np.argmax(probabilities, axis=1)
        # shape: (num_points, num_features, num_frames)
        cluster_center = np.diagonal(self.means_[:, closest_cluster], axis1=0, axis2=1)
        # (num_points, num_features, num_frames) -> (num_frames, num_points, num_features)
        cluster_center = cluster_center.transpose(2, 0, 1)
        # norm: (num_frames, num_points)
        norm = np.linalg.norm(cluster_center, axis=-1)
        # sum of squared distances
        sum_squared = np.sum(np.power(norm, 2), axis=-1)
        return sum_squared.mean()

    def score(self, X: ndarray) -> float:
        """calculate log likelihood score given data

        Args:
            X (ndarray): data tensor with expected shape (num_frames, num_points, num_features)

        Returns:
            float: log likelihood of given data
        """
        probabilities = self.gauss_cdf(X)
        score = self._log_likelihood(probabilities)
        return score

    def bic(self, X: ndarray) -> float:
        """calculates the bayesian information criterion as in

        https://scikit-learn.org/stable/modules/linear_model.html#aic-bic

        Args:
            X (ndarray): data tensor with expected shape (num_frames, num_points, num_features)

        Returns:
            float: bic score
        """
        num_points = X.shape[1]
        ll = self.score(X)
        bic = -2 * ll + np.log(num_points) * self._n_components
        return bic

    def davies_bouldin_score(self, X: ndarray) -> float:
        """calculates the davies bouldin score for each frame and averages them

        # TODO(Marco Todescator): is this score correct?
        Args:
            X (ndarray): data to calculate the score on. Expected shape: (num_frames, num_points, num_features)

        Returns:
            float: score value
        """
        labels = self.predict(X)
        scores = []
        for frame_data in X:
            scores.append(metrics.davies_bouldin_score(frame_data, labels))
        return np.mean(scores)

    def _k_means(
        self,
        X: ndarray,
    ) -> Tuple[ndarray, ndarray]:
        """calculate k means clustering on each frame and calculates the
            empirical covariance matrix for each cluster.

        For more details on k means clustering algorithm please refer to: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        Args:
            X (ndarray): data in feature space for k means clustering.
                shape: (num_frames, num_points, num_features)

        Returns:
            Tuple[ndarray, ndarray]:
             - means with shape: (num_frames, n_components, num_features)
             - covariance matrix with shape: (num_frames, n_components, num_features, num_features)
        """
        means = []
        covariances = []
        for frame_data in X:
            self._k_means_algo.fit(frame_data)
            # get mean
            means.append(self._k_means_algo.cluster_centers_)
            # get empirical covariance matrix
            covariance = []
            for cluster_idx in range(self._n_components):
                data_idx = np.argwhere(
                    self._k_means_algo.labels_ == cluster_idx
                ).squeeze()
                covariance.append(np.cov(frame_data[data_idx].T))
            covariances.append(
                np.stack(covariance)
            )  # shape: (n_components, num_features, num_features)

        covariances = np.stack(covariances)
        # regularization
        reg_matrix = identity_like(covariances) * self._reg_factor
        covariances += reg_matrix

        return np.stack(means), covariances

    def gauss_cdf(self, X: ndarray) -> ndarray:
        """calculate the gaussian probability for a given data set.
        \f[
            \mathcal{N}\left(X_t^{(j)} \mid \mu_i^{(j)}, \Sigma_i^{(j)}\right) = \frac{1}{\sqrt{(2\pi)^D|\Sigma_i^{(j)}|}} \exp \left( \frac{1}{2}(X_t^{(j)} - \mu_i^{(j)})\Sigma_i^{(j), -1}(X_t^{(j)} - \mu_i^{(j)})^T\right)
        \f]

        Variable explanation:
        D ... number of features

        Args:
            X (ndarray): data with shape: (num_frames, num_points, num_features)

        Returns:
            ndarray: probability shape (num_frames, n_components, num_points)
        """
        probs = []
        # to prevent singularity matrices
        covariances = self.covariances_ + self._cov_reg_matrix
        for frame_data, frame_means, frame_covariances in zip(
            X, self.means_, covariances
        ):
            cluster_probs = []
            for cluster_mean, cluster_cov in zip(frame_means, frame_covariances):
                cluster_probs.append(
                    multivariate_gauss_cdf(frame_data, cluster_mean, cluster_cov)
                )
            probs.append(np.stack(cluster_probs))
        return np.stack(probs)

    def _update_h(self, probabilities: ndarray) -> ndarray:
        """update h parameter according to equation 49 in appendix 1
        \f[
            h_{t, i} = \frac{\pi_i \prod_{j=1}^P \mathcal{N}\left(X_t^{(j)} \mid \mu_i^{(j)}, \Sigma_i^{(j)}\right)}{\sum_{k=1}^K \pi_k \prod_{j=1}^P \mathcal{N}\left(X_t^{(j)} \mid \mu_k^{(j)}, \Sigma_k^{(j)}\right)}
        \f]

        Args:
            data (ndarray): shape: (num_frames, num_points, num_features)
            probabilities (ndarray): shape (num_frames, n_components, num_points)
        Returns:
            ndarray: h-parameter. shape: (n_components, num_points)
        """
        cluster_probs = np.prod(
            probabilities, axis=0
        )  # shape: (n_components, num_points)
        numerator = (
            self.weights_ * cluster_probs.T
        ).T  # shape: (n_components, num_points)
        denominator = np.sum(numerator, axis=0)  # shape: (num_points)
        return numerator / denominator

    def _update_weights(self, h: ndarray) -> None:
        """update pi (weights parameter according to equation 50).
        \f[
            \pi_i \leftarrow \frac{\sum_{t=1}^N h_{t, i}}{N}
        \f]

        Args:
            h (ndarray): shape: (n_components, num_points)
        """
        self.weights_ = np.mean(h, axis=1)

    def _update_mean(self, X: ndarray, h: ndarray) -> None:
        """updates the mean parameter according to equation 51.
        \f[
            \mu_i^{(j)} \leftarrow \frac{\sum_{t=1}^N h_{t, i} X_t^{(j)}}{\sum_{t=1}^N h_{t, i}}
        \f]

        Args:
            X (ndarray): shape: (num_frames, num_points, num_features)
            h (ndarray): shape: (n_components, num_points)
        """
        num_frames, _, num_features = X.shape
        # reshape X into -> (num_frames, num_components, num_points, num_features)
        X = np.tile(X[:, None, :, :], (1, self._n_components, 1, 1))
        # reshape h into -> (num_frames, num_components, num_points, num_features)
        h = np.tile(h[None, :, :, None], (num_frames, 1, 1, num_features))

        numerator = np.sum(h * X, axis=2)
        denominator = np.sum(h, axis=2)
        self.means_ = numerator / denominator

    def _update_covariances_(self, X: ndarray, h: ndarray) -> None:
        """updates the covariance parameter according to equation 52
        \f[
            \Sigma_i^{(j)} \leftarrow \frac{\sum_{t=1}^N h_{t, i} \left(X_t^{(j)} - \mu_i^{(j)}\right)\left(X_t^{(j)} - \mu_i^{(j)}\right)^T}{\sum_{t=1}^N h_{t, i}}
        \f]

        Args:
            X (ndarray): shape: (num_frames, num_points, num_features)
            h (ndarray): shape: (n_components, num_points)
        """

        # TODO: make faster without 3 nested for loops

        # ========================================================================
        # THIS BLOCKCOMMENT IS AN EXPERIMENT TO ONLY USE NP.EINSUM INSTEAD OF FOR-
        # LOOPS IT TURNS OUT THAT THIS VERSION IS ONLY BENEFICIAL WITH A HIGH
        # NUMBER OF FRAMES AND COMPONENTS REASON IS PROBABLY THE MATRIX EXPANSION
        # ========================================================================
        # >>>>>
        # num_frames, num_points, num_features = X.shape
        # # expand X array into: [num_frames, num_components, num_points, num_features]
        # expand_data = np.tile(X[:, None, :, :], (1, self.n_components, 1, 1))
        # # expand means into: [num_frames, num_components, num_points, num_features]
        # expand_means = np.tile(self.means_[:, :, None, :], (1, 1, num_points, 1))
        # # compute: (x- mean) @ (x-mean).T
        # centered = expand_data - expand_means
        # # i: num_frames, j: num_components, k: num_points, l, m: num_features
        # squared_mat = np.einsum("ijkl,ijkm->ijklm", centered, centered)

        # # weighted sum with h
        # # i: num_frames, j: num_components, k: num_points, l, m: num_features
        # weighted_sum = np.einsum("ijklm,jk->ijlm", squared_mat, h)
        # normalizer = np.sum(h, axis=1)
        # # expand dimension to [num_frames, num_components, num_points, num_features, num_features]
        # normalizer = np.tile(
        #     normalizer[None, :, None, None],
        #     (num_frames, 1, num_features, num_features),
        # )
        # cov = weighted_sum / normalizer
        # <<<<<
        cov = []
        for frame_data, frame_mean in zip(X, self.means_):
            frame_cov = []
            for component_mean, component_h in zip(frame_mean, h):
                # mat_aggregation = []
                # for point in frame_data:
                #     centered = point - component_mean
                #     centered = np.expand_dims(centered, axis=1)
                #     mat = centered @ centered.T
                #     mat_aggregation.append(mat)
                # mat_aggregation = np.stack(mat_aggregation)
                centered = frame_data - component_mean
                # shape: (num_points, num_features, num_features)
                mat_aggregation = np.einsum("ij,ik->ijk", centered, centered)
                # swap dimensions to: (num_features, num_features, num_points)
                mat_aggregation = mat_aggregation.transpose(1, 2, 0)
                # weighted sum and division by h. shape: (num_features, num_features)
                cov_mat = (mat_aggregation @ component_h) / component_h.sum()
                frame_cov.append(cov_mat)
            cov.append(np.stack(frame_cov))
        cov = np.stack(cov)

        # shape: (num_frames, num_num_features, num_features)
        self.covariances_ = cov

    def _log_likelihood(self, probabilities: ndarray) -> float:
        """calculates the log likelihood of given probabilities
        \f[
            LL = \frac{\sum_{t=1}^N \log\left(\sum_{k=1}^K \pi_k \prod_{j=1}^J\mathcal{N}\left(X_t^{(j)} \mid \mu_k^{(j)}, \Sigma_k^{(j)}\right)\right)}{N}
        \f]

        Args:
            probabilities (ndarray): shape: (num_frames, n_components, num_points)

        Returns:
            float: log likelihood
        """
        probabilities = np.prod(probabilities, axis=0)
        # reshape to: (num_points, n_components)
        probabilities = probabilities.T
        weighted_sum = probabilities @ self.weights_  # shape (num_points)
        return np.mean(np.log(weighted_sum)).item()

    @property
    def config(self) -> Dict[str, Any]:
        config = {
            "max_iter": self._max_iter,
            "min_iter": self._min_iter,
            "threshold": self._threshold,
            "reg_factor": self._reg_factor,
        }

        config = {**config, **super().config}
        return config
