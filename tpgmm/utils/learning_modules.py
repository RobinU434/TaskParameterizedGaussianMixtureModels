from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

import numpy as np
from numpy import ndarray
from sklearn.metrics import davies_bouldin_score


class LearningModule(ABC):
    """Basic abstract class for a generic learning module"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"No fit method implemented for class {type(self).__name__}"
        )

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            f"No predict method for class {type(self).__name__} implemented"
        )

    def fit_predict(self, X: ndarray) -> ndarray:
        """convenience method; equivalent to calling fit(data) followed by predict(data).

        Args:
            data (ndarray): data in local reference frames. Shape (num_frames, num_points, num_features)

        Returns:
            ndarray: the label for each data-point. Shape (num_points)
        """
        self.fit(X)
        return self.predict(X)

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """get all model config parameters

        Returns:
            Dict[str, Any]: configuration parameters
        """
        pass


class RegressionModel(LearningModule, ABC):
    """Basic Regression Model
    Implements all interfaces and common methods for regression models
    """

class ClassificationModule(LearningModule, ABC):
    """Basic Classification Model
    Implements all interfaces and common methods for classification models
    """

    def __init__(self, n_components: int) -> None:
        super().__init__()
        self._n_components = n_components

    @abstractmethod
    def predict_proba(self, *args, **kwargs):
        logging.warning(
            f"No predict_proba method for class {type(self).__name__} implemented"
        )

    @abstractmethod
    def score(self, X: ndarray) -> float:
        """calculates score function from descendent

        Often score is calculated on the optimization objective

        Args:
            X (ndarray): data to calculate score on.

        Returns:
            float: score
        """
        pass

    @abstractmethod
    def silhouette_score(self, X: ndarray) -> float:
        logging.warning(
            f"No silhouette score method for class {type(self).__name__} implemented"
        )

    @abstractmethod
    def inertia(self, X: ndarray) -> float:
        """Sum of squared distances of samples to their closest cluster center.

        Args:
            X (ndarray): data in local reference frames. Shape (num_frames, num_points, num_features)

        Returns:
            float: intertia
        """
        logging.warning(
            f"No inertia score method for class {type(self).__name__} implemented."
        )

    def davies_bouldin_score(self, X: ndarray) -> float:
        """calculates davies_bouldin_score on given data
        \f[
            DB = \frac{1}{n} \sum_{i,j=1}^N \max_{j!=i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}
        \f]
        Where:
        N ... number of datapoints
        d() ... distance function
        c_i ... cluster center c_i
        sigma_i ... the average distance of all points in cluster i from the cluster centre ci

        Args:
            X (ndarray): data in the shape for self.predict()

        Returns:
            float: score value
        """
        label = self.predict(X)
        score = davies_bouldin_score(X, label)
        return score

    def bic(self, X: ndarray) -> float:
        """calculates the bayesian information criterion as in

        https://scikit-learn.org/stable/modules/linear_model.html#aic-bic

        Args:
            X (ndarray): data tensor with expected shape (num_points, num_features)

        Returns:
            float: bic score
        """
        num_points = X.shape[0]
        ll = self.score(X)
        bic = -2 * ll + np.log(num_points) * self._n_components
        return bic

    def aic(self, X: ndarray):
        """calculates the Akaike information criterion as in
        https://scikit-learn.org/stable/modules/linear_model.html#aic-bic

        Args:
            X (ndarray): data tensor with expected shape same form self.score(X)

        Returns:
            float: aic score
        """
        ll = self.score(X)
        aic = -2 * ll + self._n_components
        return aic

    @property
    def config(self) -> Dict[str, Any]:
        return {"n_components": self._n_components}

