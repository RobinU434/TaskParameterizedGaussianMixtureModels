import unittest
import numpy as np
from tpgmm.utils.stochastic import multivariate_gauss_cdf

class TestYourModule(unittest.TestCase):

    def test_multivariate_gauss_cdf(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        mean = np.array([2, 3])
        cov = np.array([[1, 0], [0, 1]])
        result = multivariate_gauss_cdf(data, mean, cov)
        expected = [0.05854983152431917, 0.05854983152431917, 1.9641280346397437e-05]
        np.testing.assert_allclose(result, expected, rtol=1e-5)