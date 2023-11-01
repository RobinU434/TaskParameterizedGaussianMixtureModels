import unittest
import numpy as np
from tpgmm.utils.arrays import subscript, identity_like, get_subarray


class TestArrays(unittest.TestCase):

    def test_subscript(self):
        # Test when all arguments are integers
        result = subscript(1, 2, 3)
        expected = (slice(1, 2), slice(2, 3), slice(3, 4))
        self.assertEqual(result, expected)

        # Test when some arguments are lists
        result = subscript(1, [2, 3], 4)
        expected = (slice(1, 2), [2, 3], slice(4, 5))
        self.assertEqual(result, expected)

    def test_identity_like(self):
        # Test when the input array has the last two dimensions equal
        input_array = np.zeros((3, 3, 3))
        result = identity_like(input_array)
        expected = np.stack([np.eye(3)] * 3)
        np.testing.assert_array_equal(result, expected)

        # Test when the last two dimensions are not equal
        input_array = np.zeros((3, 3, 4))
        with self.assertLogs(level='ERROR'):
            result = identity_like(input_array)
            self.assertIsNone(result)

    def test_get_subarray(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        axes = [0, 1]
        indices = [[0, 1], [1, 2]]
        result = get_subarray(data, axes, indices)
        expected = np.array([[2, 3], [5, 6]])
        np.testing.assert_array_equal(result, expected)

        # Test when not all indices are lists
        with self.assertRaises(AssertionError):
            get_subarray(data, axes, [0, 1])
