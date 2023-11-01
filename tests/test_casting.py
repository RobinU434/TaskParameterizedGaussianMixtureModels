import unittest
import numpy as np
from tpgmm.utils.casting import str_to_list, str_to_ndarray, ssv_to_ndarray


class TestCasting(unittest.TestCase):

    def test_str_to_list(self):
        input_string = "[apple, banana, cherry, date]"
        result = str_to_list(input_string)
        expected = ["apple", "banana", "cherry", "date"]
        self.assertEqual(result, expected)

    def test_str_to_ndarray(self):
        input_string = "[1.1, 2.2, 3.3, 4.4, 5.5]"
        result = str_to_ndarray(input_string)
        expected = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        np.testing.assert_array_equal(result, expected)

    def test_ssv_to_ndarray(self):
        input_string = "1.1 2.2 3.3 4.4 5.5"
        result = ssv_to_ndarray(input_string)
        expected = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        np.testing.assert_array_equal(result, expected)