import os
import unittest

import yaml

from tpgmm.utils.file_system import load_txt, load_yaml, write_yaml


class TestYourModule(unittest.TestCase):

    def test_load_yaml(self):
        path = "test_yaml.yaml"
        content = {"key1": "value1", "key2": "value2"}
        with open(path, "w") as f:
            yaml.dump(content, f)
        result = load_yaml(path)
        self.assertEqual(result, content)
        os.remove(path)

    def test_load_txt(self):
        path = "test_txt.txt"
        content = ["line1", "line2", "line3"]
        with open(path, "w") as f:
            for line in content:
                f.write(line + "\n")
        result = load_txt(path)
        self.assertEqual(result, content)
        os.remove(path)

    def test_write_yaml(self):
        path = "test_write_yaml.yaml"
        content = {"key1": "value1", "key2": "value2"}
        write_yaml(path, content)
        with open(path, "r") as f:
            result = yaml.safe_load(f)
        self.assertEqual(result, content)
        os.remove(path)