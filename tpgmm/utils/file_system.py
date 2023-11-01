import logging
from typing import Any, Dict, List
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Loads a YAML file from the file system.

    Args:
        path (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: The loaded YAML file as a dictionary.
    """
    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.fatal(exc)
            config = {}
    return config


def load_txt(path: str) -> List[str]:
    """Loads text data from a file.

    Args:
        path (str): The path to the text file.

    Returns:
        List[str]: The list of strings, each representing a line in the text file.
    """
    with open(path, "r") as file:
        data = file.readlines()
    # remove \n
    for line_idx in range(len(data)):
        data[line_idx] = data[line_idx].rstrip("\n")

    return data


def write_yaml(path: str, content: Dict[str, Any]):
    """Writes a YAML file to the file system.

    Args:
        path (str): The path where the content will be stored.
        content (Dict[str, Any]): The content of the YAML file to be written.
    """
    with open(path, "w") as outfile:
        yaml.dump(content, outfile, default_flow_style=False)
