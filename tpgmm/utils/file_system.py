import logging
from typing import Any, Dict, List
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    """loads yaml file from file system

    Args:
        path (str): where is the yaml file to load

    Returns:
        Dict[str, Any]: loaded yaml file as dictionary
    """
    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.fatal(exc)
            config = {}
    return config


def load_txt(path: str) -> List[str]:
    with open(path, "r") as file:
        data = file.readlines()
    # remove \n
    for line_idx in range(len(data)):
        data[line_idx] = data[line_idx].rstrip("\n")

    return data


def write_yaml(path: str, content: Dict[str, Any]):
    """writes yaml file at path to file system

    Args:
        path (str): where to store content
        content (Dict[str, Any]): content of yaml file
    """
    with open(path, "w") as outfile:
        yaml.dump(content, outfile, default_flow_style=False)