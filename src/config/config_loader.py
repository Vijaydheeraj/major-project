import json
import os
from typing import Any, Dict

CONFIG_FILE = 'config.json'


def load_config(config_file: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Load the configuration from a JSON file.

    Args:
        config_file (str): The path to the configuration file. Defaults to 'config.json'.

    Returns:
        Dict[str, Any]: The configuration as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def get_video_path(config: Dict[str, Any]) -> str:
    """
    Retrieve the video path from the configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        str: The path to the video files.
    """
    config_video = config.get('videos', {})
    return config_video.get('path', '')


def get_ai_model_detection(config: Dict[str, Any]) -> tuple[str, str]:
    """
    Retrieve the AI model information from the configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        tuple[str, str]: A tuple containing the Roboflow API key and the model ID.
    """
    config_ai = config.get('ai-detection', {})
    return config_ai.get('roboflow_api_key', ''), config_ai.get('model_id', '')


def get_ai_model_empty(config: Dict[str, Any]) -> tuple[str, str]:
    """
    Retrieve the AI model information from the configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        tuple[str, str]: A tuple containing the Roboflow API key and the model ID.
    """
    config_ai = config.get('ai-empty', {})
    return config_ai.get('roboflow_api_key', ''), config_ai.get('model_id', '')

def get_windows_detection(config: Dict[str, Any]) -> tuple[str, str]:
    """
    Retrieve the AI model information from the configuration.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        tuple[str, str]: A tuple containing the Roboflow API key and the model ID.
    """
    config_ai = config.get('ai-windows', {})
    return config_ai.get('roboflow_api_key', ''), config_ai.get('model_id', '')