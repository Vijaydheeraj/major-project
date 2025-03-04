import os
import sys
from inference import get_model
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config_loader import load_config, get_ai_model_empty

config = load_config()
roboflow_api_key, model_id = get_ai_model_empty(config)
if not os.environ.get('ROBOFLOW_API_KEY'):
    os.environ['ROBOFLOW_API_KEY'] = roboflow_api_key
model_empty = get_model(model_id=model_id)

def detection_empty(frame: Any) -> list:
    """
    Perform empty detection on a single frame.

    Args:
        frame (Any): The frame to perform empty detection on.

    Returns:
        list: A list of predictions from the model.
    """
    return model_empty.infer(image=frame)[0].predictions