from typing import Any
import pandas as pd
import os
import sys
from inference import get_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config_loader import load_config, get_ai_model_detection

config = load_config()
roboflow_api_key, model_id = get_ai_model_detection(config)
if not os.environ.get('ROBOFLOW_API_KEY'):
    os.environ['ROBOFLOW_API_KEY'] = roboflow_api_key
model_detection = get_model(model_id=model_id)

def detection_yolov11_fine_tuning(frame: Any) -> pd.DataFrame:
    """
    Perform object detection on a single frame using YOLOv1.1.

    Args:
        frame (Any): The frame to perform object detection on.

    Returns:
        pd.DataFrame: The DataFrame containing the detected objects.
    """
    results = model_detection.infer(frame)[0]

    # Convert predictions into DataFrame
    detections_list = []
    for prediction in results.predictions:
        x1 = prediction.x - prediction.width / 2
        y1 = prediction.y - prediction.height / 2
        x2 = prediction.x + prediction.width / 2
        y2 = prediction.y + prediction.height / 2
        confidence = prediction.confidence
        class_id = prediction.class_id
        class_name = prediction.class_name

        detections_list.append([x1, y1, x2, y2, confidence, class_id, class_name])

    detections_df = pd.DataFrame(detections_list,
                                 columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])

    return detections_df