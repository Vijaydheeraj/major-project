from ultralytics import YOLO
import pandas as pd
from typing import Any

model_yolo = YOLO("yolo11n.pt")

def detection_yolov11(frame: Any) -> pd.DataFrame:
    """
    Perform object detection on a single frame using YOLO

    Args:
        frame (Any): The frame to perform object detection on.

    Returns:
        pd.DataFrame: The DataFrame containing the detected objects.
    """
    results = model_yolo(frame, verbose=False)

    # Récupérer les boîtes englobantes et les confiances
    detections = results[0].boxes.data.cpu().numpy()

    # Convertir les résultats en DataFrame
    detections_df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
    detections_df['name'] = detections_df['class'].apply(lambda x: model_yolo.names[int(x)])

    return detections_df