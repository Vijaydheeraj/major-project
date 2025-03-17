from typing import Any
import pandas as pd
import os
import sys
from inference import get_model
from shapely.geometry import Polygon, box

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.config_loader import load_config, get_windows_detection

config = load_config()
roboflow_api_key, model_id = get_windows_detection(config)
if not os.environ.get('ROBOFLOW_API_KEY'):
    os.environ['ROBOFLOW_API_KEY'] = roboflow_api_key
model_windows = get_model(model_id=model_id)

def detection_windows(frame: Any) -> list[Polygon]:
    """
    Perform empty detection on a single frame.

    Args:
        frame (Any): The frame to perform empty detection on.

    Returns:
        list[Polygon]: The list of polygons.
    """
    results = model_windows.infer(image=frame)[0]

    polygons = []
    for prediction in results.predictions:
        points = [(point.x, point.y) for point in prediction.points]
        polygons.append(Polygon(points))

    return polygons

def filter_occluded_objects(df: pd.DataFrame, occlusion_polygons: list[Polygon]) -> pd.DataFrame:
    """
    Filter the occluded objects from the DataFrame of detections.

    Args:
        df (pd.DataFrame): The DataFrame containing the detections.
        occlusion_polygons (list[Polygon]: The list of polygons (windows).

    Returns:
        pd.DataFrame: The filtered DataFrame containing the detections.
    """
    # Fonction pour vérifier le pourcentage d'occlusion d'un objet
    def is_occluded(row):
        obj_polygon = box(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        obj_area = obj_polygon.area

        for occ_poly in occlusion_polygons:
            intersection = obj_polygon.intersection(occ_poly)
            if intersection.area / obj_area >= 0.75:
                return True
        return False

    # Filtrer les objets non occlus
    return df[~df.apply(is_occluded, axis=1)].reset_index(drop=True)