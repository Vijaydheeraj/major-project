from shapely.geometry import Polygon, box
import pandas as pd
from typing import List, Tuple

def define_occlusion_parallelograms(camera: int) -> List[List[Tuple[int, int]]]:
    coord = []
    match camera:
        case 4:
            x1, y1, x2, y2 = 730, 90, 1020, 100 # (x1, y1) top left, (x2, y2) top right of the parallelogram
            x3, y3, x4, y4 = 1098, 120, 1250, 138
            x5, y5, x6, y6 = 0, 0, 210, 0
            x7, y7, x8, y8 = 400, 90, 690, 80
            x9, y9, x10, y10 = 565, 150, 700, 150
            x11, y11, x12, y12 = 380, 160, 460, 160
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 330), (x1, y1 + 245)]) # (top left, top right, bottom right, bottom left) coordinates of the parallelogram
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 390), (x3, y3 + 370)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 800), (x5, y5 + 800)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 70), (x7, y7 + 67)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 175), (x9, y9 + 135)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 65), (x11, y11 + 60)])
        case 5:
            x1, y1, x2, y2 = 795, 90, 1065, 100
            x3, y3, x4, y4 = 1125, 120, 1275, 155
            x5, y5, x6, y6 = 455, 75, 750, 75
            x7, y7, x8, y8 = 0, 0, 210, 0
            x9, y9, x10, y10 = 620, 130, 655, 130
            x11, y11, x12, y12 = 425, 140, 515, 140
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 325), (x1, y1 + 230)])
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 378), (x3, y3 + 350)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 50), (x5, y5 + 60)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 800), (x7, y7 + 800)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 145), (x9, y9 + 135)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 70), (x11, y11 + 35)])
        case 7:
            x1, y1, x2, y2 = 770, 90, 1040, 100
            x3, y3, x4, y4 = 1125, 120, 1265, 147
            x5, y5, x6, y6 = 425, 75, 730, 75
            x7, y7, x8, y8 = 0, 0, 210, 0
            x9, y9, x10, y10 = 610, 130, 645, 130
            x11, y11, x12, y12 = 415, 140, 500, 140
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 320), (x1, y1 + 230)])
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 378), (x3, y3 + 350)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 50), (x5, y5 + 60)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 800), (x7, y7 + 800)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 145), (x9, y9 + 135)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 70), (x11, y11 + 35)])
        case 8:
            x1, y1, x2, y2 = 800, 90, 900, 100
            x3, y3, x4, y4 = 675, 95, 744, 100
            x5, y5, x6, y6 = 1085, 140, 1279, 160
            x7, y7, x8, y8 = 901, 100, 1070, 115
            x9, y9, x10, y10 = 0, 0, 210, 0
            x11, y11, x12, y12 = 355, 90, 524, 80
            coord.append([(x1, y1), (x2, y2), (x2, y2 + 190), (x1, y1 + 180)])
            coord.append([(x3, y3), (x4, y4), (x4, y4 + 174), (x3, y3 + 184)])
            coord.append([(x5, y5), (x6, y6), (x6, y6 + 360), (x5, y5 + 335)])
            coord.append([(x7, y7), (x8, y8), (x8, y8 + 330), (x7, y7 + 295)])
            coord.append([(x9, y9), (x10, y10), (x10, y10 + 800), (x9, y9 + 800)])
            coord.append([(x11, y11), (x12, y12), (x12, y12 + 40), (x11, y11 + 40)])
        case _:
            print("Erreur : Camera non reconnue")
            exit()
    return coord


def filter_occluded_objects(df: pd.DataFrame, camera_number: int) -> pd.DataFrame:
    """
    Filter the occluded objects from the DataFrame of detections.

    Args:
        df (pd.DataFrame): The DataFrame containing the detections.
        camera_number (int): The camera number.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the detections.
    """
    # Define the occlusion zones
    occlusion_zones = define_occlusion_parallelograms(camera_number)

    # Convertir les zones d'occlusion en objets Polygon
    occlusion_polygons = [Polygon(zone) for zone in occlusion_zones]

    # Fonction pour vérifier le pourcentage d'occlusion d'un objet
    def is_occluded(row):
        obj_polygon = box(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        obj_area = obj_polygon.area

        for occ_poly in occlusion_polygons:
            intersection = obj_polygon.intersection(occ_poly)
            if intersection.area / obj_area >= 0.95:
                return True
        return False

    # Filtrer les objets non occlus
    return df[~df.apply(is_occluded, axis=1)].reset_index(drop=True)
