import os
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional

def extract_camera_data(video_path: str) -> Tuple[int, Optional[str]]:
    """
    Extract the camera number and time from the video filename.

    Args:
        video_path (str): The path to the video file.

    Returns:
        Tuple[int, Optional[str]]: A tuple containing the data of the camera.
    """
    filename = os.path.basename(video_path)
    match = re.search(r'CAM(\d+)', filename)
    camera_number = int(match.group(1)) if match else 0

    time_match = re.search(r'(\d{2}h\d{2}m\d{2}s)', filename)
    time_str = time_match.group(1) if time_match else None

    return camera_number, time_str

def draw_rectangle(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
    """
    Draw a rectangle on the given image.

    Args:
        image (np.ndarray): The image on which to draw the rectangle.
        top_left (Tuple[int, int]): The top-left corner of the rectangle.
        bottom_right (Tuple[int, int]): The bottom-right corner of the rectangle.
        color (Tuple[int, int, int]): The color of the rectangle.
        thickness (int): The thickness of the rectangle lines.

    Returns:
        None
    """
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

def draw_text(image: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int], font_scale: float, thickness: int) -> None:
    """
    Draw text on the given image.

    Args:
        image (np.ndarray): The image on which to draw the text.
        text (str): The text to draw.
        position (Tuple[int, int]): The position to draw the text.
        color (Tuple[int, int, int]): The color of the text.
        font_scale (float): The scale of the font.
        thickness (int): The thickness of the text.

    Returns:
        None
    """
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_parallelogram(image: np.ndarray, pts: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int) -> None:
    """
    Draw a parallelogram on the given image.

    Args:
        image (np.ndarray): The image on which to draw the parallelogram.
        pts (List[Tuple[int, int]]): The points defining the parallelogram.
        color (Tuple[int, int, int]): The color of the parallelogram.
        thickness (int): The thickness of the parallelogram lines.

    Returns:
        None
    """
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)