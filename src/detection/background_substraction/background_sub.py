import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.detection.light.equalization.light_fast import enhance_brightness
from src.detection.windows.manual.windows import define_occlusion_parallelograms

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# Construct the absolute paths for the images
frame_ref_cam4_path = os.path.join(project_dir, "images/frame_ref_cam4_lightV2.jpg")
frame_ref_cam5_path = os.path.join(project_dir, "images/frame_ref_cam5_lightV2.jpg")
frame_ref_cam7_path = os.path.join(project_dir, "images/frame_ref_cam7_lightV2.jpg")
frame_ref_cam8_path = os.path.join(project_dir, "images/frame_ref_cam8_lightV2.jpg")

# Load reference frames using the absolute paths
frame_ref_cam4 = cv2.imread(frame_ref_cam4_path)
frame_ref_cam5 = cv2.imread(frame_ref_cam5_path)
frame_ref_cam7 = cv2.imread(frame_ref_cam7_path)
frame_ref_cam8 = cv2.imread(frame_ref_cam8_path)


def match_frame_reference(camera: int) -> np.ndarray:
    """
    Matches the camera number to the corresponding reference frame.

    Args:
        camera (int): The camera number.

    Returns:
        np.ndarray: The reference frame corresponding to the camera number.
    """
    match camera:
        case 4:
            frame_ref = frame_ref_cam4
        case 5:
            frame_ref = frame_ref_cam5
        case 7:
            frame_ref = frame_ref_cam7
        case 8:
            frame_ref = frame_ref_cam8
        case _:
            print("Erreur : Camera non reconnue")
            exit()
    return frame_ref



def background_subtraction_on_edges(camera: int, frame_tested: np.ndarray) -> pd.DataFrame:
    """
        Performs background subtraction to detect objects in a video frame using edge detection.
        The function returns a DataFrame containing the detected objects with their bounding box coordinates.

        Args:
            camera (int): The camera number.
            frame_tested (np.ndarray): The frame to be tested.

        Returns:
            pd.DataFrame: A DataFrame containing the detected objects with their bounding box coordinates.
    """
    detections_list = []

    # Select the reference frame corresponding to the camera number 
    frame_ref = match_frame_reference(camera)
        
    # Luminosity treatment
    frame_cur_light = enhance_brightness(frame_tested)

    # --- DEFINE RECTANGLES FOR EXCLUSION ZONES ---

    coord = define_occlusion_parallelograms(camera)

    # Define the points of the parallelograms
    parallelograms = []

    for points in coord:
        parallelogram = np.array(points, np.int32)
        parallelograms.append(parallelogram)

    # Convert to grayscale
    gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(frame_cur_light, cv2.COLOR_BGR2GRAY)

    # Apply a blur to reduce noise
    gray_ref = cv2.GaussianBlur(gray_ref, (5,5), 0)
    gray_cur = cv2.GaussianBlur(gray_cur, (5,5), 0)

    # Apply Canny edge detection
    edges_ref = cv2.Canny(gray_ref, 250, 300) # (frame, minVal, maxVal)
    edges_cur = cv2.Canny(gray_cur, 250, 300)

    # Edge subtraction
    diff = cv2.absdiff(edges_ref, edges_cur)

    # Threshold to detect significant differences
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)


    # Remove window zones (set pixels in this region to zero)
    for parallelogram in parallelograms:
        cv2.fillPoly(thresh, [parallelogram], 0)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect contours of present objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small objects
    min_size = 25  # (minimum size in pixels)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_size ** 2]

    # Draw detected objects on the current image
    output = frame_tested.copy()
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        detections_list.append([x, y, x + w, y + h, 0, None, None])

    detections_df = pd.DataFrame(detections_list,
                                     columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])

    return detections_df


def background_subtraction(camera: int, frame_tested: np.ndarray) -> pd.DataFrame:
    """
        Performs background subtraction to detect objects in a video frame.
        The function returns a DataFrame containing the detected objects with their bounding box coordinates.

        Args:
            camera (int): The camera number.
            frame_tested (np.ndarray): The frame to be tested.

        Returns:
            pd.DataFrame: A DataFrame containing the detected objects with their bounding box coordinates.
    """
    detections_list = []

    # Select the reference frame corresponding to the camera number 
    frame_ref = match_frame_reference(camera)
     
    # Luminosity treatment
    frame_cur_light = enhance_brightness(frame_tested)

    # --- DEFINE RECTANGLES FOR EXCLUSION ZONES ---
    coord = define_occlusion_parallelograms(camera)

    # Define the points of the parallelograms
    parallelograms = []

    for points in coord:
        parallelogram = np.array(points, np.int32)
        parallelograms.append(parallelogram)

        # Convert to grayscale for subtraction
        gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.cvtColor(frame_cur_light, cv2.COLOR_BGR2GRAY)

        # Apply image subtraction
        diff = cv2.absdiff(gray_ref, gray_cur)

        # Threshold to detect significant differences
        _, thresh = cv2.threshold(diff, 120, 255, cv2.THRESH_BINARY)


    # Remove window zones (set pixels in this region to zero)
    for parallelogram in parallelograms:
        cv2.fillPoly(thresh, [parallelogram], 0)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect contours of present objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small objects
    min_size = 25  # (minimum size in pixels)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_size ** 2]

    # Draw detected objects on the current image
    output = frame_tested.copy()
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        detections_list.append([x, y, x + w, y + h, None, None, None])

    detections_df = pd.DataFrame(detections_list,
                                 columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])

    return detections_df



