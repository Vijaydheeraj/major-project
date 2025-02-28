import os
import sys
import cv2
import pandas as pd
from inference import get_model
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.detection.light.lowlight_test import enhance_image
from src.config.config_loader import load_config, get_ai_model

config = load_config()
roboflow_api_key, model_id = get_ai_model(config)
os.environ['ROBOFLOW_API_KEY'] = roboflow_api_key
model = get_model(model_id=model_id)


def process_videos(folder_path: str) -> None:
    """
    Process all video files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing video files.

    Returns:
        None
    """
    for filename in os.listdir(folder_path):
        if not filename.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, filename)
        process_video(video_path)


def process_video(video_path: str) -> None:
    """
    Process a single video file for object detection.
    Opens a window and displays the result.

    Args:
        video_path (str): The path to the video file.

    Raises:
        IOError: If the video file cannot be opened.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Erreur: Impossible d'ouvrir la vidéo {video_path}.")

    window_name = f"Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la vidéo {video_path} ou erreur de lecture.")
            break

        # Image enhancement for better detection
        #frame = enhance_image(frame)

        # Image processing and results
        detections = process_frame(frame)

        # Show results in video
        for index, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            color = (0, 255, 0)  # Vert pour les objets détectés
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(frame: Any) -> pd.DataFrame:
    """
    Process a single frame for object detection.

    Args:
        frame (Any): The frame to process.

    Returns:
        pd.DataFrame: A DataFrame containing the detection results.
    """
    results = model.infer(frame)[0]

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

    detections_df = pd.DataFrame(detections_list, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])

    return detections_df
