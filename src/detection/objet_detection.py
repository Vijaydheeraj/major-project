import os
import cv2
import pandas as pd
from typing import Any

from src.detection.model import model

def process_videos(folder_path: str) -> None:
    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if not filename.endswith('.mp4'):
            continue
        video_path = os.path.join(folder_path, filename)
        process_video(video_path)

def process_video(video_path: str) -> None:
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Erreur: Impossible d'ouvrir la vidéo {video_path}.")

    # Créer une fenêtre pour afficher la vidéo
    window_name = f"Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la vidéo {video_path} ou erreur de lecture.")
            break

        # Traiter l'image et obtenir les résultats
        detections = process_frame(frame)

        # Afficher les résultats dans la vidéo
        for index, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            color = (0, 255, 0)  # Vert pour les objets détectés
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame: Any) -> pd.DataFrame:
    # Détecter les objets dans l'image
    results = model(frame)

    # Récupérer les boîtes englobantes et les confiances
    detections = results[0].boxes.data.cpu().numpy()

    # Convertir les résultats en DataFrame
    detections_df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
    
    # Ajouter les noms des classes
    detections_df['name'] = detections_df['class'].apply(lambda x: model.names[int(x)])

    # Filtrer les objets indésirables
    # TODO: Contraignant --> trouver une solution de remplacement
    detections_df = detections_df[~detections_df['name'].isin(["couch", "chair", "car", "bench", "train"])]

    return detections_df