import os

# Ajouter le répertoire racine au PYTHONPATH
# TODO : Retirer les 2 lignes si fonctionne sur les autres machines
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import pandas as pd
import numpy as np
from typing import Any
from src.detection.model import model
from src.detection.light import model as light_model
from PIL import Image
import torch
from src.detection.light.lowlight_test import enhance_image

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

    # Obtenir les propriétés de la vidéo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Créer un writer pour sauvegarder la vidéo améliorée
    # TODO : Mettre dans une fonction utilitaire dans utils.py et ne plus l'appeler ici
    output_path = video_path.replace('.mp4', '_enhanced.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Créer des fenêtres pour afficher les vidéos
    # TODO : N'afficher que la video amelioree ou la video originale (choix a faire)
    #  avec le traitement de l'image fait dessus en appelant process_frame (après l'appel a la fonction enhance_image)
    window_name_original = "video originale"
    window_name_enhanced = "video amelioree"
    cv2.namedWindow(window_name_original, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name_enhanced, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_original, 640, 360)
    cv2.resizeWindow(window_name_enhanced, 640, 360)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Fin de la vidéo {video_path} ou erreur de lecture.")
                break

            enhanced_frame = enhance_image(frame)
            out.write(enhanced_frame)

            cv2.imshow(window_name_original, frame)
            cv2.imshow(window_name_enhanced, enhanced_frame)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Traitement du frame {frame_count}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur, nettoyage en cours...")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def process_frame(frame: Any) -> pd.DataFrame:
    # # Ameliore l'image pour une meilleure detection
    # frame = enhance_image(frame)

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

process_video("C:/Users/Siddikh/Videos/ProjetCollectif/prise_14")