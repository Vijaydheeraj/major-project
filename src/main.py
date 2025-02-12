import os
import sys
import threading
from flask import Flask
# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.app import create_app
from src.detection.objet_detection import process_videos



app = create_app()
'''def run_video_processing():
    process_videos("C:/Users/Siddikh/Videos/ProjetCollectif/prise_14")
    
if __name__ == "__main__":
    # Démarrer le traitement vidéo dans un thread séparé
    video_thread = threading.Thread(target=run_video_processing)
    video_thread.start()

    # Démarrer l'application Flask
    app.run(debug=True)'''

if __name__ == "__main__":
    app.run(debug=True)
    process_videos("C:/Users/Siddikh/Videos/ProjetCollectif/prise_14")