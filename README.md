# Projet Collectif

## Description

Ce projet à pour but de détecter la présence d'objets dans le tramway afin de savoir s'il est, ou non, vide. 
Ce projet inclut également des configurations chargées à partir d'un fichier JSON 
et des tests unitaires pour vérifier le bon fonctionnement des routes et des configurations.

## Prérequis

- Python 3.12
- bibliothèques Python : cf. `requirements.txt`

## Structure du projet

- `app/` : contient les fichiers de l'application Flask (cette partie a été mise en pause et n'est pas utilisée 
dans le projet, elle aurait uniquement été utilisée pour regrouper les fonctions et résultats)
- `config/` : contient les fichiers de configuration
- `detection/` : contient les différentes fonctions de détection d'objets
  - `ai/` : contient les fichiers associées à l'IA de détection
  - `background_substraction/` : contient les fichiers associées à la soustraction de fond
  - `light/` : contient les fichiers de détection de luminosité
  - `object_detection.py` : **regroupe l'utilisation des différentes fonctions de détection**
- `unit_tests/` : contient les tests unitaires
- main.py : **fichier principal du projet à exécuter**