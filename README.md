# Classification d'Avions Militaires

## Description

Une webapp, où l'utilisateur upload une image d'un avion militaire, et l'application retourne le modèle de l'avion.

## Dataset

https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset

## Processus backend

Tout le processus spécifique à l'IA sera interactif et écrit en Jupyter Notebook

- ETL sur le dataset (dossier cropped), clean-up si nécessaire, identitification des 74 classes.
- Séparation en dataset de training, de validation et de test
- Création, amélioration, fine-tuning du modèle.
- Matrice de confusion, analyse des résultats
- Graphiques matplotlip
- Export du modèle entrainée
- Création d'un serveur web simple à websocket pour recevoir des images de la webapp
- Lancer tensorflow et charger le modèle pré entrainé
- Recevoir des images client, traiter l'image (transformer dans le format attendu) et renvoyer le résultat
-

## Processus frontend

Création d'un frontend simple pour uploader des images et afficher les résultats à l'utilisateur

## Etape 1 (si possible)

Détecter un avion de chasse (ou son absence) de l'image utilisateur. Trouver sa bounding box, et rogner l'image pour ne garder que l'avion et permettre au CNN une meilleure classification.
(étape sous reserve de trouver un dataset assez grand d'images d'avion avec leur bounding box, un échantillon d'une trentaines de samples disponibles sur le dataset)
Si il existe un tel dataset, entrainement d'un CNN de détection de bounding box, appliqué avant l'étape 2.
(https://github.com/tlkh/milair-dataset/tree/master) ?

## Etape 2

Classifier, avec un CNN, et identifier le modèle de l'avion, parmi 74 classes. Si la likelyhood de la classe maximale < un certain threshold, retourner "not found".

## Etape 3

Retourner le résultat et le likelyhood à l'utilisateur sur webapp

## Technologies

Tensorflow, CNN, Websockets, ReactJS
