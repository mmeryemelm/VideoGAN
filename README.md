# Synthèse de mouvements avec GAN

Ce projet fait partie de notre travail de fin d'année 2024 et est une implémentation de notre recherche sur la synthèse de mouvements. Il utilise Python, PyTorch et d'autres technologies pour modéliser et générer des séquences de mouvements complexes. Ce projet est destiné à fournir des outils et des bibliothèques utiles aux chercheurs et développeurs dans les domaines de l'animation et de la robotique.

## Table des matières
- [Description du générateur](#description-du-générateur)
- [Résultats expérimentaux](#résultats-expérimentaux)
- [Installation](#installation)
- [Configuration de la base de données](#configuration-de-la-base-de-données)
- [Utilisation](#utilisation)
- [Configuration matérielle](#configuration-matérielle)

## Description du générateur
Le générateur utilise un vecteur latent de caractéristiques, combiné avec un bruit gaussien, pour synthétiser des séquences vidéo. Ce réseau est conditionné par des étiquettes représentant différentes classes de mouvements, permettant ainsi de guider la génération vers des types de gestes spécifiques.

L'architecture du générateur intègre deux flux distincts : un flux de premier plan et un flux d'arrière-plan.

Le flux de premier plan est chargé de générer les détails dynamiques des mouvements humains, en se concentrant sur les zones de la vidéo où les actions sont les plus prononcées.
Le flux d'arrière-plan, quant à lui, s'occupe des éléments moins dynamiques, garantissant la stabilité de la scène autour du sujet.
Ces deux flux sont ensuite combinés pour produire une vidéo finale qui offre à la fois des mouvements réalistes et une cohérence visuelle globale.

![Figure du Générateur](https://github.com/mmeryemelm/videoGAN/raw/main/GAN/generateur.jpg)


## Résultats expérimentaux

Ces images sont générées par notre modèle GAN, développé spécifiquement pour la reproduction de mouvements humains. Le modèle a été évalué sur un ensemble de données de test non vues durant l'entraînement, assurant ainsi sa capacité à généraliser. Chaque image illustre un geste distinct, étiqueté pour faciliter l'identification et l'analyse ultérieures. 


![Résultats expérimentaux](https://github.com/mmeryemelm/videoGAN/blob/main/GAN/GANSGIF.gif)





## Installation

Pour exécuter ce projet, installez les dépendances nécessaires en suivant ces étapes :

```bash
git clone https://github.com/votreUsername/nomDuProjet
cd nomDuProjet
pip install -r requirements.txt
```

## Configuration de la base de données

### Étape 1: Téléchargement de la base de données Natops

1. **Clonage du dépôt GitHub :**
   - Ouvrez votre terminal.
   - Entrez la commande :
     ```bash
     git clone https://github.com/yalesong/natops.git
     ```
   - Changez de répertoire avec :
     ```bash
     cd natops
     ```

2. **Exécution du script de téléchargement :**
   - Assurez-vous que les outils `wget` et `unzip` sont installés sur votre machine.
   - Lancez le script :
     ```bash
     ./download_natops.sh
     ```

D'accord, si vous prévoyez de placer les scripts dans le répertoire du projet sur GitHub, c'est une bonne pratique pour garder le fichier `README.md` épuré et pour permettre aux utilisateurs de voir et d'exécuter les scripts directement. Voici comment vous pourriez référencer les scripts dans le `README.md` :

---

### Étape 2: Prétraitement des vidéos

Nous avons opté pour une approche modulaire dans le traitement des données vidéo. Chaque script dans notre pipeline traite une étape distincte de la préparation des données :


#### 1. Récupération des vidéos AVI
Ce script récupère uniquement les fichiers vidéo AVI de la base de données Natops. Vous pouvez trouver le script ici : [`step1videosonly.py`](./step1videosonly.py)

#### 2. Segmentation des vidéos
Ce script segmente les vidéos en utilisant le fichier `segmentation.txt` pour récupérer toutes les répétitions d'une action. Le script est disponible ici : [`step2segmentation.py`](./step2segmentation.py)

#### 3. Organisation des vidéos en dossiers
Ce script arrange les fichiers/vidéos dans une hiérarchie de dossiers selon le sujet, le geste et la répétition, et enregistre les vidéos sous forme de frames. Accédez au script ici : [`step3videotoframe.py`](./step3videotoframe.py)

#### 4. Redimensionnement des images et sélection du nombre de frames
Ce script ajuste la taille des frames à 64x64 et fixe le nombre de frames à 32 pour chaque vidéo. Vous pouvez trouver le script ici : [`step4sizeandframecount.py`](./step4sizeandframecount.py)


## Utilisation

Après l'installation et la configuration, vous pouvez démarrer la synthèse de mouvements en exécutant le script principal. Pour cela, vous avez deux options :

1. Lancez directement le script en utilisant la commande suivante dans votre terminal :

```bash
python trainGAN.py -e 400 -d 3 32 64 64 -zd 250 -nb 8 -l 50 -c 1 -s 1
```

Cette commande spécifie les paramètres de configuration du modèle, notamment :

- **-e 400** : Exécute l'entraînement pour 400 époques.
- **-d 3 32 64 64** : Définit les dimensions de chaque vidéo traitée par le réseau à 3 canaux et une résolution de 32x64x64.
- **-zd 100** : Fixe la dimension du vecteur latent à 100.
- **-nb 8** : Utilise des batches de taille 8 pour l'entraînement.
- **-l 50** : Applique un régularisateur de parcimonie avec un lambda de 50.
- **-c 1** : Commence l'entraînement à partir du checkpoint sauvegardé après la première époque, si disponible.
- **-s 1** : Enregistre un checkpoint après chaque époque.

2. Sinon vous pouvez simplement exécuter avec RUN.

## Configuration matérielle

Ce projet a été développé et testé sur une machine équipée d'un GPU avec 6 Go de mémoire et un processeur graphique ayant une capacité de 7.8, 13.8 TFLOPS.
