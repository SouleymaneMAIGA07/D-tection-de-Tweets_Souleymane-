# Détection de Tweets Suspects

Projet de Data Science visant à classifier automatiquement des tweets en deux catégories — **suspect** (label `0`) ou **normal** (label `1`) — à partir d'un jeu de données de 60 000 tweets. Le projet couvre l'ensemble du cycle de vie d'un projet de Machine Learning appliqué au texte : exploration, prétraitement, représentation vectorielle, modélisation, gestion du déséquilibre des classes, versioning avec DVC, évaluation, optimisation et déploiement.

## Auteur

Souleymane MAIGA

## Sommaire

- [Contexte](#contexte)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Pipeline DVC](#pipeline-dvc)
- [Résultats](#résultats)
- [Application Streamlit](#application-streamlit)
- [Rapport complet](#rapport-complet)

## Contexte

Le jeu de données contient 60 000 tweets répartis en deux classes fortement déséquilibrées :
- **Label 1 (normal)** : 89,7 % des tweets
- **Label 0 (suspect)** : 10,3 % des tweets

L'objectif est d'entraîner un modèle capable de détecter les tweets suspects malgré ce déséquilibre.

## Structure du projet

```
projet_tweets/
├── data/
│   └── processed/           # Données nettoyées (générées par le pipeline)
├── models/                  # Modèle et vectorizer entraînés (générés par le pipeline)
├── metrics/                 # Métriques d'évaluation (scores.json)
├── src/
│   ├── preprocess.py        # Nettoyage et prétraitement du texte
│   ├── train.py             # Entraînement du modèle
│   └── evaluate.py          # Évaluation du modèle
├── app.py                   # Application Streamlit de démonstration
├── params.yaml              # Paramètres du pipeline
├── dvc.yaml                 # Définition du pipeline DVC
├── dvc.lock                 # Verrouillage des versions de données/paramètres
├── tweets_dataset.csv.dvc   # Référence DVC vers le dataset
├── requirements.txt         # Dépendances Python
├── Untitled19.ipynb         # Notebook d'analyse exploratoire et d'expérimentation
├── Rapport_Projet.docx      # Rapport complet du projet
└── README.md
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/SouleymaneMAIGA07/D-tection-de-Tweets_Souleymane-.git
cd D-tection-de-Tweets_Souleymane-
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Récupérer les données versionnées avec DVC

```bash
dvc pull
```

> **Note :** le stockage distant DVC configuré pour ce projet est un dossier local (à des fins académiques). Pour une reproduction sur une autre machine, il faudra reconfigurer un remote DVC accessible (Google Drive, S3, etc.) ou utiliser directement `tweets_dataset.csv` fourni dans le dépôt.

## Utilisation

### Reproduire le pipeline complet

```bash
dvc repro
```

Cette commande exécute automatiquement, dans l'ordre :
1. **preprocess** — nettoyage du texte (minuscules, suppression des URLs, des mentions, des caractères spéciaux, des stop words, lemmatisation)
2. **train** — vectorisation TF-IDF et entraînement d'un modèle Random Forest avec gestion du déséquilibre des classes
3. **evaluate** — calcul des métriques (accuracy, precision, recall, F1-score) sauvegardées dans `metrics/scores.json`

### Explorer les analyses

Le notebook `Untitled19.ipynb` contient l'ensemble des analyses exploratoires, comparaisons de modèles et représentations, avec leurs visualisations (distribution des classes, wordclouds, matrice de confusion, courbe ROC, réduction de dimension PCA).

## Pipeline DVC

Le pipeline est défini dans `dvc.yaml` et versionné avec Git et DVC :

| Étape | Script | Entrées | Sorties |
|---|---|---|---|
| `preprocess` | `src/preprocess.py` | `tweets_dataset.csv` | `data/processed/clean_dataset.csv` |
| `train` | `src/train.py` | `clean_dataset.csv`, `params.yaml` | `models/model.pkl`, `models/vectorizer.pkl` |
| `evaluate` | `src/evaluate.py` | `model.pkl`, `vectorizer.pkl` | `metrics/scores.json` |

Le dataset est versionné avec DVC (`tweets_dataset.csv.dvc`) et synchronisé via un stockage distant configuré avec `dvc remote add`.

## Résultats

Trois algorithmes ont été comparés (TF-IDF + gestion du déséquilibre par pondération de classe) :

| Modèle | Accuracy | Precision (suspect) | Recall (suspect) | F1-score (suspect) |
|---|---|---|---|---|
| Régression Logistique | 0,972 | 0,888 | 0,831 | 0,858 |
| Naive Bayes | 0,925 | 0,988 | 0,278 | 0,434 |
| **Random Forest (retenu)** | **0,979** | **0,929** | **0,857** | **0,892** |

Le modèle final (Random Forest) obtient un score AUC de **0,942** et un F1-score moyen de **0,894 (± 0,010)** en validation croisée stratifiée à 5 plis, confirmant sa stabilité.

Détails complets de la méthodologie, des comparaisons et de la discussion critique : voir `Rapport_Projet.docx`.

## Application Streamlit

Une interface interactive permet de tester le modèle en direct :

```bash
streamlit run app.py
```

L'application permet de :
- Saisir un tweet
- Obtenir la prédiction (suspect / normal)
- Visualiser la probabilité associée à chaque classe

## Rapport complet

Le rapport détaillé (introduction, méthodologie, résultats, discussion, limites et perspectives) est disponible dans [`Rapport_Projet.docx`](./Rapport_Projet.docx).
