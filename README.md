# 🏦 Dashboard Marketing Bancaire

> Tableau de bord interactif pour l'analyse et la prédiction de souscription
> à un dépôt à terme bancaire, basé sur un modèle Random Forest.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Description

Ce dashboard interactif a été développé dans le cadre d'un projet de Data Science
sur la prédiction de souscription à un dépôt à terme. Il permet de :

- 📊 **Explorer** les données de manière interactive avec filtres dynamiques
- 🎯 **Prédire** la propension d'un client à souscrire (formulaire interactif)
- 💰 **Simuler** différents scénarios marketing et calculer leur ROI

---

## 🎬 Aperçu

### Architecture multi-pages

| Page | Description |
|---|---|
| 🏠 **Accueil** | Présentation du projet, KPIs clés, résumé de la méthodologie |
| 📊 **Exploration** | Filtres dynamiques + graphiques Plotly + matrice de corrélation |
| 🎯 **Prédiction** | Formulaire client + score de propension + jauge interactive |
| 💰 **Scénarios** | 3 scénarios prédéfinis + simulateur personnalisé + courbe d'optimisation |

---

## 🚀 Installation et lancement

### 1) Cloner ou télécharger le projet

```bash
git clone https://github.com/votre-username/dashboard-marketing-bancaire.git
cd dashboard-marketing-bancaire
```

### 2) Créer un environnement virtuel (recommandé)

```bash
# Sur Windows
python -m venv venv
venv\Scripts\activate

# Sur Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3) Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4) Lancer le dashboard

```bash
streamlit run Accueil.py
```

➡️ Le dashboard s'ouvre automatiquement à l'adresse `http://localhost:8501`

---

## 📁 Structure du projet

```
dashboard-marketing-bancaire/
│
├── Accueil.py                     # Page d'accueil principale
├── pages/
│   ├── 1_📊_Exploration.py        # Exploration interactive des données
│   ├── 2_🎯_Prédiction.py         # Prédiction client par client
│   └── 3_💰_Scénarios.py          # Simulation de scénarios marketing
│
├── bankfull.csv                   # Dataset (45 211 clients × 17 variables)
├── requirements.txt               # Dépendances Python
├── README.md                      # Ce fichier
│
└── (générés au premier lancement)
    ├── rf_model.joblib            # Modèle Random Forest sérialisé
    └── feature_names.joblib       # Noms des features pour la prédiction
```

---

## 🛠️ Stack technique

- **Streamlit** 1.28+ : framework de dashboard
- **Pandas / NumPy** : manipulation de données
- **Scikit-learn** : modèle Random Forest
- **Plotly** : visualisations interactives
- **Joblib** : sérialisation du modèle

---

## 📊 Méthodologie du modèle

Le modèle utilisé est un **Random Forest** entraîné sur le dataset bancaire,
avec les caractéristiques suivantes :

| Paramètre | Valeur |
|---|---|
| Nombre d'arbres | 100 |
| Profondeur max | 10 |
| Class weight | balanced (gère le déséquilibre 88/12) |
| Variable `duration` | **exclue** (data leakage) |

### Performances mesurées

| Métrique | Valeur |
|---|---|
| Accuracy | 82.86% |
| Precision | 35.63% |
| Recall | 57.66% |
| **F1-score** | **44.04%** |
| **ROC-AUC** | **79.21%** |

---

## ☁️ Déploiement en ligne (optionnel)

Pour déployer gratuitement votre dashboard sur Streamlit Cloud :

### 1) Créer un repository GitHub

Pousser tous les fichiers du projet sur GitHub :

```bash
git init
git add .
git commit -m "Initial commit - Dashboard Marketing Bancaire"
git branch -M main
git remote add origin https://github.com/votre-username/votre-repo.git
git push -u origin main
```

### 2) Déployer sur Streamlit Cloud

1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Se connecter avec GitHub
3. Cliquer sur **"New app"**
4. Sélectionner votre repository, la branche `main`, et le fichier `Accueil.py`
5. Cliquer sur **"Deploy"**

Votre dashboard sera en ligne à l'adresse `https://votre-username-votre-repo.streamlit.app` 🚀

---

## 🎯 Fonctionnalités clés

### 📊 Page Exploration
- Filtres dynamiques : âge, profession, statut marital, éducation, prêt immobilier
- Distributions interactives (histogrammes, boxplots)
- Taux de conversion par segment (avec ligne de moyenne)
- Matrice de corrélation interactive
- Téléchargement CSV des données filtrées

### 🎯 Page Prédiction
- Formulaire complet : 13 variables sociodémographiques, bancaires et campagne
- Score de propension affiché en jauge interactive
- Recommandation automatique (Cibler / Déprioriser)
- Top 10 des variables influentes du modèle

### 💰 Page Scénarios
- 3 scénarios prédéfinis : Conservateur (10%) / Équilibré (30%) / Agressif (60%)
- Comparaison avec scénario sans modèle
- Simulateur personnalisé avec slider
- Courbe d'optimisation ROI vs % ciblé
- Recommandation automatique du sweet spot

---

## 📚 Source des données

Le dataset utilisé est issu de l'UCI Machine Learning Repository :
> [Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Il contient les données d'une banque portugaise concernant ses campagnes
marketing par téléphone, avec 45 211 clients et 17 variables.

---

## 👤 Auteur

Projet réalisé dans le cadre d'une formation en Data Science.

---

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier `LICENSE` pour plus de détails.

---

⭐ **N'hésitez pas à mettre une étoile au projet si vous le trouvez utile !**
