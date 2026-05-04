# 🏦 Dashboard Marketing Bancaire

Tableau de bord interactif pour l'analyse et la prédiction de souscription
à un dépôt à terme bancaire, basé sur la méthodologie CRISP-DM.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Description

Application Streamlit single-page avec onglets, développée dans le cadre d'un
projet de Data Science. Elle permet d'**explorer**, **segmenter**, **modéliser**
et **optimiser** les campagnes de marketing bancaire.

## 🎯 Fonctionnalités

- 🏠 **Vue d'ensemble** — KPIs, contexte, méthodologie CRISP-DM
- 📊 **Analyse exploratoire (EDA)** — filtres, distributions, taux de conversion par segment, corrélations
- 🎯 **Segmentation** — K-Means + ACP avec interprétation métier
- 🤖 **Modélisation** — comparaison de 4 modèles (LR, KNN, DT, RF)
- 💰 **Décision marketing** — optimisation du seuil pour maximiser le profit / ROI
- 🔮 **Prédiction client** — formulaire interactif + score de propension

## 🚀 Lancement local

```bash
pip install -r requirements.txt
streamlit run Accueil.py
```

Le dashboard s'ouvre sur `http://localhost:8501` 🚀

## 🛠️ Stack technique

- **Streamlit** — framework de dashboard
- **Pandas / NumPy** — manipulation des données
- **Scikit-learn** — modèles ML (4 algos comparés)
- **Matplotlib / Seaborn** — visualisations stylisées

## 📊 Performances mesurées (Random Forest retenu)

| Métrique | Valeur |
|---|---|
| Accuracy | ~83% |
| Precision | ~36% |
| Recall | ~58% |
| **F1-score** | **~44%** |
| **ROC-AUC** | **~79%** |

## 📁 Structure

```
dashboard/
├── Accueil.py            # Application Streamlit (single-page avec onglets)
├── bankfull.csv          # Dataset (45 211 clients × 17 variables)
├── requirements.txt      # Dépendances
└── README.md
```

## 📚 Source des données

Bank Marketing Data Set — UCI Machine Learning Repository

---

⭐ Fait avec ❤️ par Djakpa
