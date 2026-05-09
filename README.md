# 🚦 Dashboard Trafic Routier Parisien

Tableau de bord interactif d'analyse des données de trafic routier de la
**Ville de Paris** — capteurs permanents (boucles électromagnétiques),
**1 048 575 mesures** sur **13 mois** et **2 985 tronçons**.

![Python](https://img.shields.io/badge/Python-3.9+-2A5C4D.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-D4A82A.svg)
![License](https://img.shields.io/badge/License-MIT-B33A3A.svg)

---

## 🎯 Description

Ce dashboard transforme **plus d'1 million de mesures de trafic** en
visualisations claires et actionnables. Il permet à un service Voirie de :

- 📊 **Explorer** les distributions de débit et d'occupation
- ⏱️ **Visualiser** les patterns horaires, hebdomadaires et saisonniers
- 🗺️ **Cartographier** les zones de congestion sur Paris
- 🚦 **Analyser** chaque tronçon individuellement (drill-down)

## 🎨 Architecture

| Onglet | Contenu |
|---|---|
| 🏠 **Vue d'ensemble** | KPIs, méthodologie CRISP-DM, top tronçons |
| 📊 **Exploration** | Distributions, **diagramme fondamental du trafic** |
| ⏱️ **Tendances temporelles** | Profil 24h, semaine vs week-end, **heatmap** |
| 🗺️ **Cartographie** | Carte interactive Folium (marqueurs + heatmap) |
| 🚦 **Tronçons** | Drill-down par recherche de rue |

## 🚀 Lancement local

```bash
pip install -r requirements.txt
streamlit run Accueil.py
```

➡️ Le dashboard s'ouvre sur `http://localhost:8501`

## 📁 Structure

```
dashboard_trafic/
├── Accueil.py                       # Application Streamlit
├── prepare_data.py                  # Script ETL (à lancer une fois)
├── agg_par_arc_heure.csv.gz         # Agrégation horaire × jour (2.5 Mo)
├── agg_par_arc.csv.gz               # Stats par tronçon (0.1 Mo)
├── agg_temporel.csv.gz              # Tendances temporelles globales (0.01 Mo)
├── sample_diagramme.csv.gz          # Échantillon diagramme fondamental (0.16 Mo)
├── requirements.txt
└── README.md
```

## 🛠️ Stack technique

- **Streamlit** — framework de dashboard
- **Pandas / NumPy** — manipulation de données
- **Matplotlib / Seaborn** — visualisations statiques stylisées
- **Folium** — cartographie interactive

## ⚙️ Pipeline de traitement (prepare_data.py)

Le script `prepare_data.py` transforme le fichier CSV brut (~330 Mo) en
**4 fichiers compressés (~3 Mo)** prêts pour le dashboard :

```
CSV brut (330 Mo)
    │
    ├── 1) Lecture par chunks (gestion mémoire)
    ├── 2) Suppression de colonnes redondantes
    ├── 3) Imputation par interpolation linéaire temporelle
    ├── 4) Création de variables temporelles (heure, jour, weekend)
    │
    ├── 5) Agrégation par tronçon × heure × jour de semaine
    ├── 6) Agrégation par tronçon (stats + coordonnées)
    ├── 7) Agrégation temporelle globale
    └── 8) Échantillon brut pour le diagramme fondamental
        │
        ▼
4 fichiers .csv.gz (≈ 3 Mo total) → Compatible Streamlit Cloud
```

**Réduction : 119× plus léger** — c'est une bonne pratique professionnelle
pour mettre en production un dashboard analytique.

## 🎨 Design

Palette **Paris vintage** :
- 🌿 Vert métro `#2A5C4D` (couleur principale)
- ⭐ Doré Tour Eiffel `#D4A82A` (accent)
- 🪨 Beige pierre `#E8DDC8` (fond)
- 🔻 Rouge accent `#B33A3A` (alertes, données critiques)

## 📊 Source des données

**Direction de la Voirie et des Déplacements — Ville de Paris**
[opendata.paris.fr](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/)

---

⭐ Réalisé dans le cadre du Master Data Management — aivancity
