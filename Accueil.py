"""
═══════════════════════════════════════════════════════════════════════════
🏦 DASHBOARD MARKETING BANCAIRE — Page d'accueil
═══════════════════════════════════════════════════════════════════════════
Application Streamlit pour analyser les campagnes marketing bancaires
et prédire les souscriptions à un dépôt à terme.

Auteur : [Votre nom]
Date   : 2025
═══════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── CONFIGURATION DE LA PAGE ───
st.set_page_config(
    page_title="Marketing Bancaire | Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLE CSS PERSONNALISÉ ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #595959;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .metric-card {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border-left: 5px solid #1F4E79;
    }
    .stMetric {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES (avec cache pour performance)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """Charge le dataset bancaire."""
    df = pd.read_csv('bankfull.csv', sep=';')
    return df


# ═══════════════════════════════════════════════════════════════════════════
# CONTENU DE LA PAGE D'ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════

# ─── EN-TÊTE ───
st.markdown('<p class="main-header">🏦 Marketing Bancaire — Dashboard d\'analyse</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prédiction de souscription à un dépôt à terme</p>',
            unsafe_allow_html=True)

st.markdown("---")

# ─── CHARGEMENT ───
df = load_data()

# ─── INTRODUCTION ───
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🎯 Contexte du projet

    Une **banque portugaise** mène des campagnes de marketing direct par téléphone
    pour proposer à ses clients de souscrire à un **dépôt à terme**.

    **Le problème :** le taux de conversion réel n'est que de **11.7%** —
    autrement dit, **88 appels sur 100 sont infructueux**, ce qui plombe le ROI
    des campagnes.

    **L'objectif :** construire un modèle prédictif capable d'identifier
    *à l'avance* les clients à fort potentiel de souscription, afin de :
    - 🎯 Optimiser le ciblage des campagnes
    - 💰 Augmenter le retour sur investissement
    - 📞 Réduire les sollicitations inutiles
    - ✨ Améliorer l'expérience client
    """)

with col2:
    st.markdown("### 📊 Données utilisées")
    st.info(f"""
    **Dataset :** Banque Portugaise (UCI)

    📈 **{df.shape[0]:,} clients**
    📋 **{df.shape[1]} variables**
    ✅ **Aucune valeur manquante**
    🎯 **Cible :** Souscription (yes/no)
    """)

st.markdown("---")

# ─── KPIs CLÉS ───
st.markdown("### 📈 KPIs clés du projet")

col1, col2, col3, col4 = st.columns(4)

with col1:
    nb_clients = len(df)
    st.metric(
        label="👥 Total clients",
        value=f"{nb_clients:,}"
    )

with col2:
    nb_yes = (df['y'] == 'yes').sum()
    st.metric(
        label="✅ Souscripteurs",
        value=f"{nb_yes:,}",
        delta=f"{nb_yes/nb_clients*100:.1f}% du total"
    )

with col3:
    nb_no = (df['y'] == 'no').sum()
    st.metric(
        label="❌ Non-souscripteurs",
        value=f"{nb_no:,}",
        delta=f"{nb_no/nb_clients*100:.1f}% du total",
        delta_color="inverse"
    )

with col4:
    age_moyen = df['age'].mean()
    st.metric(
        label="📅 Âge moyen",
        value=f"{age_moyen:.0f} ans"
    )

st.markdown("---")

# ─── VISUALISATION : DÉSÉQUILIBRE DE LA CIBLE ───
st.markdown("### 🎯 Le défi : un fort déséquilibre des classes")

col1, col2 = st.columns([1, 1])

with col1:
    # Donut chart
    fig_donut = go.Figure(data=[go.Pie(
        labels=['Non-souscripteurs', 'Souscripteurs'],
        values=[nb_no, nb_yes],
        hole=0.5,
        marker=dict(colors=['#E74C3C', '#27AE60']),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    fig_donut.update_layout(
        title="Répartition de la variable cible",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.markdown("""
    #### 💡 Pourquoi est-ce un défi ?

    Avec **88.3% de "no"** contre seulement **11.7% de "yes"**, un modèle naïf
    qui prédirait toujours "no" obtiendrait déjà **88% d'accuracy**...
    sans rien apprendre d'utile.

    **Conséquences techniques :**
    - L'**Accuracy seule devient trompeuse**
    - Il faut privilégier le **F1-score** et le **Recall**
    - Utilisation de `class_weight='balanced'` lors de l'entraînement

    **Conséquences business :**
    - Rater un souscripteur = revenu perdu (coût élevé)
    - Appeler un non-souscripteur = coût d'un appel (faible)
    - → On privilégie le **Recall** dans nos modèles
    """)

st.markdown("---")

# ─── NAVIGATION VERS LES PAGES ───
st.markdown("### 🧭 Navigation dans le dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📊 Exploration
    Analysez les données de manière interactive :
    - Filtres dynamiques (âge, profession…)
    - Distribution des variables
    - Taux de conversion par segment
    - Matrice de corrélation
    """)
    st.info("👈 Page **Exploration** dans le menu")

with col2:
    st.markdown("""
    #### 🎯 Prédiction
    Testez la prédiction sur un client :
    - Formulaire interactif
    - Score de propension en temps réel
    - Recommandation de ciblage
    - Top des facteurs influents
    """)
    st.info("👈 Page **Prédiction** dans le menu")

with col3:
    st.markdown("""
    #### 💰 Scénarios
    Simulez vos campagnes marketing :
    - 3 scénarios (conservateur/équilibré/agressif)
    - Calcul du ROI en temps réel
    - Comparaison avant/après modèle
    - Recommandations stratégiques
    """)
    st.info("👈 Page **Scénarios** dans le menu")

st.markdown("---")

# ─── MÉTHODOLOGIE ───
with st.expander("📚 Méthodologie utilisée (CRISP-DM)"):
    st.markdown("""
    Ce projet suit la méthodologie **CRISP-DM** (Cross-Industry Standard Process for Data Mining) :

    | Étape | Action |
    |---|---|
    | **1. Compréhension métier** | Analyse des objectifs et du contexte bancaire |
    | **2. Compréhension des données** | EDA, statistiques descriptives, visualisations |
    | **3. Préparation** | Encodage, gestion du déséquilibre, split train/test |
    | **4. Modélisation** | 4 modèles testés : Logistic Regression, KNN, Decision Tree, Random Forest |
    | **5. Évaluation** | Accuracy, Precision, Recall, F1, ROC-AUC |
    | **6. Déploiement** | Ce dashboard 🚀 |
    """)

with st.expander("🏆 Performances des modèles testés"):
    perf_df = pd.DataFrame({
        'Modèle': ['Régression Logistique', 'KNN (K=5)', 'Decision Tree', 'Random Forest'],
        'Accuracy': [76.36, 88.51, 79.68, 82.86],
        'Precision': [27.73, 51.87, 29.53, 35.63],
        'Recall': [63.56, 25.12, 53.19, 57.66],
        'F1-score': [38.61, 33.85, 37.98, 44.04],
        'ROC-AUC': [77.36, 70.77, 73.27, 79.21]
    })
    st.dataframe(perf_df.style.highlight_max(axis=0, color='#D4EDDA'),
                 use_container_width=True, hide_index=True)
    st.success("🏆 **Random Forest** est retenu : meilleur F1-score (44%) et meilleur ROC-AUC (79.2%)")


# ─── FOOTER ───
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "Dashboard réalisé avec Streamlit + Plotly | Projet Marketing Bancaire 🏦"
    "</p>",
    unsafe_allow_html=True
)
