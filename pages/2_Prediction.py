"""
═══════════════════════════════════════════════════════════════════════════
🎯 PAGE PRÉDICTION — Prédiction client par client
═══════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# ─── CONFIGURATION ───
st.set_page_config(
    page_title="Prédiction | Marketing Bancaire",
    page_icon="🎯",
    layout="wide"
)


# ═══════════════════════════════════════════════════════════════════════════
# CHARGEMENT / ENTRAÎNEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_or_train_model():
    """
    Charge le modèle s'il existe, sinon l'entraîne et le sauvegarde.
    Retourne (model, feature_names).
    """
    model_path = 'rf_model.joblib'
    features_path = 'feature_names.joblib'

    # Si déjà entraîné, charger
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names

    # Sinon, entraîner
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('bankfull.csv', sep=';')

    # Encodages (cohérents avec le notebook)
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    for col in ['housing', 'loan', 'default']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df['education'] = df['education'].map({
        'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 1
    })

    # One-hot
    df = pd.get_dummies(df, columns=['job', 'marital', 'contact', 'month', 'poutcome'],
                        drop_first=False, dtype=int)

    # Suppression de duration (data leakage)
    df = df.drop(columns=['duration'])

    X = df.drop(columns=['y'])
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    feature_names = list(X.columns)

    # Sauvegarde
    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)

    return model, feature_names


# ─── CHARGEMENT ───
with st.spinner("⚙️ Chargement du modèle..."):
    model, feature_names = load_or_train_model()

# ─── EN-TÊTE ───
st.title("🎯 Prédiction de souscription")
st.markdown("Renseignez les caractéristiques d'un client pour prédire sa probabilité de souscription.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# FORMULAIRE DE SAISIE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 👤 Profil du client")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📋 Sociodémographique**")
    age = st.slider("Âge", 18, 95, 35)
    job = st.selectbox("Profession", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown'
    ])
    marital = st.selectbox("Statut marital", ['single', 'married', 'divorced'])
    education = st.selectbox("Éducation", ['primary', 'secondary', 'tertiary', 'unknown'])

with col2:
    st.markdown("**💰 Bancaire**")
    balance = st.number_input("Solde (€)", -10000, 100000, 1500, step=100)
    default = st.radio("Crédit en défaut ?", ['no', 'yes'], horizontal=True)
    housing = st.radio("Prêt immobilier ?", ['no', 'yes'], horizontal=True)
    loan = st.radio("Prêt personnel ?", ['no', 'yes'], horizontal=True)

with col3:
    st.markdown("**📞 Campagne**")
    contact = st.selectbox("Type de contact", ['cellular', 'telephone', 'unknown'])
    month = st.selectbox("Mois", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ], index=4)
    day = st.slider("Jour du mois", 1, 31, 15)
    campaign = st.number_input("Nombre de contacts dans cette campagne", 1, 50, 2)

# Variables historiques
st.markdown("**📜 Historique (campagnes précédentes)**")
col1, col2, col3 = st.columns(3)
with col1:
    pdays = st.number_input("Jours depuis le dernier contact", -1, 1000, -1,
                            help="-1 si jamais contacté")
with col2:
    previous = st.number_input("Nombre de contacts précédents", 0, 50, 0)
with col3:
    poutcome = st.selectbox("Résultat campagne précédente",
                             ['unknown', 'failure', 'other', 'success'])

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU VECTEUR DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════
def build_feature_vector(feature_names):
    """Construit le vecteur de features dans l'ordre attendu par le modèle."""
    # Mapping des valeurs
    mapping_education = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 1}

    # Initialiser un dictionnaire avec toutes les features à 0
    features = {name: 0 for name in feature_names}

    # Variables numériques et binaires directes
    features['age'] = age
    features['balance'] = balance
    features['day'] = day
    features['campaign'] = campaign
    features['pdays'] = pdays
    features['previous'] = previous
    features['default'] = 1 if default == 'yes' else 0
    features['housing'] = 1 if housing == 'yes' else 0
    features['loan'] = 1 if loan == 'yes' else 0
    features['education'] = mapping_education[education]

    # Variables one-hot encodées
    if f'job_{job}' in features:
        features[f'job_{job}'] = 1
    if f'marital_{marital}' in features:
        features[f'marital_{marital}'] = 1
    if f'contact_{contact}' in features:
        features[f'contact_{contact}'] = 1
    if f'month_{month}' in features:
        features[f'month_{month}'] = 1
    if f'poutcome_{poutcome}' in features:
        features[f'poutcome_{poutcome}'] = 1

    return pd.DataFrame([features])[feature_names]


# ═══════════════════════════════════════════════════════════════════════════
# PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════
if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
    X_client = build_feature_vector(feature_names)

    # Prédiction
    prediction = model.predict(X_client)[0]
    proba_yes = model.predict_proba(X_client)[0, 1]
    proba_no = 1 - proba_yes

    st.markdown("---")
    st.markdown("## 🎯 Résultat de la prédiction")

    # ─── RÉSULTAT PRINCIPAL ───
    col1, col2 = st.columns([1, 2])

    with col1:
        if prediction == 1:
            st.success(f"### ✅ Client à CIBLER")
            st.markdown(f"#### Probabilité de souscription : **{proba_yes*100:.1f}%**")
        else:
            st.error(f"### ❌ Client peu prometteur")
            st.markdown(f"#### Probabilité de souscription : **{proba_yes*100:.1f}%**")

        # Niveau de confiance
        if proba_yes > 0.7:
            st.info("🔥 **Très forte propension** — appel prioritaire")
        elif proba_yes > 0.5:
            st.info("⭐ **Bonne propension** — à recontacter")
        elif proba_yes > 0.3:
            st.warning("🤔 **Propension modérée** — à évaluer")
        else:
            st.warning("📉 **Faible propension** — déprioriser")

    with col2:
        # Jauge Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba_yes * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score de propension (%)"},
            delta={'reference': 11.7, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1F4E79"},
                'steps': [
                    {'range': [0, 30], 'color': "#FADBD8"},
                    {'range': [30, 50], 'color': "#FCF3CF"},
                    {'range': [50, 70], 'color': "#D5F5E3"},
                    {'range': [70, 100], 'color': "#52BE80"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ─── DÉTAIL DES PROBABILITÉS ───
    st.markdown("### 📊 Probabilités détaillées")

    proba_df = pd.DataFrame({
        'Classe': ['❌ Ne souscrit pas', '✅ Souscrit'],
        'Probabilité': [proba_no, proba_yes]
    })

    fig = go.Figure(go.Bar(
        x=proba_df['Probabilité'] * 100,
        y=proba_df['Classe'],
        orientation='h',
        marker=dict(color=['#E74C3C', '#27AE60']),
        text=[f"{p*100:.1f}%" for p in proba_df['Probabilité']],
        textposition='inside'
    ))
    fig.update_layout(
        height=200,
        xaxis_title="Probabilité (%)",
        margin=dict(l=10, r=10, t=10, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── TOP DES VARIABLES INFLUENTES ───
    st.markdown("### 🏆 Top des variables qui ont influencé la prédiction")

    importances = pd.DataFrame({
        'Variable': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=importances['Importance'],
        y=importances['Variable'],
        orientation='h',
        marker=dict(color='#27AE60')
    ))
    fig.update_layout(
        title="Top 10 — Importance globale des variables (Random Forest)",
        xaxis_title="Importance",
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **Note** : ces importances sont calculées sur **l'ensemble du modèle**.
    Pour une explication individuelle (variable par variable pour CE client),
    on utiliserait des techniques avancées comme **SHAP**.
    """)

else:
    st.info("👆 Renseignez les informations du client puis cliquez sur **Lancer la prédiction**.")

# ─── FOOTER ───
st.markdown("---")
st.caption("🤖 Modèle utilisé : Random Forest (100 arbres, max_depth=10, class_weight='balanced')")
