"""
═══════════════════════════════════════════════════════════════════════════
💰 PAGE SCÉNARIOS — Simulation marketing & calcul du ROI
═══════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ─── CONFIGURATION ───
st.set_page_config(
    page_title="Scénarios | Marketing Bancaire",
    page_icon="💰",
    layout="wide"
)


# ═══════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES + MODÈLE + PROBAS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data_with_predictions():
    """Charge les données et calcule les probabilités pour chaque client."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('bankfull.csv', sep=';')
    df_brut = df.copy()  # Garder une copie pour affichage

    # Préparation
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    for col in ['housing', 'loan', 'default']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df['education'] = df['education'].map({
        'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 1
    })

    df = pd.get_dummies(df, columns=['job', 'marital', 'contact', 'month', 'poutcome'],
                        drop_first=False, dtype=int)
    df = df.drop(columns=['duration'])

    X = df.drop(columns=['y'])
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Charger ou entraîner le modèle
    model_path = 'rf_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)

    # Probabilités sur le set de test
    probas = model.predict_proba(X_test)[:, 1]

    # Construire un DataFrame avec scores + vraie cible
    df_scores = pd.DataFrame({
        'index': X_test.index,
        'proba': probas,
        'reel': y_test.values
    }).sort_values('proba', ascending=False).reset_index(drop=True)

    return df_scores, df_brut


# ─── CHARGEMENT ───
with st.spinner("⚙️ Chargement des prédictions..."):
    df_scores, df_brut = load_data_with_predictions()

NB_TOTAL = len(df_scores)
NB_VRAIS_YES = df_scores['reel'].sum()
TAUX_BASE = NB_VRAIS_YES / NB_TOTAL * 100

# ─── EN-TÊTE ───
st.title("💰 Simulation de scénarios marketing")
st.markdown("Comparez différentes stratégies de ciblage et calculez leur ROI en temps réel.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# PARAMÈTRES BUSINESS (sidebar)
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Paramètres business")
st.sidebar.markdown("Ajustez les paramètres économiques :")

cout_appel = st.sidebar.number_input(
    "💸 Coût d'un appel (€)",
    min_value=1.0, max_value=50.0, value=5.0, step=0.5
)

gain_souscription = st.sidebar.number_input(
    "💰 Gain par souscription (€)",
    min_value=50.0, max_value=2000.0, value=200.0, step=10.0
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
📊 **Données du test set :**
- {NB_TOTAL:,} clients à évaluer
- {NB_VRAIS_YES:,} vrais souscripteurs
- Taux de base : {TAUX_BASE:.1f}%
""")


# ═══════════════════════════════════════════════════════════════════════════
# FONCTION DE CALCUL D'UN SCÉNARIO
# ═══════════════════════════════════════════════════════════════════════════
def calcul_scenario(df_scores, pct_cible, cout_appel, gain_souscription):
    """
    Calcule les indicateurs d'un scénario où on cible les top X% du score.
    """
    nb_appels = int(len(df_scores) * pct_cible / 100)
    if nb_appels == 0:
        return None

    top_clients = df_scores.head(nb_appels)
    nb_souscriptions = top_clients['reel'].sum()

    cout_total = nb_appels * cout_appel
    gain_total = nb_souscriptions * gain_souscription
    benefice = gain_total - cout_total
    roi = (benefice / cout_total * 100) if cout_total > 0 else 0
    taux_conversion = (nb_souscriptions / nb_appels * 100) if nb_appels > 0 else 0

    # % des vrais yes captés
    nb_total_yes = df_scores['reel'].sum()
    pct_yes_captes = (nb_souscriptions / nb_total_yes * 100) if nb_total_yes > 0 else 0

    return {
        'nb_appels': nb_appels,
        'nb_souscriptions': int(nb_souscriptions),
        'cout_total': cout_total,
        'gain_total': gain_total,
        'benefice': benefice,
        'roi': roi,
        'taux_conversion': taux_conversion,
        'pct_yes_captes': pct_yes_captes
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3 SCÉNARIOS PRÉDÉFINIS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Comparaison des 3 scénarios types")

col1, col2, col3 = st.columns(3)

# Scénario 1 : Conservateur (top 10%)
sc_cons = calcul_scenario(df_scores, 10, cout_appel, gain_souscription)
with col1:
    st.markdown("#### 🛡️ Conservateur")
    st.caption("Cibler le top 10% du score")
    st.metric("📞 Appels", f"{sc_cons['nb_appels']:,}")
    st.metric("✅ Souscriptions", f"{sc_cons['nb_souscriptions']:,}")
    st.metric("🎯 Conversion", f"{sc_cons['taux_conversion']:.1f}%",
              f"+{sc_cons['taux_conversion'] - TAUX_BASE:.1f} pts")
    st.metric("💰 ROI", f"{sc_cons['roi']:.0f}%")
    st.metric("🎁 Bénéfice", f"{sc_cons['benefice']:,.0f} €")

# Scénario 2 : Équilibré (top 30%)
sc_eq = calcul_scenario(df_scores, 30, cout_appel, gain_souscription)
with col2:
    st.markdown("#### ⚖️ Équilibré")
    st.caption("Cibler le top 30% du score")
    st.metric("📞 Appels", f"{sc_eq['nb_appels']:,}")
    st.metric("✅ Souscriptions", f"{sc_eq['nb_souscriptions']:,}")
    st.metric("🎯 Conversion", f"{sc_eq['taux_conversion']:.1f}%",
              f"+{sc_eq['taux_conversion'] - TAUX_BASE:.1f} pts")
    st.metric("💰 ROI", f"{sc_eq['roi']:.0f}%")
    st.metric("🎁 Bénéfice", f"{sc_eq['benefice']:,.0f} €")

# Scénario 3 : Agressif (top 60%)
sc_agg = calcul_scenario(df_scores, 60, cout_appel, gain_souscription)
with col3:
    st.markdown("#### 🚀 Agressif")
    st.caption("Cibler le top 60% du score")
    st.metric("📞 Appels", f"{sc_agg['nb_appels']:,}")
    st.metric("✅ Souscriptions", f"{sc_agg['nb_souscriptions']:,}")
    st.metric("🎯 Conversion", f"{sc_agg['taux_conversion']:.1f}%",
              f"+{sc_agg['taux_conversion'] - TAUX_BASE:.1f} pts")
    st.metric("💰 ROI", f"{sc_agg['roi']:.0f}%")
    st.metric("🎁 Bénéfice", f"{sc_agg['benefice']:,.0f} €")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# COMPARAISON AVEC SCÉNARIO BASELINE (sans modèle)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 🆚 Avec ou sans modèle ?")

# Sans modèle = on appelle tous les clients
sc_baseline = {
    'nb_appels': NB_TOTAL,
    'nb_souscriptions': NB_VRAIS_YES,
    'cout_total': NB_TOTAL * cout_appel,
    'gain_total': NB_VRAIS_YES * gain_souscription,
}
sc_baseline['benefice'] = sc_baseline['gain_total'] - sc_baseline['cout_total']
sc_baseline['roi'] = sc_baseline['benefice'] / sc_baseline['cout_total'] * 100
sc_baseline['taux_conversion'] = TAUX_BASE

# Tableau comparatif
df_comparaison = pd.DataFrame({
    'Scénario': ['❌ Sans modèle (baseline)', '🛡️ Conservateur', '⚖️ Équilibré', '🚀 Agressif'],
    'Nb appels': [sc_baseline['nb_appels'], sc_cons['nb_appels'], sc_eq['nb_appels'], sc_agg['nb_appels']],
    'Nb souscriptions': [sc_baseline['nb_souscriptions'], sc_cons['nb_souscriptions'],
                         sc_eq['nb_souscriptions'], sc_agg['nb_souscriptions']],
    'Taux conversion (%)': [round(sc_baseline['taux_conversion'], 1),
                             round(sc_cons['taux_conversion'], 1),
                             round(sc_eq['taux_conversion'], 1),
                             round(sc_agg['taux_conversion'], 1)],
    'Bénéfice (€)': [int(sc_baseline['benefice']), int(sc_cons['benefice']),
                     int(sc_eq['benefice']), int(sc_agg['benefice'])],
    'ROI (%)': [round(sc_baseline['roi']), round(sc_cons['roi']),
                round(sc_eq['roi']), round(sc_agg['roi'])]
})

st.dataframe(
    df_comparaison.style.highlight_max(subset=['ROI (%)', 'Bénéfice (€)', 'Taux conversion (%)'],
                                       color='#D4EDDA'),
    use_container_width=True, hide_index=True
)

# Visualisation
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure(go.Bar(
        x=df_comparaison['Scénario'],
        y=df_comparaison['ROI (%)'],
        marker=dict(color=['#95A5A6', '#3498DB', '#27AE60', '#E67E22']),
        text=[f"{v}%" for v in df_comparaison['ROI (%)']],
        textposition='outside'
    ))
    fig.update_layout(
        title="ROI par scénario (%)",
        yaxis_title="ROI (%)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(go.Bar(
        x=df_comparaison['Scénario'],
        y=df_comparaison['Bénéfice (€)'],
        marker=dict(color=['#95A5A6', '#3498DB', '#27AE60', '#E67E22']),
        text=[f"{v:,} €" for v in df_comparaison['Bénéfice (€)']],
        textposition='outside'
    ))
    fig.update_layout(
        title="Bénéfice par scénario (€)",
        yaxis_title="Bénéfice (€)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATEUR PERSONNALISÉ
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 🎚️ Simulateur personnalisé")
st.markdown("Ajustez le pourcentage de la base à cibler et voyez l'impact en temps réel :")

pct_personnalise = st.slider(
    "🎯 Pourcentage de la base à cibler (top X% du score)",
    min_value=1, max_value=100, value=30, step=1,
    help="Ex: 30% = on appelle les 30% des clients avec le score le plus élevé"
)

sc_perso = calcul_scenario(df_scores, pct_personnalise, cout_appel, gain_souscription)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📞 Appels", f"{sc_perso['nb_appels']:,}")
with col2:
    st.metric("✅ Souscriptions", f"{sc_perso['nb_souscriptions']:,}",
              delta=f"{sc_perso['pct_yes_captes']:.0f}% des vrais yes")
with col3:
    st.metric("🎯 Conversion", f"{sc_perso['taux_conversion']:.1f}%",
              delta=f"+{sc_perso['taux_conversion'] - TAUX_BASE:.1f} pts vs baseline")
with col4:
    st.metric("💰 ROI", f"{sc_perso['roi']:.0f}%",
              delta=f"+{sc_perso['roi'] - sc_baseline['roi']:.0f} pts vs baseline")

# Détails financiers
st.markdown("#### 💵 Détails financiers")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"💸 **Coût total** : {sc_perso['cout_total']:,.0f} €")
with col2:
    st.info(f"💰 **Gain total** : {sc_perso['gain_total']:,.0f} €")
with col3:
    if sc_perso['benefice'] > 0:
        st.success(f"🎁 **Bénéfice net** : {sc_perso['benefice']:,.0f} €")
    else:
        st.error(f"🎁 **Bénéfice net** : {sc_perso['benefice']:,.0f} €")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# COURBE D'OPTIMISATION : ROI VS % CIBLÉ
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 📈 Courbe d'optimisation : trouver le sweet spot")

# Calculer le ROI pour chaque % de 1 à 100
pct_range = list(range(1, 101))
rois = []
benefs = []
conversions = []

for p in pct_range:
    sc = calcul_scenario(df_scores, p, cout_appel, gain_souscription)
    rois.append(sc['roi'])
    benefs.append(sc['benefice'])
    conversions.append(sc['taux_conversion'])

# Trouver le % optimal
idx_max_roi = rois.index(max(rois))
pct_optimal_roi = pct_range[idx_max_roi]
idx_max_benef = benefs.index(max(benefs))
pct_optimal_benef = pct_range[idx_max_benef]

fig = go.Figure()

# ROI
fig.add_trace(go.Scatter(
    x=pct_range, y=rois, mode='lines',
    name='ROI (%)', line=dict(color='#E67E22', width=3)
))

# Marqueurs optimaux
fig.add_trace(go.Scatter(
    x=[pct_optimal_roi], y=[max(rois)],
    mode='markers', name=f'ROI max ({pct_optimal_roi}%)',
    marker=dict(size=15, color='red', symbol='star')
))

fig.update_layout(
    title="Évolution du ROI en fonction du % de la base ciblée",
    xaxis_title="% de la base ciblée",
    yaxis_title="ROI (%)",
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.success(f"""
    🏆 **ROI maximal** : {max(rois):.0f}%
    📊 Atteint en ciblant **{pct_optimal_roi}%** de la base
    """)
with col2:
    st.success(f"""
    💰 **Bénéfice maximal** : {max(benefs):,.0f} €
    📊 Atteint en ciblant **{pct_optimal_benef}%** de la base
    """)

st.info("""
💡 **Comprendre la courbe** : plus on cible un petit % (top du score),
plus le ROI est élevé (clients de qualité). Mais en volume absolu,
le bénéfice augmente jusqu'à un certain seuil avant de chuter
(on appelle des clients moins prometteurs).
""")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# RECOMMANDATIONS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 🎯 Recommandation stratégique")

if pct_optimal_benef <= 15:
    reco = "🛡️ **Stratégie conservatrice recommandée**"
    detail = "Concentrez-vous sur le top du score pour maximiser le ROI."
elif pct_optimal_benef <= 40:
    reco = "⚖️ **Stratégie équilibrée recommandée**"
    detail = "Bon compromis entre volume d'affaires et rentabilité."
else:
    reco = "🚀 **Stratégie volume recommandée**"
    detail = "Le marché tolère un ciblage large pour maximiser le bénéfice absolu."

st.success(f"""
### {reco}

{detail}

**📋 Plan d'action concret :**
1. 🎯 **Cibler les {pct_optimal_benef}%** de clients avec le score le plus élevé
2. 📞 **{calcul_scenario(df_scores, pct_optimal_benef, cout_appel, gain_souscription)['nb_appels']:,} appels** à passer (vs {NB_TOTAL:,} sans modèle)
3. 💰 **Bénéfice attendu : {max(benefs):,.0f} €** (vs {sc_baseline['benefice']:,.0f} € sans modèle)
4. 🚀 **Économie d'appels : {NB_TOTAL - calcul_scenario(df_scores, pct_optimal_benef, cout_appel, gain_souscription)['nb_appels']:,}** sans perte significative de souscriptions
""")

# ─── FOOTER ───
st.markdown("---")
st.caption("💡 Tous les calculs sont basés sur le modèle Random Forest et le set de test (20% des données).")
