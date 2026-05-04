"""
═══════════════════════════════════════════════════════════════════════════
📊 PAGE EXPLORATION — Analyse interactive des données
═══════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── CONFIGURATION ───
st.set_page_config(
    page_title="Exploration | Marketing Bancaire",
    page_icon="📊",
    layout="wide"
)


# ─── CHARGEMENT ───
@st.cache_data
def load_data():
    df = pd.read_csv('bankfull.csv', sep=';')
    return df


df = load_data()

# ─── EN-TÊTE ───
st.title("📊 Exploration interactive des données")
st.markdown("Filtrez et analysez les données pour découvrir les segments les plus prometteurs.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTRES
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.header("🎛️ Filtres")
st.sidebar.markdown("Affinez votre analyse :")

# Filtre âge
age_min, age_max = st.sidebar.slider(
    "📅 Tranche d'âge",
    int(df['age'].min()), int(df['age'].max()),
    (int(df['age'].min()), int(df['age'].max()))
)

# Filtre profession
jobs = ['Tous'] + sorted(df['job'].unique().tolist())
selected_job = st.sidebar.selectbox("💼 Profession", jobs)

# Filtre statut marital
maritals = ['Tous'] + sorted(df['marital'].unique().tolist())
selected_marital = st.sidebar.selectbox("💍 Statut marital", maritals)

# Filtre éducation
educations = ['Tous'] + sorted(df['education'].unique().tolist())
selected_education = st.sidebar.selectbox("🎓 Éducation", educations)

# Filtre crédit immo
housing_filter = st.sidebar.radio("🏠 Prêt immobilier ?", ['Tous', 'yes', 'no'])

# ─── APPLICATION DES FILTRES ───
df_filtered = df.copy()
df_filtered = df_filtered[
    (df_filtered['age'] >= age_min) & (df_filtered['age'] <= age_max)
]
if selected_job != 'Tous':
    df_filtered = df_filtered[df_filtered['job'] == selected_job]
if selected_marital != 'Tous':
    df_filtered = df_filtered[df_filtered['marital'] == selected_marital]
if selected_education != 'Tous':
    df_filtered = df_filtered[df_filtered['education'] == selected_education]
if housing_filter != 'Tous':
    df_filtered = df_filtered[df_filtered['housing'] == housing_filter]

# ─── KPIs DYNAMIQUES ───
st.markdown("### 📈 Résultats des filtres")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Clients filtrés",
        f"{len(df_filtered):,}",
        delta=f"{len(df_filtered)/len(df)*100:.1f}% du total"
    )

with col2:
    nb_yes_filtered = (df_filtered['y'] == 'yes').sum()
    st.metric(
        "✅ Souscripteurs",
        f"{nb_yes_filtered:,}"
    )

with col3:
    if len(df_filtered) > 0:
        taux = nb_yes_filtered / len(df_filtered) * 100
    else:
        taux = 0
    delta_taux = taux - 11.7
    st.metric(
        "🎯 Taux de conversion",
        f"{taux:.1f}%",
        delta=f"{delta_taux:+.1f} pts vs moyenne"
    )

with col4:
    if len(df_filtered) > 0:
        balance_moyen = df_filtered['balance'].mean()
    else:
        balance_moyen = 0
    st.metric(
        "💰 Solde moyen",
        f"{balance_moyen:,.0f} €"
    )

if len(df_filtered) == 0:
    st.warning("⚠️ Aucun client ne correspond à ces filtres. Modifiez vos critères.")
    st.stop()

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# ONGLETS D'ANALYSE
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distributions",
    "🎯 Taux de conversion",
    "🔗 Corrélations",
    "📋 Données brutes"
])

# ─── TAB 1 : DISTRIBUTIONS ───
with tab1:
    st.markdown("### 📊 Distribution des variables")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution âge
        fig = px.histogram(
            df_filtered, x='age', color='y', nbins=30,
            color_discrete_map={'yes': '#27AE60', 'no': '#E74C3C'},
            title="Distribution de l'âge par souscription",
            labels={'age': 'Âge', 'count': 'Effectif'}
        )
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution balance
        fig = px.box(
            df_filtered, x='y', y='balance', color='y',
            color_discrete_map={'yes': '#27AE60', 'no': '#E74C3C'},
            title="Solde bancaire par souscription",
            labels={'balance': 'Solde (€)', 'y': 'Souscription'}
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution par profession
        job_counts = df_filtered['job'].value_counts().reset_index()
        job_counts.columns = ['Profession', 'Effectif']
        fig = px.bar(
            job_counts, x='Effectif', y='Profession', orientation='h',
            color='Effectif', color_continuous_scale='Blues',
            title="Répartition par profession"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution par mois
        ordre_mois = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month_counts = df_filtered['month'].value_counts().reindex(ordre_mois).fillna(0).reset_index()
        month_counts.columns = ['Mois', 'Effectif']
        fig = px.bar(
            month_counts, x='Mois', y='Effectif',
            color='Effectif', color_continuous_scale='Greens',
            title="Répartition par mois de contact"
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2 : TAUX DE CONVERSION ───
with tab2:
    st.markdown("### 🎯 Taux de conversion par segment")
    st.caption("Identifiez les segments les plus prometteurs (au-dessus de la moyenne 11.7%)")

    # Sélecteur de variable
    variable = st.selectbox(
        "Choisissez une variable à analyser :",
        ['job', 'marital', 'education', 'month', 'contact', 'poutcome',
         'housing', 'loan', 'default']
    )

    # Calcul des taux
    taux_par_modalite = df_filtered.groupby(variable)['y'].apply(
        lambda x: (x == 'yes').sum() / len(x) * 100
    ).sort_values(ascending=True).reset_index()
    taux_par_modalite.columns = [variable, 'Taux (%)']

    # Effectifs
    effectifs = df_filtered[variable].value_counts().reset_index()
    effectifs.columns = [variable, 'Effectif']
    taux_par_modalite = taux_par_modalite.merge(effectifs, on=variable)

    # Couleurs : vert si > moyenne, rouge sinon
    moyenne = (df_filtered['y'] == 'yes').sum() / len(df_filtered) * 100
    couleurs = ['#27AE60' if x > moyenne else '#E74C3C'
                for x in taux_par_modalite['Taux (%)']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=taux_par_modalite['Taux (%)'],
        y=taux_par_modalite[variable],
        orientation='h',
        marker=dict(color=couleurs),
        text=[f"{t:.1f}% (n={n})" for t, n in zip(
            taux_par_modalite['Taux (%)'], taux_par_modalite['Effectif']
        )],
        textposition='outside'
    ))
    fig.add_vline(x=moyenne, line_dash="dash", line_color="blue",
                  annotation_text=f"Moyenne ({moyenne:.1f}%)")
    fig.update_layout(
        title=f"Taux de souscription par '{variable}'",
        xaxis_title="Taux de conversion (%)",
        yaxis_title=variable,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau détaillé
    st.markdown("#### 📋 Détails par modalité")
    taux_display = taux_par_modalite.sort_values('Taux (%)', ascending=False)
    taux_display['Taux (%)'] = taux_display['Taux (%)'].round(2)
    st.dataframe(taux_display, use_container_width=True, hide_index=True)

# ─── TAB 3 : CORRÉLATIONS ───
with tab3:
    st.markdown("### 🔗 Matrice de corrélation des variables numériques")

    variables_num = ['age', 'balance', 'day', 'duration',
                     'campaign', 'pdays', 'previous']
    corr_matrix = df_filtered[variables_num].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Matrice de corrélation"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Lecture :**
    - 🔴 Rouge foncé = corrélation positive forte (proche de +1)
    - ⚪ Blanc = pas de corrélation (proche de 0)
    - 🔵 Bleu foncé = corrélation négative forte (proche de -1)
    """)

    # Scatter plot interactif
    st.markdown("#### 🔍 Explorer une relation spécifique")

    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("Variable X :", variables_num, index=0)
    with col2:
        var_y = st.selectbox("Variable Y :", variables_num, index=1)

    fig = px.scatter(
        df_filtered.sample(min(2000, len(df_filtered))),
        x=var_x, y=var_y, color='y',
        color_discrete_map={'yes': '#27AE60', 'no': '#E74C3C'},
        opacity=0.5,
        title=f"Relation entre '{var_x}' et '{var_y}'"
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── TAB 4 : DONNÉES BRUTES ───
with tab4:
    st.markdown("### 📋 Aperçu des données filtrées")

    col1, col2 = st.columns([1, 3])

    with col1:
        nb_lignes = st.number_input("Nombre de lignes à afficher :",
                                     min_value=5, max_value=500, value=20)

    with col2:
        st.info(f"📊 **{len(df_filtered):,} clients** correspondent à vos filtres")

    st.dataframe(df_filtered.head(nb_lignes), use_container_width=True, hide_index=True)

    # Téléchargement CSV
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv,
        file_name='donnees_filtrees.csv',
        mime='text/csv'
    )

# ─── FOOTER ───
st.markdown("---")
st.caption("💡 Astuce : utilisez les filtres dans la barre latérale pour explorer des segments spécifiques.")
