"""
═══════════════════════════════════════════════════════════════════════════
🚦 DASHBOARD TRAFIC PARISIEN — Application Streamlit
═══════════════════════════════════════════════════════════════════════════
Analyse interactive des données de trafic routier de la Ville de Paris
(capteurs permanents, 1M+ mesures, 13 mois de données).

Architecture single-page avec onglets, chargement de fichiers pré-agrégés.

Auteur : Cynthia Djakpa
═══════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ─── CONSTANTES ───
DATA_AGG_HORAIRE = "agg_par_arc_heure.csv.gz"
DATA_AGG_TRONCONS = "agg_par_arc.csv.gz"
DATA_AGG_TEMPOREL = "agg_temporel.csv.gz"
DATA_SAMPLE_DIAG = "sample_diagramme.csv.gz"

NB_LIGNES_TOTAL = 1_048_575      # connue à l'avance
NB_TRONCONS = 2985
PERIODE = "Sept 2023 → Oct 2024 (13 mois)"


# ═══════════════════════════════════════════════════════════════════════════
# STYLE MATPLOTLIB — PALETTE PARIS VINTAGE
# ═══════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid")
for k, v in {
    "axes.facecolor": "#FAF7F0", "figure.facecolor": "#FAF7F0",
    "axes.edgecolor": "#D4C5A9", "grid.color": "#E8DDC8",
    "axes.labelcolor": "#2A5C4D", "xtick.color": "#3A3A3A",
    "ytick.color": "#3A3A3A", "text.color": "#3A3A3A",
    "axes.titlecolor": "#2A5C4D",
}.items():
    plt.rcParams[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Trafic Parisien — Dashboard",
    page_icon="🚦",
    layout="wide",
)

# ─── STYLE CSS — PALETTE PARIS VINTAGE ───
# Vert métro #2A5C4D · Doré Tour Eiffel #D4A82A · Beige pierre #E8DDC8
# Gris foncé #3A3A3A · Rouge accent #B33A3A · Crème #FAF7F0
st.markdown("""
<style>
.main { background: linear-gradient(135deg,#FAF7F0 0%,#F5EFE0 50%,#EFE4D0 100%); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#2A5C4D 0%,#1F4438 100%) !important;
    border-right: 1px solid #1A3A30;
}
section[data-testid="stSidebar"] * {
    color: #F5EFE0 !important;
}
section[data-testid="stSidebar"] a {
    color: #D4A82A !important;
    text-decoration: underline;
}

h1, h2, h3 { color:#2A5C4D !important; font-family: 'Georgia', serif; }

.metric-card {
    background: linear-gradient(135deg,#FFFFFF 0%,#FAF7F0 100%);
    border-radius: 4px;
    padding: 1rem;
    border: 1px solid #D4C5A9;
    border-left: 4px solid #D4A82A;
    box-shadow: 0 2px 6px rgba(58,58,58,.10);
    min-height: 88px;
    color: #3A3A3A;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D4C5A9;
    border-left: 4px solid #2A5C4D;
    padding: 0.8rem;
    border-radius: 4px;
    box-shadow: 0 2px 6px rgba(58,58,58,.08);
}
div[data-testid="stMetric"] label {
    color: #2A5C4D !important;
    font-weight: 600;
}

div[data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #D4A82A; }
button[data-baseweb="tab"] {
    background-color: #F5EFE0 !important;
    border-radius: 4px 4px 0 0 !important;
    color: #2A5C4D !important;
    border: 1px solid #D4C5A9 !important;
    border-bottom: none !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 500 !important;
    font-family: 'Georgia', serif !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #2A5C4D !important;
    color: #FAF7F0 !important;
    border-color: #2A5C4D !important;
    box-shadow: 0 -2px 4px rgba(212,168,42,.3);
}

.stButton > button {
    background: linear-gradient(135deg,#2A5C4D 0%,#1F4438 100%);
    color: #FAF7F0;
    border: 1px solid #D4A82A;
    border-radius: 4px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    width: 100%;
    font-family: 'Georgia', serif;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#D4A82A 0%,#B8901E 100%);
    color: #2A5C4D;
}

.stAlert { border-radius: 4px; border-left: 4px solid #D4A82A; }

.sb-section {
    font-size: 0.72rem;
    font-weight: 700;
    color: #D4A82A !important;
    text-transform: uppercase;
    letter-spacing: .12em;
    margin-top: 1rem;
    margin-bottom: 0.25rem;
    border-bottom: 1px solid #D4A82A;
    padding-bottom: 0.2rem;
    font-family: 'Georgia', serif;
}

/* Titre principal style "Paris" */
h1:first-of-type {
    border-bottom: 3px double #D4A82A;
    padding-bottom: 0.5rem;
}

/* Style des selectbox / radio (sidebar) */
section[data-testid="stSidebar"] .stMarkdown p {
    color: #FAF7F0 !important;
}

/* Code inline */
code {
    background-color: #E8DDC8 !important;
    color: #B33A3A !important;
    padding: 2px 6px;
    border-radius: 3px;
}

/* Dataframes */
.dataframe {
    border: 1px solid #D4C5A9;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES (cache pour performance)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """Charge les 4 fichiers pré-agrégés."""
    agg_horaire = pd.read_csv(DATA_AGG_HORAIRE, compression="gzip")
    agg_troncons = pd.read_csv(DATA_AGG_TRONCONS, compression="gzip")
    agg_temporel = pd.read_csv(DATA_AGG_TEMPOREL, compression="gzip")
    sample_diag = pd.read_csv(DATA_SAMPLE_DIAG, compression="gzip")
    return agg_horaire, agg_troncons, agg_temporel, sample_diag


def _fig(w, h):
    """Helper pour créer une figure matplotlib."""
    return plt.subplots(figsize=(w, h))


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ─── CHARGEMENT ───
    with st.spinner("⚙️ Chargement des données..."):
        agg_horaire, agg_troncons, agg_temporel, sample_diag = load_data()

    # ═══════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown('<p class="sb-section">📊 PROJET</p>', unsafe_allow_html=True)
        st.markdown("**Trafic routier parisien**")
        st.caption("Analyse des capteurs permanents de la Ville de Paris")

        st.markdown('<p class="sb-section">📁 DONNÉES</p>', unsafe_allow_html=True)
        st.caption(f"📄 {NB_LIGNES_TOTAL:,} mesures horaires")
        st.caption(f"🛣️ {NB_TRONCONS:,} tronçons surveillés")
        st.caption(f"📅 {PERIODE}")

        st.markdown('<p class="sb-section">🛠️ MÉTHODOLOGIE</p>',
                    unsafe_allow_html=True)
        st.caption("✅ Lecture par chunks (gestion mémoire)")
        st.caption("✅ Imputation par interpolation temporelle")
        st.caption("✅ Pré-agrégation pour performance")

        st.markdown('<p class="sb-section">🌐 SOURCE</p>', unsafe_allow_html=True)
        st.caption("Direction de la Voirie et des Déplacements — Ville de Paris")
        st.markdown("[opendata.paris.fr](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/)")

    # ═══════════════════════════════════════════════════════════════════════
    # EN-TÊTE
    # ═══════════════════════════════════════════════════════════════════════
    st.title("🚦 Trafic routier parisien — Dashboard d'analyse")
    st.caption(
        f"Analyse de **{NB_LIGNES_TOTAL:,} mesures horaires** issues de "
        f"**{NB_TRONCONS:,} capteurs** sur **{PERIODE}**."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # ONGLETS
    # ═══════════════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "🏠 Vue d'ensemble",
        "📊 Exploration",
        "⏱️ Tendances temporelles",
        "🗺️ Cartographie",
        "🚦 Tronçons",
    ])

    # ──────────────────────────────────────────────────────────────────────
    # TAB 0 : VUE D'ENSEMBLE
    # ──────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Vue d'ensemble du projet")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("📊 Mesures", f"{NB_LIGNES_TOTAL:,}")
        with c2:
            st.metric("🛣️ Tronçons", f"{NB_TRONCONS:,}")
        with c3:
            debit_moyen = agg_troncons["debit_moyen"].mean()
            st.metric("🚗 Débit moyen", f"{debit_moyen:.0f} véh/h")
        with c4:
            occup_moyenne = agg_troncons["occupation_moyenne"].mean()
            st.metric("⏱️ Occupation moy.", f"{occup_moyenne:.1f} %")

        st.markdown("---")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("### 🎯 Contexte du projet")
            st.markdown(f"""
            En tant que **data analyst pour la Ville de Paris**, ce dashboard
            analyse les données de trafic routier issues des **boucles
            électromagnétiques** implantées dans la chaussée du réseau
            routier parisien.

            **Indicateurs principaux :**
            - 🚗 **Débit horaire** : véhicules par heure
            - ⏱️ **Taux d'occupation** : % de temps où un véhicule est sur le capteur
            - 🚦 **État du trafic** : Fluide / Pré-saturé / Saturé / Bloqué

            **Volume de données :** **{NB_LIGNES_TOTAL:,} mesures**, soit ~1 Million
            de lignes, traitées avec une approche **Big Data**
            (chargement par chunks).
            """)

            st.markdown("### 🔬 Pipeline de traitement (CRISP-DM)")
            st.markdown("""
            1. **Chargement** par chunks de 100k lignes (gestion mémoire)
            2. **Suppression** des colonnes redondantes (15 → 9)
            3. **Imputation** par interpolation linéaire temporelle
            4. **Formatage** des dates et création de variables temporelles
            5. **Statistiques** descriptives par tronçon
            6. **Séries temporelles** : profils horaires & hebdomadaires
            7. **Visualisations** : histogrammes, diagramme fondamental
            8. **Encodage** ordinal (état trafic) + One-Hot (état arc)
            9. **Cartographie** interactive avec Folium
            """)

        with col2:
            # Répartition par état du trafic
            st.markdown("### 🚦 État dominant des tronçons")
            etat_counts = agg_troncons["etat_dominant"].value_counts()

            fig_donut, ax_donut = _fig(4.5, 4.5)
            colors_etat = {
                "Fluide": "#27AE60",
                "Pré-saturé": "#F39C12",
                "Saturé": "#E67E22",
                "Bloqué": "#C00000",
                "Inconnu": "#95A5A6",
            }
            colors = [colors_etat.get(e, "#2A5C4D") for e in etat_counts.index]

            ax_donut.pie(
                etat_counts.values, labels=etat_counts.index,
                autopct='%1.1f%%', colors=colors,
                wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
                textprops={"fontsize": 10},
                startangle=90
            )
            ax_donut.set_title("État dominant", fontsize=11, pad=10,
                                fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_donut, use_container_width=True)
            plt.close(fig_donut)

        st.markdown("---")

        # Top 10 et bottom 10
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🚗 Top 10 — Tronçons les plus chargés")
            top10 = agg_troncons.nlargest(10, "debit_moyen")[
                ["Libelle", "debit_moyen", "occupation_moyenne"]
            ].reset_index(drop=True)
            top10.columns = ["Tronçon", "Débit moy. (véh/h)",
                              "Occupation (%)"]
            top10.index = top10.index + 1
            st.dataframe(top10.round(1), use_container_width=True, height=400)

        with col2:
            st.markdown("### 🌿 Bottom 10 — Tronçons les moins fréquentés")
            bot10 = agg_troncons.nsmallest(10, "debit_moyen")[
                ["Libelle", "debit_moyen", "occupation_moyenne"]
            ].reset_index(drop=True)
            bot10.columns = ["Tronçon", "Débit moy. (véh/h)",
                              "Occupation (%)"]
            bot10.index = bot10.index + 1
            st.dataframe(bot10.round(1), use_container_width=True, height=400)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 1 : EXPLORATION
    # ──────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Exploration des distributions")

        # KPIs distributions
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Débit médian", f"{agg_troncons['debit_median'].median():.0f} véh/h")
        with c2:
            st.metric("Débit max observé",
                       f"{agg_troncons['debit_max'].max():.0f} véh/h")
        with c3:
            st.metric("Occupation max",
                       f"{agg_troncons['occupation_max'].max():.0f} %")

        st.markdown("---")

        # 📊 Histogrammes
        st.markdown("### 📊 Distribution des indicateurs")
        col1, col2 = st.columns(2)

        with col1:
            fig_h1, ax_h1 = _fig(7, 4.5)
            ax_h1.hist(agg_troncons["debit_moyen"], bins=40,
                        color="#2A5C4D", edgecolor="white", alpha=0.85)
            moy = agg_troncons["debit_moyen"].mean()
            med = agg_troncons["debit_moyen"].median()
            ax_h1.axvline(moy, color="red", linestyle="--", linewidth=2,
                           label=f"Moyenne ({moy:.0f})")
            ax_h1.axvline(med, color="green", linestyle="--", linewidth=2,
                           label=f"Médiane ({med:.0f})")
            ax_h1.set_title("Distribution du débit moyen par tronçon",
                             fontsize=11, fontweight="bold")
            ax_h1.set_xlabel("Débit moyen (véh/h)")
            ax_h1.set_ylabel("Nombre de tronçons")
            ax_h1.legend()
            plt.tight_layout()
            st.pyplot(fig_h1, use_container_width=True)
            plt.close(fig_h1)

        with col2:
            fig_h2, ax_h2 = _fig(7, 4.5)
            ax_h2.hist(agg_troncons["occupation_moyenne"], bins=40,
                        color="#27AE60", edgecolor="white", alpha=0.85)
            moy_o = agg_troncons["occupation_moyenne"].mean()
            ax_h2.axvline(moy_o, color="red", linestyle="--", linewidth=2,
                           label=f"Moyenne ({moy_o:.1f}%)")
            ax_h2.set_title("Distribution du taux d'occupation moyen",
                             fontsize=11, fontweight="bold")
            ax_h2.set_xlabel("Taux d'occupation (%)")
            ax_h2.set_ylabel("Nombre de tronçons")
            ax_h2.legend()
            plt.tight_layout()
            st.pyplot(fig_h2, use_container_width=True)
            plt.close(fig_h2)

        st.markdown("---")

        # 🚦 DIAGRAMME FONDAMENTAL — la pièce maîtresse
        st.markdown("### 🚦 Diagramme fondamental du trafic")
        st.markdown("""
        Le **diagramme fondamental** est un outil classique de l'ingénierie du trafic.
        Il croise **débit** et **taux d'occupation** pour révéler 3 régimes :

        - 🟢 **Fluide** : débit ↗ avec l'occupation
        - 🟡 **Capacité maximale** : pic de débit (15-25% d'occupation)
        - 🔴 **Saturé** : au-delà, **le débit chute** alors que l'occupation continue → bouchons
        """)

        fig_diag, ax_diag = _fig(11, 6)
        scatter = ax_diag.scatter(
            sample_diag["Taux d'occupation"], sample_diag["Débit horaire"],
            c=sample_diag["Débit horaire"], cmap="plasma",
            alpha=0.4, s=8
        )
        plt.colorbar(scatter, ax=ax_diag, label="Débit (véh/h)")
        ax_diag.set_title(
            "Diagramme fondamental du trafic\n(Débit vs Taux d'occupation)",
            fontsize=12, fontweight="bold", pad=15
        )
        ax_diag.set_xlabel("Taux d'occupation (%)")
        ax_diag.set_ylabel("Débit horaire (véh/h)")
        plt.tight_layout()
        st.pyplot(fig_diag, use_container_width=True)
        plt.close(fig_diag)

        st.info(
            "💡 **Lecture métier :** un même débit peut correspondre à un trafic "
            "*fluide* (faible occupation) ou *saturé* (forte occupation). "
            "Le **taux d'occupation** est l'indicateur clé pour distinguer "
            "ces deux situations radicalement différentes pour l'usager."
        )

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2 : TENDANCES TEMPORELLES
    # ──────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Analyse des tendances temporelles")

        # Profil 24h global (depuis agg_temporel)
        profil_24h = (
            agg_temporel.groupby("heure")
                        .agg(debit_moyen=("debit_moyen", "mean"),
                             occupation_moyenne=("occupation_moyenne", "mean"))
                        .reset_index()
        )

        heure_pic = profil_24h.loc[profil_24h["debit_moyen"].idxmax(), "heure"]
        heure_calme = profil_24h.loc[profil_24h["debit_moyen"].idxmin(), "heure"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🚦 Heure de pointe", f"{int(heure_pic)}h")
        with c2:
            st.metric("🌙 Heure la plus calme", f"{int(heure_calme)}h")
        with c3:
            st.metric("📊 Amplitude jour/nuit",
                       f"×{profil_24h['debit_moyen'].max() / max(profil_24h['debit_moyen'].min(), 1):.1f}")

        st.markdown("---")

        # 📈 Profil 24h
        st.markdown("### 📈 Profil de trafic sur 24 heures")

        fig_24h, ax1 = _fig(12, 5)

        color1 = "#2A5C4D"
        ax1.plot(profil_24h["heure"], profil_24h["debit_moyen"],
                 color=color1, linewidth=2.5, marker="o",
                 label="Débit horaire")
        ax1.fill_between(profil_24h["heure"], profil_24h["debit_moyen"],
                          alpha=0.3, color=color1)
        ax1.set_xlabel("Heure de la journée")
        ax1.set_ylabel("Débit moyen (véh/h)", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(range(24))

        ax2 = ax1.twinx()
        color2 = "#B33A3A"
        ax2.plot(profil_24h["heure"], profil_24h["occupation_moyenne"],
                 color=color2, linewidth=2.5, marker="s", linestyle="--",
                 label="Taux d'occupation")
        ax2.set_ylabel("Taux d'occupation moyen (%)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("Évolution du trafic au cours de la journée",
                   fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_24h, use_container_width=True)
        plt.close(fig_24h)

        st.markdown("---")

        # 📅 Semaine vs week-end
        st.markdown("### 📅 Comparaison semaine vs week-end")

        profil_we = (
            agg_temporel.groupby(["heure", "est_weekend"])
                        .agg(debit_moyen=("debit_moyen", "mean"))
                        .reset_index()
        )
        profil_we_pivot = profil_we.pivot(
            index="heure", columns="est_weekend", values="debit_moyen"
        )
        profil_we_pivot.columns = ["Semaine", "Week-end"]

        fig_we, ax_we = _fig(12, 5)
        profil_we_pivot.plot(ax=ax_we, linewidth=2.5, marker="o",
                              color=["#2A5C4D", "#D4A82A"])
        ax_we.set_title("Débit horaire moyen — Semaine vs Week-end",
                         fontsize=12, fontweight="bold")
        ax_we.set_xlabel("Heure")
        ax_we.set_ylabel("Débit moyen (véh/h)")
        ax_we.set_xticks(range(24))
        ax_we.legend(title="Type de jour")
        plt.tight_layout()
        st.pyplot(fig_we, use_container_width=True)
        plt.close(fig_we)

        st.markdown("---")

        # 🔥 HEATMAP — la pièce visuelle phare
        st.markdown("### 🔥 Heatmap : congestion par heure × jour de la semaine")

        labels_jours = ["Lundi", "Mardi", "Mercredi", "Jeudi",
                         "Vendredi", "Samedi", "Dimanche"]

        heatmap_data = (
            agg_temporel.groupby(["heure", "num_jour_sem"])
                        .agg(debit_moyen=("debit_moyen", "mean"))
                        .reset_index()
                        .pivot(index="heure", columns="num_jour_sem",
                                values="debit_moyen")
        )
        heatmap_data.columns = labels_jours

        fig_hm, ax_hm = _fig(11, 8)
        sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".0f",
                     cbar_kws={"label": "Débit moyen (véh/h)"},
                     ax=ax_hm, linewidths=0.3, linecolor="white")
        ax_hm.set_title("Heatmap du débit horaire — Heure × Jour de la semaine",
                          fontsize=12, fontweight="bold", pad=15)
        ax_hm.set_xlabel("Jour de la semaine")
        ax_hm.set_ylabel("Heure de la journée")
        plt.tight_layout()
        st.pyplot(fig_hm, use_container_width=True)
        plt.close(fig_hm)

        st.info(
            "💡 **Insights :** zone la plus chaude entre **17h-19h en semaine** "
            "(pointe domicile-travail). Pic secondaire à **8h-9h**. "
            "Pattern week-end décalé vers **14h-18h** (loisirs)."
        )

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3 : CARTOGRAPHIE
    # ──────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Cartographie interactive du trafic parisien")

        st.markdown("""
        Visualisation des **2 985 tronçons surveillés** sur la carte de Paris.
        Cliquez sur un point pour voir le détail.
        """)

        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            niveau_filtre = st.radio(
                "Filtre par niveau de trafic :",
                ["Tous", "Fort", "Moyen", "Faible"],
                horizontal=True
            )
        with col2:
            type_carte = st.radio(
                "Type de visualisation :",
                ["📍 Marqueurs", "🔥 Heatmap"],
                horizontal=True
            )

        # Filtrer les données
        geo_df = agg_troncons.dropna(subset=["lat", "lon"]).copy()
        q33 = geo_df["debit_moyen"].quantile(0.33)
        q66 = geo_df["debit_moyen"].quantile(0.66)

        if niveau_filtre == "Fort":
            geo_df = geo_df[geo_df["debit_moyen"] >= q66]
        elif niveau_filtre == "Moyen":
            geo_df = geo_df[(geo_df["debit_moyen"] >= q33) &
                            (geo_df["debit_moyen"] < q66)]
        elif niveau_filtre == "Faible":
            geo_df = geo_df[geo_df["debit_moyen"] < q33]

        st.caption(f"📍 **{len(geo_df):,}** tronçons affichés")

        # Carte
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=12,
                       tiles="cartodbpositron")

        if type_carte == "📍 Marqueurs":

            def get_color(debit):
                if debit < q33: return "#27AE60"
                elif debit < q66: return "#F39C12"
                else: return "#C00000"

            def get_radius(debit):
                max_d = geo_df["debit_moyen"].max()
                return 3 + (debit / max_d * 8) if max_d > 0 else 3

            # Limiter le nombre de marqueurs pour performance
            display_df = geo_df.sample(min(500, len(geo_df)), random_state=42)

            for _, row in display_df.iterrows():
                libelle = str(row["Libelle"])[:60]
                debit = row["debit_moyen"]
                occup = row["occupation_moyenne"]
                etat = row["etat_dominant"]
                popup_html = (
                    f"<b>{libelle}</b><br>"
                    f"🚗 Débit moyen : {debit:.0f} véh/h<br>"
                    f"⏱️ Occupation : {occup:.1f} %<br>"
                    f"🚦 État dominant : {etat}"
                )

                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=get_radius(debit),
                    popup=folium.Popup(popup_html, max_width=300),
                    color=get_color(debit),
                    fillColor=get_color(debit),
                    fillOpacity=0.7,
                    weight=2,
                ).add_to(m)

        else:  # Heatmap
            heat_data = [[row["lat"], row["lon"], row["debit_moyen"]]
                         for _, row in geo_df.iterrows()]
            HeatMap(
                heat_data, radius=15, blur=20, max_zoom=13,
                gradient={0.2: "#27AE60", 0.5: "#F39C12", 0.8: "#C00000"}
            ).add_to(m)

        # Afficher la carte
        st_folium(m, width=None, height=600,
                   returned_objects=[],
                   key=f"map_{niveau_filtre}_{type_carte}")

        # Légende sous la carte
        st.markdown("""
        **🚦 Légende :**
        🟢 Faible (< 33%) ·
        🟠 Moyen (33-66%) ·
        🔴 Élevé (> 66%) du débit moyen
        """)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4 : TRONÇONS (drill-down par rue)
    # ──────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Analyse détaillée par tronçon")

        # Sélecteur de tronçon avec recherche
        st.markdown("### 🔍 Sélectionnez un tronçon")

        col1, col2 = st.columns([3, 1])
        with col1:
            recherche = st.text_input(
                "Rechercher un tronçon (par nom de rue / avenue) :",
                value="Champs"
            )

        # Filtrer les tronçons
        if recherche:
            options = agg_troncons[
                agg_troncons["Libelle"].str.contains(recherche, case=False,
                                                      na=False)
            ].sort_values("debit_moyen", ascending=False)
        else:
            options = agg_troncons.sort_values("debit_moyen", ascending=False)

        if len(options) == 0:
            st.warning("Aucun tronçon trouvé pour cette recherche.")
            st.stop()

        with col2:
            st.metric("Résultats", len(options))

        # Choix du tronçon
        if len(options) > 0:
            options_dict = dict(zip(
                options["Identifiant arc"],
                options["Libelle"] + " (Identifiant : " +
                options["Identifiant arc"].astype(str) + ")"
            ))
            selected_arc = st.selectbox(
                "Tronçon à analyser :",
                options=list(options_dict.keys()),
                format_func=lambda x: options_dict[x]
            )

            # Récupérer les infos du tronçon
            info = agg_troncons[
                agg_troncons["Identifiant arc"] == selected_arc
            ].iloc[0]

            st.markdown("---")
            st.markdown(f"### 📍 {info['Libelle']}")

            # KPIs du tronçon
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("🚗 Débit moyen",
                           f"{info['debit_moyen']:.0f} véh/h")
            with c2:
                st.metric("🚀 Débit max",
                           f"{info['debit_max']:.0f} véh/h")
            with c3:
                st.metric("⏱️ Occupation moy.",
                           f"{info['occupation_moyenne']:.1f} %")
            with c4:
                st.metric("🚦 État dominant", info["etat_dominant"])

            st.markdown("---")

            # Profil 24h × jour de la semaine pour ce tronçon
            arc_data = agg_horaire[
                agg_horaire["Identifiant arc"] == selected_arc
            ].copy()

            if len(arc_data) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📈 Profil de trafic sur 24h")
                    profil_arc = arc_data.groupby("heure")["debit_moyen"].mean()

                    fig_arc, ax_arc = _fig(7, 4.5)
                    ax_arc.plot(profil_arc.index, profil_arc.values,
                                 color="#2A5C4D", linewidth=2.5, marker="o")
                    ax_arc.fill_between(profil_arc.index, profil_arc.values,
                                          alpha=0.3, color="#2A5C4D")
                    ax_arc.set_title(f"Débit horaire moyen",
                                       fontsize=11, fontweight="bold")
                    ax_arc.set_xlabel("Heure")
                    ax_arc.set_ylabel("Débit (véh/h)")
                    ax_arc.set_xticks(range(0, 24, 2))
                    plt.tight_layout()
                    st.pyplot(fig_arc, use_container_width=True)
                    plt.close(fig_arc)

                with col2:
                    st.markdown("#### 🔥 Heatmap horaire × jour")
                    labels_jours = ["Lun", "Mar", "Mer", "Jeu",
                                     "Ven", "Sam", "Dim"]

                    heatmap_arc = arc_data.pivot_table(
                        values="debit_moyen", index="heure",
                        columns="num_jour_sem"
                    )
                    if heatmap_arc.shape[1] == 7:
                        heatmap_arc.columns = labels_jours

                    fig_arc_hm, ax_arc_hm = _fig(7, 4.5)
                    sns.heatmap(heatmap_arc, cmap="YlOrRd", ax=ax_arc_hm,
                                 cbar_kws={"label": "véh/h"})
                    ax_arc_hm.set_title("Trafic par heure × jour",
                                          fontsize=11, fontweight="bold")
                    ax_arc_hm.set_xlabel("")
                    ax_arc_hm.set_ylabel("Heure")
                    plt.tight_layout()
                    st.pyplot(fig_arc_hm, use_container_width=True)
                    plt.close(fig_arc_hm)

                # Mini-carte du tronçon
                if not pd.isna(info["lat"]) and not pd.isna(info["lon"]):
                    st.markdown("#### 🗺️ Localisation")
                    m_arc = folium.Map(
                        location=[info["lat"], info["lon"]],
                        zoom_start=15, tiles="cartodbpositron"
                    )
                    folium.CircleMarker(
                        location=[info["lat"], info["lon"]],
                        radius=10,
                        popup=info["Libelle"],
                        color="#C00000", fillColor="#C00000",
                        fillOpacity=0.8, weight=3
                    ).add_to(m_arc)
                    st_folium(m_arc, width=None, height=400,
                              returned_objects=[],
                              key=f"map_arc_{selected_arc}")


if __name__ == "__main__":
    main()
