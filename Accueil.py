"""
═══════════════════════════════════════════════════════════════════════════
🏦 DASHBOARD MARKETING BANCAIRE — Application Streamlit
═══════════════════════════════════════════════════════════════════════════
Dashboard ML pour la prédiction de souscription à un dépôt à terme.
Architecture single-page avec onglets.

Auteur : Djakpa
Date   : 2026
═══════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─── CONSTANTES ───
DATA_PATH = "bankfull.csv"
COST = 5          # Coût d'un appel téléphonique (€)
GAIN = 200        # Gain moyen par souscription (€)

# ═══════════════════════════════════════════════════════════════════════════
# STYLE MATPLOTLIB
# ═══════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="Blues")
for k, v in {
    "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#d9e6f2", "grid.color": "#dfeaf5",
    "axes.labelcolor": "#16324f", "xtick.color": "#16324f",
    "ytick.color": "#16324f", "text.color": "#16324f",
    "axes.titlecolor": "#1F4E79",
}.items():
    plt.rcParams[k] = v

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Marketing Bancaire — Dashboard ML",
    page_icon="🏦",
    layout="wide",
)

# ─── STYLE CSS (identique à votre style préféré) ───
st.markdown("""
<style>
.main { background: linear-gradient(135deg,#f4f8fc 0%,#fbfdff 50%,#eef5fb 100%); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#eaf2fb 0%,#f4f8fc 100%) !important;
    border-right: 1px solid #d7e3f1;
}

h1,h2,h3 { color:#1F4E79 !important; }

.metric-card {
    background: linear-gradient(135deg,#ffffff 0%,#f9fcff 100%);
    border-radius:14px; padding:1rem;
    border:1px solid #d9e6f2;
    box-shadow:0 3px 10px rgba(31,78,121,.10);
    min-height:88px; color:#16324f;
}

div[data-testid="stMetric"] {
    background-color:#ffffff; border:1px solid #d9e6f2;
    padding:0.8rem; border-radius:12px;
    box-shadow:0 2px 8px rgba(31,78,121,.08);
}

div[data-baseweb="tab-list"] { gap:8px; }
button[data-baseweb="tab"] {
    background-color:#edf4fb !important;
    border-radius:10px 10px 0 0 !important;
    color:#1F4E79 !important;
    border:1px solid #d7e3f1 !important;
    padding:0.5rem 1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color:#1F4E79 !important;
    color:white !important;
    border-color:#1F4E79 !important;
}

.stButton > button {
    background: linear-gradient(135deg,#1F4E79 0%,#2E7D32 100%);
    color:white; border:none; border-radius:10px;
    padding:0.6rem 1.2rem; font-weight:600; width:100%;
}
.stButton > button:hover { filter:brightness(1.05); }
.stAlert { border-radius:12px; }

.sb-section {
    font-size:0.72rem; font-weight:700; color:#1F4E79;
    text-transform:uppercase; letter-spacing:.09em;
    margin-top:1rem; margin-bottom:0.25rem;
    border-bottom:1px solid #c9ddf0; padding-bottom:0.2rem;
}

.result-box {
    border-radius:12px; padding:1rem 1.1rem;
    margin-top:0.6rem; font-size:0.95rem; line-height:1.6;
}
.result-target   { background:#e8f5e9; border:1.5px solid #27ae60; color:#1b5e20; }
.result-notarget { background:#fdecea; border:1.5px solid #e53935; color:#7f0000; }

.prob-bar-wrap {
    background:#dce8f5; border-radius:8px;
    height:16px; overflow:hidden; margin:0.5rem 0 0.2rem;
}
.prob-bar-fill { height:100%; border-radius:8px; }
.prob-value { font-size:1.7rem; font-weight:800; margin-top:0.1rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CHARGEMENT & PRÉPARATION
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset bancaire."""
    return pd.read_csv(path, sep=";")


@st.cache_data
def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Crée des variables dérivées utiles à l'analyse."""
    df = df_raw.copy()

    # Tranches d'âge (variable catégorielle dérivée)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["<30", "30-45", "45-60", "60+"]
    )

    # Tranches de solde
    df["balance_group"] = pd.cut(
        df["balance"],
        bins=[-np.inf, 0, 1000, 5000, np.inf],
        labels=["Négatif", "Faible", "Moyen", "Élevé"]
    )

    # Variable cible numérique
    df["y_num"] = (df["y"] == "yes").astype(int)

    # A-t-il déjà été contacté ?
    df["contacted_before"] = (df["pdays"] != -1).astype(int)

    return df


def get_model_columns() -> tuple[list[str], list[str]]:
    """Retourne (variables numériques, variables catégorielles) pour le modèle."""
    return (
        ["age", "balance", "day", "campaign", "pdays", "previous", "contacted_before"],
        ["job", "marital", "education", "default", "housing", "loan",
         "contact", "month", "poutcome"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df: pd.DataFrame, test_size: float = 0.2,
                 random_state: int = 42) -> dict:
    """Entraîne 4 modèles de classification et retourne les résultats."""
    num_cols, cat_cols = get_model_columns()
    X = df[num_cols + cat_cols]
    y = df["y_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline de prétraitement
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    # 4 modèles à comparer
    estimators = {
        "Régression logistique": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                     random_state=random_state),
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10,
                                                class_weight="balanced",
                                                random_state=random_state, n_jobs=-1),
    }

    results = {}
    for name, est in estimators.items():
        pipe = Pipeline([("pre", pre), ("clf", est)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "pipeline": pipe,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "cm": confusion_matrix(y_test, y_pred),
            "pr_curve": precision_recall_curve(y_test, y_proba),
            "roc_curve": roc_curve(y_test, y_proba),
        }

    # Tableau de comparaison
    comparison = pd.DataFrame([
        {"Modèle": k, "Accuracy": v["accuracy"], "Precision": v["precision"],
         "Recall": v["recall"], "F1": v["f1"], "ROC-AUC": v["roc_auc"]}
        for k, v in results.items()
    ]).set_index("Modèle")

    return {
        "results": results,
        "comparison": comparison,
        "X_test": X_test,
        "y_test": y_test,
        "best_model": "Random Forest",
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERING (SEGMENTATION)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def perform_clustering(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """Segmentation par K-Means sur les variables numériques."""
    cluster_features = ["age", "balance", "campaign", "pdays", "previous"]
    X_clu = df[cluster_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clu)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # ACP pour visualisation 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    seg_df = df.copy()
    seg_df["Cluster"] = clusters
    seg_df["PCA1"] = X_pca[:, 0]
    seg_df["PCA2"] = X_pca[:, 1]

    # Profil moyen par cluster
    cluster_profile = seg_df.groupby("Cluster").agg(
        nb_clients=("age", "count"),
        age_moyen=("age", "mean"),
        balance_moyen=("balance", "mean"),
        campaign_moyen=("campaign", "mean"),
        taux_souscription=("y_num", "mean"),
    ).reset_index()
    cluster_profile["taux_souscription"] = cluster_profile["taux_souscription"] * 100

    # Attribuer un nom métier
    def label_cluster(row):
        if row["age_moyen"] < 30:
            return "Jeunes actifs"
        elif row["age_moyen"] >= 55:
            return "Seniors / Retraités"
        elif row["balance_moyen"] > 2000:
            return "Aisés établis"
        else:
            return "Actifs moyens"

    cluster_profile["Segment métier"] = cluster_profile.apply(label_cluster, axis=1)

    return {
        "seg_df": seg_df,
        "cluster_profile": cluster_profile,
        "n_clusters": n_clusters,
    }


# ═══════════════════════════════════════════════════════════════════════════
# IMPORTANCE DES VARIABLES
# ═══════════════════════════════════════════════════════════════════════════
def feat_importance(pipeline) -> pd.DataFrame:
    """Récupère l'importance des variables d'un pipeline scikit-learn."""
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    # Récupérer les noms des features après one-hot
    num_cols, cat_cols = get_model_columns()
    cat_names = list(pre.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_names

    # Importance selon le modèle
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame({"Variable": feature_names,
                             "Importance": [0]*len(feature_names)})

    df_imp = pd.DataFrame({
        "Variable": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    return df_imp


# ═══════════════════════════════════════════════════════════════════════════
# CALCUL D'UNE STRATÉGIE MARKETING (pour Décision)
# ═══════════════════════════════════════════════════════════════════════════
def compute_strategy(y_test, y_proba, threshold, cost, gain):
    """Calcule les indicateurs business pour un seuil donné."""
    y_pred = (y_proba >= threshold).astype(int)

    nb_contacts = int(y_pred.sum())
    nb_repondeurs = int(((y_pred == 1) & (y_test == 1)).sum())
    nb_fp = int(((y_pred == 1) & (y_test == 0)).sum())
    nb_fn = int(((y_pred == 0) & (y_test == 1)).sum())

    cout_total = nb_contacts * cost
    gain_total = nb_repondeurs * gain
    profit = gain_total - cout_total
    roi = (profit / cout_total * 100) if cout_total > 0 else 0
    taux_conv = (nb_repondeurs / nb_contacts * 100) if nb_contacts > 0 else 0
    taux_capture = (nb_repondeurs / y_test.sum() * 100) if y_test.sum() > 0 else 0

    return {
        "Seuil": threshold,
        "Clients contactés": nb_contacts,
        "Répondeurs captés": nb_repondeurs,
        "Faux positifs": nb_fp,
        "Faux négatifs": nb_fn,
        "Taux de conversion (%)": round(taux_conv, 2),
        "Taux de capture (%)": round(taux_capture, 2),
        "Coût total (€)": cout_total,
        "Profit estimé (€)": profit,
        "ROI (%)": round(roi, 2),
    }


def _fig(w, h):
    """Helper pour créer une figure matplotlib."""
    return plt.subplots(figsize=(w, h))


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ─── CHARGEMENT ───
    df_raw = load_data(DATA_PATH)
    df = prepare_features(df_raw)

    # ─── ENTRAÎNEMENT ───
    with st.spinner("⚙️ Entraînement des modèles en cours..."):
        modeling = train_models(df)

    with st.spinner("⚙️ Segmentation en cours..."):
        segmentation = perform_clustering(df, n_clusters=4)

    # Modèle de référence pour la décision marketing
    best_model_name = modeling["best_model"]
    y_test = modeling["y_test"]
    y_proba_best = modeling["results"][best_model_name]["y_proba"]

    # ═══════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown('<p class="sb-section">📊 PROJET</p>', unsafe_allow_html=True)
        st.markdown("**Marketing Bancaire**")
        st.caption("Prédiction de souscription à un dépôt à terme")

        st.markdown('<p class="sb-section">📁 DONNÉES</p>', unsafe_allow_html=True)
        st.caption(f"📄 {len(df):,} clients × {df_raw.shape[1]} variables")
        st.caption(f"✅ Souscripteurs : {df['y_num'].sum():,} ({df['y_num'].mean()*100:.1f}%)")

        st.markdown('<p class="sb-section">🤖 MODÈLE RETENU</p>', unsafe_allow_html=True)
        st.success(f"🏆 **{best_model_name}**")
        st.caption(f"ROC-AUC : {modeling['results'][best_model_name]['roc_auc']:.3f}")

        st.markdown('<p class="sb-section">💰 PARAMÈTRES BUSINESS</p>',
                    unsafe_allow_html=True)
        global COST, GAIN
        COST = st.number_input("Coût d'un appel (€)", 1.0, 50.0, 5.0, step=0.5)
        GAIN = st.number_input("Gain par souscription (€)", 50.0, 2000.0,
                                200.0, step=10.0)

    # ═══════════════════════════════════════════════════════════════════════
    # EN-TÊTE
    # ═══════════════════════════════════════════════════════════════════════
    st.title("🏦 Marketing Bancaire — Dashboard ML")
    st.caption("Analyse, segmentation, prédiction et optimisation des campagnes "
               "de souscription à un dépôt à terme.")

    # ═══════════════════════════════════════════════════════════════════════
    # ONGLETS
    # ═══════════════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "🏠 Vue d'ensemble",
        "📊 Analyse exploratoire",
        "🎯 Segmentation",
        "🤖 Modélisation",
        "💰 Décision marketing",
        "🔮 Prédiction",
    ])

    # ──────────────────────────────────────────────────────────────────────
    # TAB 0 : VUE D'ENSEMBLE
    # ──────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Vue d'ensemble du projet")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("👥 Clients", f"{len(df):,}")
        with c2:
            st.metric("✅ Souscripteurs", f"{df['y_num'].sum():,}",
                      delta=f"{df['y_num'].mean()*100:.1f}%")
        with c3:
            st.metric("📅 Âge moyen", f"{df['age'].mean():.0f} ans")
        with c4:
            st.metric("💰 Solde moyen", f"{df['balance'].mean():,.0f} €")

        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🎯 Contexte du projet")
            st.markdown("""
            Une **banque portugaise** mène des campagnes de marketing direct par
            téléphone pour proposer à ses clients de souscrire à un **dépôt à terme**.

            **Le problème :** le taux de conversion réel n'est que de **11.7%** —
            soit **88 appels sur 100 sont infructueux**, ce qui plombe le ROI des
            campagnes.

            **L'objectif :** construire un modèle prédictif capable d'identifier
            *à l'avance* les clients à fort potentiel de souscription.
            """)

            st.markdown("### 🔬 Méthodologie CRISP-DM")
            st.markdown("""
            1. **Compréhension métier** — analyse des objectifs et du contexte
            2. **Compréhension des données** — EDA, statistiques descriptives
            3. **Préparation** — encodage, gestion du déséquilibre, split train/test
            4. **Modélisation** — 4 algorithmes testés (LR, KNN, DT, RF)
            5. **Évaluation** — F1-score, ROC-AUC, matrice de confusion
            6. **Déploiement** — ce dashboard 🚀
            """)

        with col2:
            # Donut de la cible
            fig_donut, ax_donut = _fig(5, 5)
            counts = df["y"].value_counts()
            colors = ["#1F4E79" if x == "no" else "#27AE60" for x in counts.index]
            ax_donut.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                          colors=colors, wedgeprops=dict(width=0.4),
                          textprops={'fontsize': 11})
            ax_donut.set_title("Répartition de la cible", fontsize=12, pad=10)
            plt.tight_layout()
            st.pyplot(fig_donut, use_container_width=True)
            plt.close(fig_donut)

        st.markdown("---")
        st.markdown("### 🏆 Récapitulatif des performances")
        st.dataframe(modeling["comparison"].round(3), use_container_width=True)
        st.success(
            f"🏆 **{best_model_name}** retenu pour son meilleur compromis "
            f"F1-score / ROC-AUC sur les classes déséquilibrées."
        )

    # ──────────────────────────────────────────────────────────────────────
    # TAB 1 : ANALYSE EXPLORATOIRE (EDA)
    # ──────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Analyse exploratoire des données")

        # Filtres
        with st.expander("🎛️ Filtres", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                age_range = st.slider("Tranche d'âge",
                                      int(df["age"].min()), int(df["age"].max()),
                                      (int(df["age"].min()), int(df["age"].max())))
            with c2:
                jobs_sel = st.multiselect("Profession(s)",
                                          sorted(df["job"].unique()),
                                          default=sorted(df["job"].unique()))
            with c3:
                marital_sel = st.multiselect("Statut marital",
                                              sorted(df["marital"].unique()),
                                              default=sorted(df["marital"].unique()))

        df_f = df[
            (df["age"].between(age_range[0], age_range[1])) &
            (df["job"].isin(jobs_sel)) &
            (df["marital"].isin(marital_sel))
        ]

        st.caption(f"📊 **{len(df_f):,}** clients filtrés "
                   f"({len(df_f)/len(df)*100:.1f}% du total)")

        # KPIs filtrés
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Clients", f"{len(df_f):,}")
        with c2:
            taux_f = df_f["y_num"].mean()*100 if len(df_f) > 0 else 0
            delta_f = taux_f - df["y_num"].mean()*100
            st.metric("Taux conversion", f"{taux_f:.1f}%",
                      delta=f"{delta_f:+.1f} pts vs moyenne")
        with c3:
            st.metric("Solde moyen",
                       f"{df_f['balance'].mean():,.0f} €" if len(df_f) > 0 else "—")

        if len(df_f) == 0:
            st.warning("Aucun client ne correspond aux filtres.")
            st.stop()

        st.markdown("---")

        # Distributions numériques
        st.markdown("### 📊 Distributions des variables clés")
        c1, c2 = st.columns(2)
        with c1:
            fig_a, ax_a = _fig(6, 4)
            for cat, color in [("yes", "#27AE60"), ("no", "#1F4E79")]:
                ax_a.hist(df_f[df_f["y"] == cat]["age"], bins=30,
                          alpha=0.6, label=cat, color=color)
            ax_a.set_title("Distribution de l'âge", fontsize=12, pad=10)
            ax_a.set_xlabel("Âge"); ax_a.set_ylabel("Effectif")
            ax_a.legend(title="Souscription")
            plt.tight_layout(); st.pyplot(fig_a, use_container_width=True); plt.close(fig_a)

        with c2:
            fig_b, ax_b = _fig(6, 4)
            df_box = df_f[df_f["balance"].between(-2000, 20000)]  # filtre outliers extrêmes
            sns.boxplot(data=df_box, x="y", y="balance",
                         palette={"no": "#1F4E79", "yes": "#27AE60"}, ax=ax_b)
            ax_b.set_title("Solde par souscription", fontsize=12, pad=10)
            ax_b.set_xlabel("Souscription"); ax_b.set_ylabel("Solde (€)")
            plt.tight_layout(); st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)

        # Taux de conversion par segment
        st.markdown("### 🎯 Taux de conversion par segment")

        var_cat = st.selectbox(
            "Variable à analyser",
            ["job", "marital", "education", "month", "contact",
             "poutcome", "age_group", "housing", "loan"]
        )

        taux_seg = df_f.groupby(var_cat)["y_num"].agg(["mean", "count"]).reset_index()
        taux_seg["mean"] = taux_seg["mean"] * 100
        taux_seg = taux_seg.sort_values("mean", ascending=False)

        fig_t, ax_t = _fig(10, 5)
        moyenne_globale = df["y_num"].mean() * 100
        colors_t = ["#27AE60" if x > moyenne_globale else "#E74C3C"
                    for x in taux_seg["mean"]]
        ax_t.bar(taux_seg[var_cat].astype(str), taux_seg["mean"],
                 color=colors_t, edgecolor='black')
        ax_t.axhline(y=moyenne_globale, color='blue', linestyle='--',
                     label=f"Moyenne ({moyenne_globale:.1f}%)")
        ax_t.set_title(f"Taux de souscription par '{var_cat}'", fontsize=12, pad=10)
        ax_t.set_ylabel("Taux (%)")
        ax_t.tick_params(axis='x', rotation=45)
        ax_t.legend()
        plt.tight_layout(); st.pyplot(fig_t, use_container_width=True); plt.close(fig_t)

        with st.expander("📋 Tableau détaillé"):
            taux_display = taux_seg.copy()
            taux_display.columns = [var_cat, "Taux conv. (%)", "Effectif"]
            taux_display["Taux conv. (%)"] = taux_display["Taux conv. (%)"].round(2)
            st.dataframe(taux_display, use_container_width=True, hide_index=True)

        # Matrice de corrélation
        st.markdown("### 🔗 Matrice de corrélation")
        num_vars = ["age", "balance", "day", "campaign", "pdays",
                    "previous", "y_num"]
        corr = df_f[num_vars].corr()
        fig_c, ax_c = _fig(8, 6)
        sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, fmt=".2f", ax=ax_c)
        ax_c.set_title("Matrice de corrélation", fontsize=12, pad=10)
        plt.tight_layout(); st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2 : SEGMENTATION
    # ──────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Segmentation des clients (K-Means)")

        cp = segmentation["cluster_profile"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("🎯 Clusters", segmentation["n_clusters"])
        with c2:
            st.metric("📊 Variables", "5")
        with c3:
            best_clu = cp.loc[cp["taux_souscription"].idxmax(), "Cluster"]
            best_taux = cp["taux_souscription"].max()
            st.metric(f"🏆 Cluster #{best_clu}",
                       f"{best_taux:.1f}% conv.")
        with c4:
            st.metric("📈 Inertie ratio",
                       f"{cp['nb_clients'].max()/cp['nb_clients'].min():.1f}")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            # PCA scatter
            fig_pca, ax_pca = _fig(7, 6)
            sample_seg = segmentation["seg_df"].sample(min(3000, len(segmentation["seg_df"])),
                                                        random_state=42)
            colors_pca = ["#1F4E79", "#E67E22", "#27AE60", "#9B59B6", "#E74C3C"]
            for i in range(segmentation["n_clusters"]):
                mask = sample_seg["Cluster"] == i
                ax_pca.scatter(sample_seg[mask]["PCA1"], sample_seg[mask]["PCA2"],
                                c=colors_pca[i % len(colors_pca)],
                                label=f"Cluster {i}", alpha=0.5, s=20)
            ax_pca.set_title("Visualisation 2D (ACP) des clusters", fontsize=12, pad=10)
            ax_pca.set_xlabel("Composante 1"); ax_pca.set_ylabel("Composante 2")
            ax_pca.legend(title="Cluster", bbox_to_anchor=(1.02, 1),
                           loc="upper left", borderaxespad=0)
            plt.tight_layout(); st.pyplot(fig_pca, use_container_width=True); plt.close(fig_pca)

        with c2:
            # Bar age vs balance par cluster
            fig_bar, ax_bar = _fig(7, 6)
            x = np.arange(len(cp)); w = 0.35
            ax_bar.bar(x - w/2, cp["balance_moyen"], w,
                       label="Solde moy. (€)", color="#1F4E79", alpha=0.85)
            ax2b = ax_bar.twinx()
            ax2b.bar(x + w/2, cp["age_moyen"], w,
                     label="Âge moy.", color="#9ecae1", alpha=0.85)
            ax_bar.set_title("Solde & Âge par cluster", fontsize=12, pad=10)
            ax_bar.set_xlabel("Cluster")
            ax_bar.set_ylabel("Solde moy. (€)", color="#1F4E79")
            ax2b.set_ylabel("Âge moy.", color="#4A90E2")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels([f"C{int(c)}" for c in cp["Cluster"]])
            l1, lb1 = ax_bar.get_legend_handles_labels()
            l2, lb2 = ax2b.get_legend_handles_labels()
            ax_bar.legend(l1+l2, lb1+lb2, loc="upper left", fontsize=9)
            plt.tight_layout(); st.pyplot(fig_bar, use_container_width=True); plt.close(fig_bar)

        st.markdown("### 📋 Profils des clusters")
        st.dataframe(cp.round(2), use_container_width=True, hide_index=True)

        st.markdown("#### Lecture métier")
        for _, row in cp.round(2).iterrows():
            st.markdown(
                f"**Cluster {int(row['Cluster'])} — {row['Segment métier']}** : "
                f"{int(row['nb_clients']):,} clients · âge moy. {row['age_moyen']:.0f} ans · "
                f"solde moy. {row['balance_moyen']:.0f} € · "
                f"taux souscription **{row['taux_souscription']:.1f}%**"
            )

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3 : MODÉLISATION
    # ──────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Modélisation prédictive de la souscription")
        st.dataframe(modeling["comparison"].round(3), use_container_width=True)

        model_choice = st.selectbox("Modèle à visualiser",
                                     list(modeling["results"].keys()),
                                     index=3)
        sel = modeling["results"][model_choice]

        ca, cb = st.columns(2)
        with ca:
            fig_cm, ax_cm = _fig(6, 5)
            sns.heatmap(sel["cm"], annot=True, fmt="d", cbar=False, cmap="Blues",
                        ax=ax_cm,
                        xticklabels=["Non-souscripteur", "Souscripteur"],
                        yticklabels=["Non-souscripteur", "Souscripteur"],
                        annot_kws={"size": 14})
            ax_cm.set_title(f"Matrice de confusion — {model_choice}",
                             fontsize=12, pad=10)
            ax_cm.set_xlabel("Prédit"); ax_cm.set_ylabel("Réel")
            plt.tight_layout(); st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)

        with cb:
            try:
                imp = feat_importance(sel["pipeline"]).head(15)
                fig_imp, ax_imp = _fig(7, 6)
                sns.barplot(data=imp, x="Importance", y="Variable", ax=ax_imp,
                             palette="Blues_r")
                ax_imp.set_title(f"Top 15 variables — {model_choice}",
                                  fontsize=12, pad=10)
                ax_imp.set_xlabel("Importance"); ax_imp.set_ylabel("")
                plt.tight_layout(); st.pyplot(fig_imp, use_container_width=True); plt.close(fig_imp)
            except Exception as e:
                st.info("Importance des variables non disponible pour ce modèle.")

        st.markdown("### Courbes ROC — Comparaison des modèles")
        fig_roc, ax_roc = _fig(10, 5)
        for (name, res), color in zip(
            modeling["results"].items(),
            ["#1F4E79", "#2E7D32", "#e67e22", "#8e44ad"]
        ):
            fpr, tpr, _ = res["roc_curve"]
            ax_roc.plot(fpr, tpr,
                         label=f"{name} (AUC={res['roc_auc']:.3f})",
                         color=color, linewidth=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax_roc.set_xlabel("Taux de faux positifs", fontsize=11)
        ax_roc.set_ylabel("Taux de vrais positifs", fontsize=11)
        ax_roc.set_title("Courbes ROC — Comparaison des modèles",
                          fontsize=13, pad=10)
        ax_roc.legend(fontsize=10); ax_roc.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_roc, use_container_width=True); plt.close(fig_roc)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4 : DÉCISION MARKETING
    # ──────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Optimisation de la décision marketing")

        # Seuils testés
        thr_r = np.round(np.arange(0.00, 0.96, 0.01), 2)
        rows = [compute_strategy(y_test, y_proba_best, t, COST, GAIN) for t in thr_r]
        thr_df = pd.DataFrame(rows)

        # Meilleurs seuils
        best_profit_thr = float(thr_df.loc[thr_df["Profit estimé (€)"].idxmax(), "Seuil"])
        best_roi_thr = float(thr_df.loc[thr_df["ROI (%)"].idxmax(), "Seuil"])

        # Stratégies clés
        s_all = compute_strategy(y_test, y_proba_best, 0.00, COST, GAIN)
        s_std = compute_strategy(y_test, y_proba_best, 0.50, COST, GAIN)
        s_opt_profit = compute_strategy(y_test, y_proba_best, best_profit_thr, COST, GAIN)
        s_opt_roi = compute_strategy(y_test, y_proba_best, best_roi_thr, COST, GAIN)

        strat_df = pd.DataFrame([
            {"Stratégie": "Tous les clients", **s_all},
            {"Stratégie": "Seuil standard 0.50", **s_std},
            {"Stratégie": "Seuil optimisé profit", **s_opt_profit},
            {"Stratégie": "Seuil optimisé ROI", **s_opt_roi},
        ])

        strat_df = strat_df[[
            "Stratégie", "Seuil", "Clients contactés", "Répondeurs captés",
            "Faux positifs", "Faux négatifs", "Taux de conversion (%)",
            "Taux de capture (%)", "Coût total (€)", "Profit estimé (€)", "ROI (%)"
        ]]

        st.dataframe(strat_df, use_container_width=True, hide_index=True)

        st.info(
            f"**Recommandation profit :** cibler les clients avec une probabilité ≥ "
            f"**{best_profit_thr:.2f}** "
            f"→ profit estimé **{s_opt_profit['Profit estimé (€)']:,.0f} €** · "
            f"ROI **{s_opt_profit['ROI (%)']:.2f}%** · "
            f"capture **{s_opt_profit['Taux de capture (%)']:.2f}%** des répondeurs."
        )

        # Courbes
        p_vals = thr_df["Profit estimé (€)"].tolist()
        r_vals = thr_df["ROI (%)"].tolist()
        n_vals = thr_df["Clients contactés"].tolist()

        fig_th, axs = plt.subplots(1, 3, figsize=(16, 5))

        # Profit
        axs[0].plot(thr_r, p_vals, color="#6f4b8b", linewidth=2)
        axs[0].axvline(best_profit_thr, color="red", linestyle="--",
                        label=f"Seuil profit max = {best_profit_thr:.2f}")
        axs[0].axhline(0, color="gray", linestyle=":", alpha=0.6)
        axs[0].set_title("Profit net estimé", fontsize=12, pad=10)
        axs[0].set_xlabel("Seuil de décision")
        axs[0].set_ylabel("Profit (€)")
        axs[0].legend(fontsize=9)
        axs[0].grid(alpha=0.25)

        # ROI
        axs[1].plot(thr_r, r_vals, color="#d946af", linewidth=2)
        axs[1].axvline(best_roi_thr, color="red", linestyle="--",
                        label=f"Seuil ROI max = {best_roi_thr:.2f}")
        axs[1].axhline(0, color="gray", linestyle=":", alpha=0.6)
        axs[1].set_title("ROI (%) par seuil", fontsize=12, pad=10)
        axs[1].set_xlabel("Seuil de décision")
        axs[1].set_ylabel("ROI (%)")
        axs[1].legend(fontsize=9)
        axs[1].grid(alpha=0.25)

        # Volume contacté
        axs[2].plot(thr_r, n_vals, color="#4b8b4b", linewidth=2)
        axs[2].axvline(best_profit_thr, color="red", linestyle="--",
                        label="Seuil profit max")
        axs[2].axvline(best_roi_thr, color="orange", linestyle="--",
                        label="Seuil ROI max")
        axs[2].set_title("Volume de clients contactés", fontsize=12, pad=10)
        axs[2].set_xlabel("Seuil de décision")
        axs[2].set_ylabel("Nombre de clients")
        axs[2].legend(fontsize=9)
        axs[2].grid(alpha=0.25)

        plt.suptitle("Optimisation de la stratégie de ciblage",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig_th, use_container_width=True)
        plt.close(fig_th)

        st.markdown("#### Top 10 seuils selon le profit")
        st.dataframe(
            thr_df.sort_values("Profit estimé (€)", ascending=False).head(10),
            use_container_width=True, hide_index=True
        )

    # ──────────────────────────────────────────────────────────────────────
    # TAB 5 : PRÉDICTION CLIENT
    # ──────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Prédiction pour un client")
        st.caption(f"Modèle utilisé : **{best_model_name}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Profil sociodémographique**")
            in_age = st.slider("Âge", 18, 95, 35)
            in_job = st.selectbox("Profession", sorted(df["job"].unique()))
            in_marital = st.selectbox("Statut marital",
                                       sorted(df["marital"].unique()))
            in_education = st.selectbox("Éducation",
                                         sorted(df["education"].unique()))
        with c2:
            st.markdown("**💰 Profil bancaire**")
            in_balance = st.number_input("Solde (€)", -10000, 100000, 1500, step=100)
            in_default = st.radio("Crédit en défaut ?", ["no", "yes"], horizontal=True)
            in_housing = st.radio("Prêt immobilier ?", ["no", "yes"], horizontal=True)
            in_loan = st.radio("Prêt personnel ?", ["no", "yes"], horizontal=True)
        with c3:
            st.markdown("**📞 Campagne**")
            in_contact = st.selectbox("Type de contact",
                                       sorted(df["contact"].unique()))
            in_month = st.selectbox("Mois", sorted(df["month"].unique()))
            in_day = st.slider("Jour", 1, 31, 15)
            in_campaign = st.number_input("Nb contacts campagne", 1, 50, 2)

        st.markdown("**📜 Historique**")
        c1, c2, c3 = st.columns(3)
        with c1:
            in_pdays = st.number_input("Jours depuis dernier contact",
                                        -1, 1000, -1)
        with c2:
            in_previous = st.number_input("Nb contacts précédents", 0, 50, 0)
        with c3:
            in_poutcome = st.selectbox("Résultat campagne précédente",
                                        sorted(df["poutcome"].unique()))

        if st.button("🚀 Lancer la prédiction"):
            client_input = pd.DataFrame([{
                "age": in_age, "balance": in_balance, "day": in_day,
                "campaign": in_campaign, "pdays": in_pdays,
                "previous": in_previous,
                "contacted_before": 1 if in_pdays != -1 else 0,
                "job": in_job, "marital": in_marital,
                "education": in_education, "default": in_default,
                "housing": in_housing, "loan": in_loan,
                "contact": in_contact, "month": in_month,
                "poutcome": in_poutcome,
            }])

            best_pipe = modeling["results"][best_model_name]["pipeline"]
            proba = best_pipe.predict_proba(client_input)[0, 1]
            pred = int(proba >= 0.5)

            c1, c2 = st.columns([1, 2])
            with c1:
                if pred == 1:
                    st.markdown(
                        f'<div class="result-box result-target">'
                        f'<b>✅ CLIENT À CIBLER</b><br>'
                        f'Probabilité de souscription :<br>'
                        f'<span class="prob-value">{proba*100:.1f}%</span>'
                        f'</div>', unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-box result-notarget">'
                        f'<b>❌ Client peu prometteur</b><br>'
                        f'Probabilité de souscription :<br>'
                        f'<span class="prob-value">{proba*100:.1f}%</span>'
                        f'</div>', unsafe_allow_html=True
                    )

            with c2:
                fig_g, ax_g = _fig(8, 4)
                ax_g.barh(["Probabilité"], [proba*100],
                          color="#27AE60" if pred == 1 else "#E74C3C")
                ax_g.barh(["Probabilité"], [100 - proba*100],
                          left=[proba*100], color="#dce8f5")
                ax_g.axvline(50, color="black", linestyle="--", alpha=0.5,
                              label="Seuil 50%")
                ax_g.set_xlim(0, 100)
                ax_g.set_xlabel("Probabilité (%)")
                ax_g.set_title("Score de propension", fontsize=12, pad=10)
                ax_g.legend()
                plt.tight_layout()
                st.pyplot(fig_g, use_container_width=True)
                plt.close(fig_g)


if __name__ == "__main__":
    main()
