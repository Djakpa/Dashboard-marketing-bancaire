"""
═══════════════════════════════════════════════════════════════════════════
🔧 SCRIPT DE PRÉ-AGRÉGATION (à lancer UNE SEULE FOIS en local)
═══════════════════════════════════════════════════════════════════════════
Transforme le CSV brut de 330 Mo en 3 fichiers Parquet optimisés (~5-10 Mo)
qui seront ensuite chargés par le dashboard Streamlit.

Bénéfices :
- ✅ Compatible Streamlit Cloud (limite à 100 Mo par fichier)
- ✅ Démarrage du dashboard ultra-rapide (~1 seconde)
- ✅ Mémoire RAM réduite (factor 30-60x)
- ✅ Pratique professionnelle standard (ETL → données analytiques)
═══════════════════════════════════════════════════════════════════════════

UTILISATION :
1. Placez ce script dans le même dossier que `comptages-routiers-permanents.csv`
2. Lancez : python prepare_data.py
3. 3 fichiers .parquet seront créés (à uploader sur GitHub avec le dashboard)
"""

import os
import time
import pandas as pd
import numpy as np


# ─── CONFIGURATION ───
CSV_PATH = "comptages-routiers-permanents.csv"
CHUNK_SIZE = 100_000

# Sortie
OUT_AGG_HORAIRE = "agg_par_arc_heure.csv.gz"      # ~ 0.5 Mo
OUT_AGG_TRONCONS = "agg_par_arc.csv.gz"            # ~ 0.3 Mo
OUT_AGG_TEMPOREL = "agg_temporel.csv.gz"           # ~ 0.1 Mo
OUT_SAMPLE_DIAG = "sample_diagramme.csv.gz"        # ~ 0.5 Mo


def main():
    print("=" * 70)
    print("🔧 PRÉ-AGRÉGATION DES DONNÉES TRAFIC PARISIEN")
    print("=" * 70)

    if not os.path.exists(CSV_PATH):
        print(f"❌ Fichier introuvable : {CSV_PATH}")
        print(f"   Placez le CSV dans le même dossier que ce script.")
        return

    # ──────────────────────────────────────────────────────────────────
    # ÉTAPE 1 — Chargement par chunks
    # ──────────────────────────────────────────────────────────────────
    print(f"\n📂 Étape 1/4 — Chargement par chunks (CHUNK_SIZE={CHUNK_SIZE:,})")
    t0 = time.time()

    chunks = []
    total = 0
    for i, chunk in enumerate(pd.read_csv(CSV_PATH, sep=";",
                                           chunksize=CHUNK_SIZE,
                                           low_memory=False)):
        # On supprime les colonnes inutiles dès le chunk pour gagner en RAM
        chunk = chunk.drop(columns=[
            "Identifiant noeud amont", "Libelle noeud amont",
            "Identifiant noeud aval", "Libelle noeud aval",
            "Date debut dispo data", "Date fin dispo data",
            "geo_shape",  # trop volumineux pour notre besoin (on garde geo_point_2d)
        ], errors="ignore")
        chunks.append(chunk)
        total += len(chunk)
        print(f"   Chunk {i+1} : {len(chunk):,} lignes · cumul {total:,}")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"   ✅ Lu en {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────
    # ÉTAPE 2 — Nettoyage et formatage
    # ──────────────────────────────────────────────────────────────────
    print(f"\n🧹 Étape 2/4 — Nettoyage & formatage")
    t0 = time.time()

    # Doublons
    df = df.drop_duplicates(
        subset=["Identifiant arc", "Date et heure de comptage"], keep="first"
    )

    # Date
    df["Date et heure de comptage"] = pd.to_datetime(
        df["Date et heure de comptage"], errors="coerce", utc=True
    )

    # Tri pour interpolation
    df = df.sort_values(["Identifiant arc", "Date et heure de comptage"])

    # Imputation par interpolation linéaire au sein de chaque tronçon
    cols_imputer = ["Débit horaire", "Taux d'occupation"]
    df[cols_imputer] = (
        df.groupby("Identifiant arc")[cols_imputer]
          .transform(lambda s: s.interpolate(method="linear",
                                              limit_direction="both"))
    )
    # Filet de sécurité
    for col in cols_imputer:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Variables temporelles
    df["heure"] = df["Date et heure de comptage"].dt.hour
    df["num_jour_sem"] = df["Date et heure de comptage"].dt.dayofweek  # 0=lundi
    df["est_weekend"] = df["num_jour_sem"].isin([5, 6]).astype(int)
    df["mois"] = df["Date et heure de comptage"].dt.month

    print(f"   ✅ Nettoyé en {time.time()-t0:.1f}s")
    print(f"   📊 Lignes nettes : {len(df):,}")

    # ──────────────────────────────────────────────────────────────────
    # ÉTAPE 3 — Parser les coordonnées géographiques
    # ──────────────────────────────────────────────────────────────────
    print(f"\n📍 Étape 3/4 — Coordonnées géographiques")

    def parse_geo(s):
        try:
            if pd.isna(s):
                return (None, None)
            parts = str(s).split(",")
            return (float(parts[0].strip()), float(parts[1].strip()))
        except Exception:
            return (None, None)

    geo_unique = df.drop_duplicates(subset="Identifiant arc")[
        ["Identifiant arc", "Libelle", "geo_point_2d"]
    ].copy()
    geo_unique["coords"] = geo_unique["geo_point_2d"].apply(parse_geo)
    geo_unique["lat"] = geo_unique["coords"].apply(lambda x: x[0])
    geo_unique["lon"] = geo_unique["coords"].apply(lambda x: x[1])
    geo_unique = geo_unique.drop(columns=["coords", "geo_point_2d"])

    # ──────────────────────────────────────────────────────────────────
    # ÉTAPE 4 — Génération des 3 fichiers d'agrégation
    # ──────────────────────────────────────────────────────────────────
    print(f"\n💾 Étape 4/4 — Création des fichiers Parquet")

    # ─── 4.1 — Profil HORAIRE × JOUR_SEMAINE par tronçon (pour heatmaps + filtres) ───
    print(f"   ⚙️  Agrégation horaire × jour de semaine...")
    agg_h = (
        df.groupby(["Identifiant arc", "heure", "num_jour_sem", "est_weekend"])
          .agg(
              debit_moyen=("Débit horaire", "mean"),
              occupation_moyenne=("Taux d'occupation", "mean"),
              nb_mesures=("Débit horaire", "count"),
          )
          .reset_index()
          .round(2)
    )
    # Joindre les libellés et coords
    agg_h = agg_h.merge(geo_unique, on="Identifiant arc", how="left")
    agg_h.to_csv(OUT_AGG_HORAIRE, index=False, compression="gzip")
    print(f"   ✅ {OUT_AGG_HORAIRE} : "
          f"{os.path.getsize(OUT_AGG_HORAIRE)/1024/1024:.2f} Mo "
          f"({len(agg_h):,} lignes)")

    # ─── 4.2 — Stats globales par tronçon (pour cartes + classements) ───
    print(f"   ⚙️  Agrégation par tronçon...")
    agg_t = (
        df.groupby(["Identifiant arc", "Libelle"])
          .agg(
              debit_moyen=("Débit horaire", "mean"),
              debit_median=("Débit horaire", "median"),
              debit_max=("Débit horaire", "max"),
              debit_min=("Débit horaire", "min"),
              debit_ecart_type=("Débit horaire", "std"),
              occupation_moyenne=("Taux d'occupation", "mean"),
              occupation_max=("Taux d'occupation", "max"),
              nb_mesures=("Débit horaire", "count"),
          )
          .reset_index()
          .round(2)
    )
    # Mode de l'état du trafic dominant
    etat_dominant = (
        df.groupby("Identifiant arc")["Etat trafic"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "Inconnu")
          .reset_index()
          .rename(columns={"Etat trafic": "etat_dominant"})
    )
    agg_t = agg_t.merge(etat_dominant, on="Identifiant arc", how="left")
    agg_t = agg_t.merge(
        geo_unique.drop(columns=["Libelle"]),
        on="Identifiant arc", how="left"
    )
    agg_t.to_csv(OUT_AGG_TRONCONS, index=False, compression="gzip")
    print(f"   ✅ {OUT_AGG_TRONCONS} : "
          f"{os.path.getsize(OUT_AGG_TRONCONS)/1024/1024:.2f} Mo "
          f"({len(agg_t):,} lignes)")

    # ─── 4.3 — Profil temporel global (sans tronçon : agrégation totale) ───
    # Pour les graphiques globaux : line plot 24h, semaine vs week-end
    print(f"   ⚙️  Agrégation temporelle globale...")
    agg_tg = (
        df.groupby(["heure", "num_jour_sem", "est_weekend", "mois"])
          .agg(
              debit_moyen=("Débit horaire", "mean"),
              occupation_moyenne=("Taux d'occupation", "mean"),
              debit_total=("Débit horaire", "sum"),
              nb_mesures=("Débit horaire", "count"),
          )
          .reset_index()
          .round(2)
    )
    # Échantillon brut pour le diagramme fondamental (50k points c'est largement assez)
    sample_diagrame = df[["Débit horaire", "Taux d'occupation",
                            "Etat trafic"]].sample(
        n=min(50_000, len(df)), random_state=42
    ).round(2)
    sample_diagrame.to_csv(OUT_SAMPLE_DIAG, index=False, compression="gzip")
    agg_tg.to_csv(OUT_AGG_TEMPOREL, index=False, compression="gzip")
    print(f"   ✅ {OUT_AGG_TEMPOREL} : "
          f"{os.path.getsize(OUT_AGG_TEMPOREL)/1024/1024:.2f} Mo "
          f"({len(agg_tg):,} lignes)")
    print(f"   ✅ {OUT_SAMPLE_DIAG} : "
          f"{os.path.getsize(OUT_SAMPLE_DIAG)/1024/1024:.2f} Mo")

    # ──────────────────────────────────────────────────────────────────
    # RÉCAP
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"🎉 PRÉ-AGRÉGATION TERMINÉE !")
    print(f"{'=' * 70}")
    total_size = sum(os.path.getsize(f) for f in [
        OUT_AGG_HORAIRE, OUT_AGG_TRONCONS, OUT_AGG_TEMPOREL,
        OUT_SAMPLE_DIAG
    ]) / 1024 / 1024
    print(f"📦 Total des fichiers          : {total_size:.2f} Mo")
    print(f"📦 Fichier CSV original       : "
          f"{os.path.getsize(CSV_PATH)/1024/1024:.0f} Mo")
    print(f"⚡ Réduction                  : "
          f"{os.path.getsize(CSV_PATH)/(total_size*1024*1024):.0f}x plus léger")
    print(f"\n✅ Fichiers à uploader sur GitHub avec le dashboard :")
    for f in [OUT_AGG_HORAIRE, OUT_AGG_TRONCONS, OUT_AGG_TEMPOREL,
              OUT_SAMPLE_DIAG]:
        print(f"   - {f}")


if __name__ == "__main__":
    main()
