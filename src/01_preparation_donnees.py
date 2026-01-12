#!/usr/bin/env python3
"""
Script de préparation et fusion des données pour l'analyse des accidents de vélo en Île-de-France.

Ce script:
1. Charge les 3 datasets (accidents, aménagements, comptages)
2. Filtre les données pour l'Île-de-France
3. Agrège les données par commune (code INSEE)
4. Crée des features supplémentaires
5. Fusionne le tout dans un dataset final
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Chemins des fichiers
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ACCIDENTS_FILE = os.path.join(DATA_DIR, 'accidentsVelo.csv')
AMENAGEMENTS_FILE = os.path.join(DATA_DIR, 'amenagements-velo-en-ile-de-france.csv')
COMPTAGES_FILE = os.path.join(DATA_DIR, 'comptage-velo-donnees-compteurs.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'dataset_final_idf.csv')

# Départements Île-de-France
IDF_DEPS = ['75', '77', '78', '91', '92', '93', '94', '95']

def charger_accidents():
    """Charge et filtre les données d'accidents pour l'IDF."""
    print("=" * 60)
    print("Chargement des données d'accidents...")
    
    df = pd.read_csv(ACCIDENTS_FILE, dtype=str)
    print(f"  Total accidents France: {len(df)}")
    
    # Filtrer IDF
    df = df[df['dep'].isin(IDF_DEPS)].copy()
    print(f"  Accidents IDF: {len(df)}")
    
    # Créer le code INSEE complet (dép + com)
    # Le champ 'com' contient déjà le code commune complet avec le département
    df['code_insee'] = df['com'].str.zfill(5)
    
    # Convertir les colonnes numériques
    numeric_cols = ['lat', 'long', 'grav', 'age', 'lum', 'atm', 'surf']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convertir la date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['annee'] = df['date'].dt.year
    
    return df

def agreger_accidents_par_commune(df_accidents):
    """Agrège les accidents par commune."""
    print("\nAgrégation des accidents par commune...")
    
    agg = df_accidents.groupby('code_insee').agg(
        nb_accidents=('Num_Acc', 'count'),
        nb_accidents_graves=('grav', lambda x: (x <= 2).sum()),  # 1=tué, 2=hospitalisé
        nb_accidents_mortels=('grav', lambda x: (x == 1).sum()),
        gravite_moyenne=('grav', 'mean'),
        age_moyen_victimes=('age', 'mean'),
        lat_moyenne=('lat', 'mean'),
        long_moyenne=('long', 'mean'),
        # Conditions d'accidents
        nb_accidents_nuit=('lum', lambda x: (x.isin([2, 3, 4, 5])).sum()),  # Crépuscule ou nuit
        nb_accidents_pluie=('atm', lambda x: (x.isin([2, 3, 4, 5, 6, 7])).sum()),  # Mauvais temps
        nb_accidents_mouille=('surf', lambda x: (x == 2).sum()),  # Surface mouillée
        # Période
        annee_min=('annee', 'min'),
        annee_max=('annee', 'max')
    ).reset_index()
    
    # Calculer le taux d'accidents graves
    agg['taux_accidents_graves'] = agg['nb_accidents_graves'] / agg['nb_accidents']
    agg['taux_accidents_mortels'] = agg['nb_accidents_mortels'] / agg['nb_accidents']
    
    print(f"  Communes avec accidents: {len(agg)}")
    return agg

def charger_amenagements():
    """Charge et agrège les aménagements vélo par commune."""
    print("\n" + "=" * 60)
    print("Chargement des données d'aménagements vélo...")
    
    df = pd.read_csv(AMENAGEMENTS_FILE, sep=';', dtype=str)
    print(f"  Total aménagements: {len(df)}")
    
    # Code INSEE
    df['code_insee'] = df['insee_com'].str.zfill(5)
    
    # Convertir longueur
    df['longueur'] = pd.to_numeric(df['longueur'], errors='coerce')
    
    # Agrégation par commune
    print("\nAgrégation des aménagements par commune...")
    agg = df.groupby('code_insee').agg(
        nom_commune=('nom_com', 'first'),
        nb_amenagements=('osm_id', 'count'),
        longueur_totale_amenagements=('longueur', 'sum'),
        longueur_moyenne_amenagement=('longueur', 'mean'),
        # Types de voies
        nb_voies_principales=('highway', lambda x: x.isin(['primary', 'secondary', 'tertiary']).sum()),
        nb_voies_residentielles=('highway', lambda x: (x == 'residential').sum()),
        nb_pistes_cyclables=('highway', lambda x: (x == 'cycleway').sum()),
        # Types d'aménagements
        nb_double_sens=('sens_voit', lambda x: (x == 'DOUBLE').sum()),
        nb_sens_unique=('sens_voit', lambda x: (x == 'UNIQUE').sum()),
        # Revêtement
        nb_asphalt=('revetement', lambda x: (x == 'asphalt').sum()),
        nb_autres_revetements=('revetement', lambda x: (x != 'asphalt').sum()),
    ).reset_index()
    
    # Calculer des ratios
    agg['ratio_pistes_cyclables'] = agg['nb_pistes_cyclables'] / agg['nb_amenagements']
    agg['ratio_double_sens'] = agg['nb_double_sens'] / agg['nb_amenagements']
    
    print(f"  Communes avec aménagements: {len(agg)}")
    return agg

def charger_comptages():
    """Charge et agrège les comptages vélo."""
    print("\n" + "=" * 60)
    print("Chargement des données de comptage vélo...")
    
    # Le fichier comptage n'a pas de code INSEE directement
    # On va devoir utiliser les coordonnées géographiques pour faire un lien
    # Pour simplifier, on agrège par site de comptage et on extrait les coordonnées
    
    df = pd.read_csv(COMPTAGES_FILE, sep=';', dtype=str)
    print(f"  Total enregistrements comptage: {len(df)}")
    
    # Convertir le comptage horaire en numérique
    df['comptage'] = pd.to_numeric(df['Comptage horaire'], errors='coerce')
    
    # Extraire les coordonnées
    df['coords'] = df['Coordonnées géographiques']
    
    # Agrégation par site de comptage
    agg_sites = df.groupby('Identifiant du site de comptage').agg(
        nom_site=('Nom du site de comptage', 'first'),
        comptage_total=('comptage', 'sum'),
        comptage_moyen_horaire=('comptage', 'mean'),
        comptage_max_horaire=('comptage', 'max'),
        nb_mesures=('comptage', 'count'),
        coords=('coords', 'first')
    ).reset_index()
    
    # Extraire lat/long des coordonnées
    def parse_coords(coord_str):
        try:
            if pd.isna(coord_str):
                return None, None
            parts = coord_str.replace(' ', '').split(',')
            return float(parts[0]), float(parts[1])
        except:
            return None, None
    
    agg_sites[['lat_compteur', 'long_compteur']] = agg_sites['coords'].apply(
        lambda x: pd.Series(parse_coords(x))
    )
    
    print(f"  Sites de comptage: {len(agg_sites)}")
    
    return agg_sites

def associer_comptages_aux_communes(df_comptages, df_accidents_agg):
    """
    Associe les comptages aux communes les plus proches.
    Utilise les coordonnées des accidents pour faire le lien.
    """
    print("\nAssociation des comptages aux communes...")
    
    # Pour chaque compteur, trouver la commune la plus proche
    from scipy.spatial.distance import cdist
    
    # Filtrer les données avec coordonnées valides
    compteurs_valid = df_comptages[df_comptages['lat_compteur'].notna()].copy()
    communes_valid = df_accidents_agg[df_accidents_agg['lat_moyenne'].notna()].copy()
    
    if len(compteurs_valid) == 0 or len(communes_valid) == 0:
        print("  Pas assez de coordonnées valides pour l'association")
        return pd.DataFrame(columns=['code_insee', 'nb_compteurs', 'comptage_total_commune', 
                                     'comptage_moyen_commune'])
    
    # Calculer les distances
    coords_compteurs = compteurs_valid[['lat_compteur', 'long_compteur']].values
    coords_communes = communes_valid[['lat_moyenne', 'long_moyenne']].values
    
    distances = cdist(coords_compteurs, coords_communes, metric='euclidean')
    
    # Assigner chaque compteur à la commune la plus proche
    compteurs_valid['idx_commune'] = distances.argmin(axis=1)
    compteurs_valid['code_insee'] = compteurs_valid['idx_commune'].apply(
        lambda i: communes_valid.iloc[i]['code_insee']
    )
    
    # Agrégation par commune
    agg_comptages = compteurs_valid.groupby('code_insee').agg(
        nb_compteurs=('Identifiant du site de comptage', 'count'),
        comptage_total_commune=('comptage_total', 'sum'),
        comptage_moyen_commune=('comptage_moyen_horaire', 'mean'),
        comptage_max_commune=('comptage_max_horaire', 'max')
    ).reset_index()
    
    print(f"  Communes avec compteurs associés: {len(agg_comptages)}")
    return agg_comptages

def creer_dataset_final():
    """Crée le dataset final en fusionnant toutes les sources."""
    print("\n" + "=" * 60)
    print("CRÉATION DU DATASET FINAL")
    print("=" * 60)
    
    # 1. Charger et agréger les accidents
    df_accidents = charger_accidents()
    df_accidents_agg = agreger_accidents_par_commune(df_accidents)
    
    # 2. Charger et agréger les aménagements
    df_amenagements = charger_amenagements()
    
    # 3. Charger et associer les comptages
    df_comptages = charger_comptages()
    df_comptages_communes = associer_comptages_aux_communes(df_comptages, df_accidents_agg)
    
    # 4. Fusion des datasets
    print("\n" + "=" * 60)
    print("Fusion des datasets...")
    
    # Fusion accidents + aménagements (left join pour garder toutes les communes avec accidents)
    df_final = pd.merge(
        df_accidents_agg,
        df_amenagements,
        on='code_insee',
        how='outer',
        indicator='_merge_amenagements'
    )
    
    # Fusion avec comptages
    df_final = pd.merge(
        df_final,
        df_comptages_communes,
        on='code_insee',
        how='left'
    )
    
    # Statistiques de fusion
    print(f"\n  Résultat fusion:")
    print(f"    - Communes avec accidents ET aménagements: {(df_final['_merge_amenagements'] == 'both').sum()}")
    print(f"    - Communes avec accidents uniquement: {(df_final['_merge_amenagements'] == 'left_only').sum()}")
    print(f"    - Communes avec aménagements uniquement: {(df_final['_merge_amenagements'] == 'right_only').sum()}")
    
    # 5. Traitement des valeurs manquantes
    print("\nTraitement des valeurs manquantes...")
    
    # Communes sans accidents: mettre 0
    accident_cols = ['nb_accidents', 'nb_accidents_graves', 'nb_accidents_mortels', 
                     'nb_accidents_nuit', 'nb_accidents_pluie', 'nb_accidents_mouille']
    for col in accident_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0).astype(int)
    
    # Ratios: mettre 0 si pas d'accidents
    ratio_cols = ['taux_accidents_graves', 'taux_accidents_mortels', 'gravite_moyenne']
    for col in ratio_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
    
    # Aménagements: mettre 0 si pas d'aménagements
    amenagement_cols = ['nb_amenagements', 'longueur_totale_amenagements', 'nb_pistes_cyclables',
                        'nb_voies_principales', 'nb_voies_residentielles', 'nb_double_sens', 
                        'nb_sens_unique', 'nb_asphalt', 'nb_autres_revetements']
    for col in amenagement_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
    
    # Comptages: mettre 0 si pas de compteurs
    comptage_cols = ['nb_compteurs', 'comptage_total_commune', 'comptage_moyen_commune', 
                     'comptage_max_commune']
    for col in comptage_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
    
    # 6. Créer des features supplémentaires
    print("\nCréation de features supplémentaires...")
    
    # Département à partir du code INSEE
    df_final['departement'] = df_final['code_insee'].str[:2]
    
    # Densité d'aménagements (aménagements par accident évité ?)
    df_final['ratio_amenagements_par_accident'] = np.where(
        df_final['nb_accidents'] > 0,
        df_final['nb_amenagements'] / df_final['nb_accidents'],
        df_final['nb_amenagements']
    )
    
    # Longueur moyenne par accident
    df_final['longueur_amenagement_par_accident'] = np.where(
        df_final['nb_accidents'] > 0,
        df_final['longueur_totale_amenagements'] / df_final['nb_accidents'],
        df_final['longueur_totale_amenagements']
    )
    
    # Indicateur de risque élevé (plus de X accidents)
    seuil_risque = df_final['nb_accidents'].quantile(0.75)
    df_final['risque_eleve'] = (df_final['nb_accidents'] >= seuil_risque).astype(int)
    print(f"    Seuil de risque élevé: >= {seuil_risque} accidents")
    
    # Catégorie de risque (pour analyse)
    df_final['categorie_risque'] = pd.cut(
        df_final['nb_accidents'],
        bins=[0, 1, 5, 20, 100, float('inf')],
        labels=['Très faible', 'Faible', 'Modéré', 'Élevé', 'Très élevé'],
        include_lowest=True
    )
    
    # Indicateur Paris
    df_final['est_paris'] = (df_final['departement'] == '75').astype(int)
    
    # Supprimer les colonnes temporaires
    df_final = df_final.drop(columns=['_merge_amenagements'], errors='ignore')
    
    # 7. Filtrer pour ne garder que les communes IDF avec données exploitables
    df_final = df_final[df_final['departement'].isin(IDF_DEPS)].copy()
    
    # 8. Sauvegarder
    print("\n" + "=" * 60)
    print(f"Sauvegarde du dataset final: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    # 9. Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU DATASET FINAL")
    print("=" * 60)
    print(f"  Nombre de communes: {len(df_final)}")
    print(f"  Nombre de colonnes: {len(df_final.columns)}")
    print(f"\n  Colonnes:")
    for col in df_final.columns:
        print(f"    - {col}")
    
    print(f"\n  Statistiques par département:")
    print(df_final.groupby('departement').agg({
        'code_insee': 'count',
        'nb_accidents': 'sum',
        'nb_amenagements': 'sum'
    }).rename(columns={'code_insee': 'nb_communes'}))
    
    print(f"\n  Distribution des catégories de risque:")
    print(df_final['categorie_risque'].value_counts().sort_index())
    
    return df_final

if __name__ == "__main__":
    df = creer_dataset_final()
    print("\n✅ Dataset final créé avec succès!")
