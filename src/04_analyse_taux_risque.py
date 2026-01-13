#!/usr/bin/env python3
"""
Analyse et prédiction des taux de risque d'accidents vélo en Île-de-France.

Ce script:
1. Compare deux taux de risque: par km d'aménagement et pour 10 000 habitants
2. Applique des modèles de Machine Learning pour prédire ces taux
3. Évalue les performances de chaque approche
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'resultats')
os.makedirs(RESULTS_DIR, exist_ok=True)

def charger_donnees():
    """Charge le dataset final."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'dataset_final_idf.csv'))
    print(f"Dataset chargé: {len(df)} communes, {len(df.columns)} colonnes")
    return df

def analyser_distribution(df, colonne, titre):
    """Analyse la distribution d'une variable cible."""
    print(f"\n{'='*60}")
    print(f"ANALYSE: {titre}")
    print('='*60)
    
    data = df[colonne]
    data_non_zero = data[data > 0]
    
    print(f"\nStatistiques descriptives:")
    print(f"   - Count: {len(data)}")
    print(f"   - Non-zéros: {len(data_non_zero)} ({100*len(data_non_zero)/len(data):.1f}%)")
    print(f"   - Min: {data.min():.4f}")
    print(f"   - Max: {data.max():.4f}")
    print(f"   - Moyenne: {data.mean():.4f}")
    print(f"   - Médiane: {data.median():.4f}")
    print(f"   - Écart-type: {data.std():.4f}")
    
    # Skewness et Kurtosis
    skew = stats.skew(data_non_zero) if len(data_non_zero) > 0 else 0
    kurt = stats.kurtosis(data_non_zero) if len(data_non_zero) > 0 else 0
    print(f"   - Skewness: {skew:.2f}")
    print(f"   - Kurtosis: {kurt:.2f}")
    
    return {
        'skewness': skew,
        'kurtosis': kurt,
        'cv': data.std()/data.mean() if data.mean() > 0 else 0,
        'pct_zeros': 100*(1 - len(data_non_zero)/len(data))
    }

def comparer_distributions(df):
    """Compare les distributions des deux taux de risque."""
    print("\n" + "="*80)
    print("ANALYSE DES DISTRIBUTIONS DES TAUX DE RISQUE")
    print("="*80)
    
    # Analyser chaque taux
    stats_km = analyser_distribution(df, 'taux_risque_par_km', 
                                      'TAUX DE RISQUE PAR KM D\'AMÉNAGEMENT')
    stats_hab = analyser_distribution(df, 'taux_risque_par_habitant', 
                                       'TAUX DE RISQUE POUR 10 000 HABITANTS')
    
    # Graphique comparatif
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogrammes
    for ax, col, titre in zip(axes[0], 
                               ['taux_risque_par_km', 'taux_risque_par_habitant'],
                               ['Taux/km', 'Taux/10k hab']):
        data = df[col]
        ax.hist(data[data > 0], bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution: {titre}')
        ax.set_xlabel(col)
        ax.set_ylabel('Fréquence')
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Moyenne: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', label=f'Médiane: {data.median():.2f}')
        ax.legend(fontsize=8)
    
    # Log-histogrammes
    for ax, col, titre in zip(axes[1], 
                               ['taux_risque_par_km', 'taux_risque_par_habitant'],
                               ['Taux/km (log)', 'Taux/10k hab (log)']):
        data = df[col]
        data_log = np.log1p(data[data > 0])
        ax.hist(data_log, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax.set_title(f'Distribution log: {titre}')
        ax.set_xlabel(f'log(1 + {col})')
        ax.set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'distributions_taux_risque.png'), dpi=150)
    plt.close()
    
    return stats_km, stats_hab

def analyser_correlations(df):
    """Analyse les corrélations avec les features."""
    print("\n\n" + "="*80)
    print("ANALYSE DES CORRÉLATIONS")
    print("="*80)
    
    # Features à analyser
    features = ['nb_amenagements', 'longueur_totale_amenagements', 'population', 
                'nb_pistes_cyclables', 'comptage_total_commune', 'densite_pop_amenagement']
    
    targets = ['taux_risque_par_km', 'taux_risque_par_habitant']
    
    # Calculer les corrélations
    print("\nCorrélations de Pearson:")
    print(f"\n{'Feature':<35} {'Par km':<15} {'Par habitant':<15}")
    print("-"*65)
    
    for feat in features:
        if feat in df.columns:
            corrs = []
            for target in targets:
                corr = df[feat].corr(df[target])
                corrs.append(corr)
            print(f"{feat:<35} {corrs[0]:<15.3f} {corrs[1]:<15.3f}")
    
    # Graphique de corrélations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, target, titre in zip(axes, targets, ['Taux/km', 'Taux/10k hab']):
        corrs = df[features].corrwith(df[target])
        colors = ['green' if c > 0 else 'red' for c in corrs]
        ax.barh(features, corrs, color=colors, edgecolor='black')
        ax.set_xlabel('Corrélation')
        ax.set_title(f'Corrélations avec {titre}')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlations_taux.png'), dpi=150)
    plt.close()

def preparer_features(df):
    """Prépare les features pour le machine learning."""
    
    # Features numériques à utiliser
    feature_cols = [
        'nb_amenagements', 'longueur_totale_amenagements', 'longueur_moyenne_amenagement',
        'nb_voies_principales', 'nb_voies_residentielles', 'nb_pistes_cyclables',
        'nb_double_sens', 'nb_sens_unique', 'nb_asphalt', 'nb_autres_revetements',
        'ratio_pistes_cyclables', 'ratio_double_sens',
        'nb_compteurs', 'comptage_total_commune', 'comptage_moyen_commune',
        'population', 'densite_pop_amenagement'
    ]
    
    # Filtrer les colonnes disponibles
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    
    # Remplacer les valeurs manquantes par 0
    X = X.fillna(0)
    
    # Remplacer les infinis par des grandes valeurs
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, available_features

def entrainer_modeles(X_train, X_test, y_train, y_test, nom_cible):
    """Entraîne plusieurs modèles et compare leurs performances."""
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèles à tester
    modeles = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    }
    
    resultats = []
    meilleur_modele = None
    meilleur_r2 = -np.inf
    
    for nom, modele in modeles.items():
        # Utiliser les données normalisées pour les modèles linéaires
        if nom in ['Ridge', 'Lasso', 'ElasticNet']:
            modele.fit(X_train_scaled, y_train)
            y_pred = modele.predict(X_test_scaled)
        else:
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_test)
        
        # Métriques
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (en évitant division par 0)
        mask = y_test > 0.01
        if mask.sum() > 0:
            mape = 100 * np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))
        else:
            mape = np.nan
        
        resultats.append({
            'Modèle': nom,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE (%)': mape
        })
        
        if r2 > meilleur_r2:
            meilleur_r2 = r2
            meilleur_modele = (nom, modele, y_pred)
    
    return pd.DataFrame(resultats), meilleur_modele

def predire_taux_risque(df, cible, nom_cible):
    """Pipeline complet de prédiction pour un taux de risque."""
    
    print(f"\n{'='*80}")
    print(f"PRÉDICTION: {nom_cible}")
    print('='*80)
    
    # Préparer les données
    X, feature_names = preparer_features(df)
    y = df[cible].copy()
    
    # Transformation log pour réduire l'asymétrie
    y_log = np.log1p(y)
    
    print(f"\nFeatures utilisées ({len(feature_names)}):")
    for f in feature_names:
        print(f"   - {f}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    print(f"\nDonnées: {len(X_train)} train, {len(X_test)} test")
    
    # Entraîner les modèles
    resultats, (meilleur_nom, meilleur_modele, y_pred_log) = entrainer_modeles(
        X_train, X_test, y_train, y_test, nom_cible
    )
    
    # Afficher les résultats
    print(f"\n--- Résultats (cible transformée en log) ---")
    resultats_sorted = resultats.sort_values('R²', ascending=False)
    print(resultats_sorted.to_string(index=False))
    
    # Transformer les prédictions en valeurs originales
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_log)
    
    # Métriques sur valeurs originales
    r2_orig = r2_score(y_test_orig, y_pred_orig)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    
    print(f"\n--- Meilleur modèle: {meilleur_nom} ---")
    print(f"   R² (valeurs originales): {r2_orig:.4f}")
    print(f"   MAE (valeurs originales): {mae_orig:.4f}")
    
    # Importance des features (si modèle arbre)
    if hasattr(meilleur_modele, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': meilleur_modele.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n--- Importance des features ---")
        print(importances.head(10).to_string(index=False))
    
    return resultats, meilleur_nom, y_test_orig, y_pred_orig

def visualiser_predictions(y_test_km, y_pred_km, y_test_hab, y_pred_hab, 
                           meilleur_km, meilleur_hab):
    """Visualise les prédictions vs valeurs réelles."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Taux par km
    ax = axes[0, 0]
    ax.scatter(y_test_km, y_pred_km, alpha=0.5, s=20)
    max_val = max(y_test_km.max(), y_pred_km.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Prédiction parfaite')
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title(f'Taux/km - {meilleur_km}')
    ax.legend()
    
    # Taux par km (zoom)
    ax = axes[0, 1]
    mask = y_test_km < np.percentile(y_test_km, 95)
    ax.scatter(y_test_km[mask], y_pred_km[mask], alpha=0.5, s=20)
    max_val = np.percentile(y_test_km, 95)
    ax.plot([0, max_val], [0, max_val], 'r--', label='Prédiction parfaite')
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title(f'Taux/km (zoom P95) - {meilleur_km}')
    ax.legend()
    
    # Taux par habitant
    ax = axes[1, 0]
    ax.scatter(y_test_hab, y_pred_hab, alpha=0.5, s=20, color='green')
    max_val = max(y_test_hab.max(), y_pred_hab.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Prédiction parfaite')
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title(f'Taux/10k hab - {meilleur_hab}')
    ax.legend()
    
    # Taux par habitant (zoom)
    ax = axes[1, 1]
    mask = y_test_hab < np.percentile(y_test_hab, 95)
    ax.scatter(y_test_hab[mask], y_pred_hab[mask], alpha=0.5, s=20, color='green')
    max_val = np.percentile(y_test_hab, 95)
    ax.plot([0, max_val], [0, max_val], 'r--', label='Prédiction parfaite')
    ax.set_xlabel('Valeurs réelles')
    ax.set_ylabel('Prédictions')
    ax.set_title(f'Taux/10k hab (zoom P95) - {meilleur_hab}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'predictions_taux_risque.png'), dpi=150)
    plt.close()
    
    print(f"\n✅ Graphique sauvegardé: predictions_taux_risque.png")

def comparer_resultats(resultats_km, resultats_hab):
    """Compare les performances sur les deux cibles."""
    
    print("\n" + "="*80)
    print("COMPARAISON DES RÉSULTATS")
    print("="*80)
    
    # Meilleur modèle pour chaque cible
    best_km = resultats_km.loc[resultats_km['R²'].idxmax()]
    best_hab = resultats_hab.loc[resultats_hab['R²'].idxmax()]
    
    print(f"\n{'Métrique':<20} {'Taux/km':<25} {'Taux/10k hab':<25}")
    print("-"*70)
    print(f"{'Meilleur modèle':<20} {best_km['Modèle']:<25} {best_hab['Modèle']:<25}")
    print(f"{'R²':<20} {best_km['R²']:<25.4f} {best_hab['R²']:<25.4f}")
    print(f"{'RMSE':<20} {best_km['RMSE']:<25.4f} {best_hab['RMSE']:<25.4f}")
    print(f"{'MAE':<20} {best_km['MAE']:<25.4f} {best_hab['MAE']:<25.4f}")
    
    # Graphique comparatif
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modeles = resultats_km['Modèle'].tolist()
    x = np.arange(len(modeles))
    width = 0.35
    
    r2_km = resultats_km['R²'].tolist()
    r2_hab = resultats_hab['R²'].tolist()
    
    bars1 = ax.bar(x - width/2, r2_km, width, label='Taux/km', color='steelblue')
    bars2 = ax.bar(x + width/2, r2_hab, width, label='Taux/10k hab', color='forestgreen')
    
    ax.set_xlabel('Modèle')
    ax.set_ylabel('R²')
    ax.set_title('Comparaison des performances R² par modèle')
    ax.set_xticks(x)
    ax.set_xticklabels(modeles, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparaison_modeles_taux.png'), dpi=150)
    plt.close()

def main():
    """Fonction principale."""
    print("="*80)
    print("ANALYSE ET PRÉDICTION DES TAUX DE RISQUE")
    print("="*80)
    
    # Charger les données
    df = charger_donnees()
    
    # Filtrer les communes avec données complètes pour l'analyse
    df_complet = df[(df['population'] > 0) & (df['longueur_totale_amenagements'] > 0)].copy()
    print(f"\nCommunes avec population ET aménagements: {len(df_complet)}")
    
    # 1. Analyse des distributions
    comparer_distributions(df_complet)
    
    # 2. Analyse des corrélations
    analyser_correlations(df_complet)
    
    # 3. Prédiction du taux de risque par km
    resultats_km, meilleur_km, y_test_km, y_pred_km = predire_taux_risque(
        df_complet, 'taux_risque_par_km', 'TAUX DE RISQUE PAR KM D\'AMÉNAGEMENT'
    )
    
    # 4. Prédiction du taux de risque par habitant
    resultats_hab, meilleur_hab, y_test_hab, y_pred_hab = predire_taux_risque(
        df_complet, 'taux_risque_par_habitant', 'TAUX DE RISQUE POUR 10 000 HABITANTS'
    )
    
    # 5. Visualisation des prédictions
    visualiser_predictions(y_test_km, y_pred_km, y_test_hab, y_pred_hab,
                           meilleur_km, meilleur_hab)
    
    # 6. Comparaison finale
    comparer_resultats(resultats_km, resultats_hab)
    
    print("\n\n✅ Analyse terminée!")
    print(f"   Graphiques sauvegardés dans: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
