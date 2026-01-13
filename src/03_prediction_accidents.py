#!/usr/bin/env python3
"""
Script d'analyse ML pour la prédiction du nombre d'accidents de vélo par commune.

Question: Prédire le nombre d'accidents dans une commune en fonction du nombre d'aménagements.

Ce script compare plusieurs algorithmes de régression:
- Régression Linéaire
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              mean_absolute_percentage_error)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DATASET_FILE = os.path.join(DATA_DIR, 'dataset_final_idf.csv')

# Créer le dossier outputs s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def charger_donnees():
    """Charge et prépare les données pour la régression."""
    print("=" * 70)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 70)
    
    df = pd.read_csv(DATASET_FILE)
    print(f"Dataset chargé: {len(df)} communes")
    
    # Features pour la régression
    feature_cols = [
        'nb_amenagements',
        'longueur_totale_amenagements',
        'longueur_moyenne_amenagement',
        'nb_voies_principales',
        'nb_voies_residentielles',
        'nb_pistes_cyclables',
        'nb_double_sens',
        'nb_sens_unique',
        'nb_asphalt',
        'nb_autres_revetements',
        'ratio_pistes_cyclables',
        'ratio_double_sens',
        'nb_compteurs',
        'comptage_total_commune',
        'comptage_moyen_commune',
        'est_paris'
    ]
    
    # Vérifier que les colonnes existent
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"Features disponibles: {len(available_cols)}/{len(feature_cols)}")
    
    # Préparer X et y
    X = df[available_cols].copy()
    y = df['nb_accidents'].copy()
    
    # Remplacer les valeurs infinies et NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"\nStatistiques de la variable cible (nb_accidents):")
    print(y.describe())
    
    return X, y, available_cols, df

def preparer_donnees(X, y):
    """Prépare les données pour l'entraînement."""
    print("\n" + "=" * 70)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 70)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Ensemble d'entraînement: {len(X_train)} communes")
    print(f"Ensemble de test: {len(X_test)} communes")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def definir_modeles():
    """Définit les modèles de régression à comparer."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42, max_iter=10000),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=5
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, random_state=42, max_depth=5,
            learning_rate=0.1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100, random_state=42, max_depth=5,
            verbose=-1
        )
    }
    return models

def evaluer_modeles(models, X_train_scaled, X_test_scaled, y_train, y_test):
    """Évalue tous les modèles et retourne les résultats."""
    print("\n" + "=" * 70)
    print("ÉVALUATION DES MODÈLES")
    print("=" * 70)
    
    results = []
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"Entraînement: {name}")
        print(f"{'─' * 50}")
        
        # Entraînement
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Prédictions
        y_pred = model.predict(X_test_scaled)
        # S'assurer que les prédictions sont positives
        y_pred = np.maximum(y_pred, 0)
        predictions[name] = y_pred
        
        # Métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (attention aux valeurs nulles)
        mask = y_test > 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) * 100
        else:
            mape = np.nan
        
        # Validation croisée (R²)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                     scoring='r2')
        
        results.append({
            'Modèle': name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE (%)': mape,
            'CV R² (mean)': cv_scores.mean(),
            'CV R² (std)': cv_scores.std()
        })
        
        print(f"  RMSE:     {rmse:.2f}")
        print(f"  MAE:      {mae:.2f}")
        print(f"  R²:       {r2:.3f}")
        print(f"  MAPE:     {mape:.1f}%")
        print(f"  CV R²:    {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    return pd.DataFrame(results), trained_models, predictions

def afficher_tableau_comparatif(results_df):
    """Affiche un tableau comparatif des résultats."""
    print("\n" + "=" * 70)
    print("TABLEAU COMPARATIF DES MODÈLES")
    print("=" * 70)
    
    # Trier par R²
    results_sorted = results_df.sort_values('R²', ascending=False)
    
    print("\n" + results_sorted.to_string(index=False))
    
    # Sauvegarder
    results_sorted.to_csv(os.path.join(OUTPUT_DIR, 'regression_results.csv'), index=False)
    
    return results_sorted

def tracer_predictions_vs_reel(predictions, y_test, best_model_name):
    """Trace les prédictions vs valeurs réelles."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Sélectionner 4 modèles
    models_to_plot = ['Linear Regression', 'Random Forest', 'XGBoost', best_model_name]
    models_to_plot = [m for m in models_to_plot if m in predictions][:4]
    
    for ax, model_name in zip(axes.flatten(), models_to_plot):
        y_pred = predictions[model_name]
        
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Ligne parfaite
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
        
        # R² sur le graphique
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Valeurs réelles', fontsize=10)
        ax.set_ylabel('Prédictions', fontsize=10)
        ax.set_title(model_name, fontsize=12)
        ax.legend(loc='lower right')
    
    plt.suptitle('Prédictions vs Valeurs Réelles - Nombre d\'Accidents', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_vs_reel.png'), dpi=150)
    plt.close()
    print(f"\n✓ Graphique prédictions vs réel sauvegardé: {OUTPUT_DIR}/predictions_vs_reel.png")

def tracer_predictions_vs_reel_zoom(predictions, y_test, best_model_name):
    """Trace les prédictions vs valeurs réelles - version zoomée sur les petites valeurs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Sélectionner 4 modèles
    models_to_plot = ['Linear Regression', 'Random Forest', 'XGBoost', best_model_name]
    models_to_plot = [m for m in models_to_plot if m in predictions][:4]
    
    # Définir la limite de zoom (ex: 75e percentile des valeurs)
    zoom_limit = np.percentile(y_test, 90)
    zoom_limit = max(zoom_limit, 50)  # Au moins 50 accidents
    
    for ax, model_name in zip(axes.flatten(), models_to_plot):
        y_pred = predictions[model_name]
        
        # Filtrer pour le zoom
        mask = (y_test <= zoom_limit) & (y_pred <= zoom_limit)
        y_test_zoom = y_test[mask]
        y_pred_zoom = y_pred[mask]
        
        ax.scatter(y_test_zoom, y_pred_zoom, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Ligne parfaite
        ax.plot([0, zoom_limit], [0, zoom_limit], 'r--', linewidth=2, label='Prédiction parfaite')
        
        # R² sur les données zoomées
        if len(y_test_zoom) > 1:
            r2 = r2_score(y_test_zoom, y_pred_zoom)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}\n(n={len(y_test_zoom)})', transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Valeurs réelles', fontsize=10)
        ax.set_ylabel('Prédictions', fontsize=10)
        ax.set_title(model_name, fontsize=12)
        ax.set_xlim(0, zoom_limit)
        ax.set_ylim(0, zoom_limit)
        ax.legend(loc='lower right')
    
    plt.suptitle(f'Prédictions vs Valeurs Réelles - Zoom (≤ {int(zoom_limit)} accidents)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_vs_reel_zoom.png'), dpi=150)
    plt.close()
    print(f"✓ Graphique prédictions vs réel (zoom) sauvegardé: {OUTPUT_DIR}/predictions_vs_reel_zoom.png")

def tracer_residus(predictions, y_test, best_model_name):
    """Trace l'analyse des résidus pour le meilleur modèle."""
    y_pred = predictions[best_model_name]
    residus = y_test.values - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Distribution des résidus
    axes[0].hist(residus, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Résidu', fontsize=12)
    axes[0].set_ylabel('Fréquence', fontsize=12)
    axes[0].set_title('Distribution des Résidus', fontsize=12)
    
    # Résidus vs Prédictions
    axes[1].scatter(y_pred, residus, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prédictions', fontsize=12)
    axes[1].set_ylabel('Résidus', fontsize=12)
    axes[1].set_title('Résidus vs Prédictions', fontsize=12)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residus, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12)
    
    plt.suptitle(f'Analyse des Résidus - {best_model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_analysis.png'), dpi=150)
    plt.close()
    print(f"✓ Analyse des résidus sauvegardée: {OUTPUT_DIR}/residuals_analysis.png")

def tracer_importance_features(trained_models, feature_names):
    """Trace l'importance des features pour les modèles basés sur les arbres."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
    
    for ax, model_name in zip(axes.flatten(), tree_models):
        if model_name in trained_models:
            model = trained_models[model_name]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(indices)))
            ax.barh(range(len(indices)), importances[indices], color=colors)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name}')
            ax.invert_yaxis()
    
    plt.suptitle("Importance des Features par Modèle (Régression)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_regression.png'), dpi=150)
    plt.close()
    print(f"✓ Importance des features sauvegardée: {OUTPUT_DIR}/feature_importance_regression.png")

def tracer_comparaison_metriques(results_df):
    """Trace un graphique de comparaison des métriques."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² par modèle
    results_sorted = results_df.sort_values('R²', ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(results_sorted)))
    
    axes[0].barh(results_sorted['Modèle'], results_sorted['R²'], color=colors)
    axes[0].set_xlabel('R² Score', fontsize=12)
    axes[0].set_title('Coefficient de Détermination (R²)', fontsize=12)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # RMSE par modèle
    results_sorted_rmse = results_df.sort_values('RMSE', ascending=False)
    colors_rmse = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results_sorted_rmse)))
    
    axes[1].barh(results_sorted_rmse['Modèle'], results_sorted_rmse['RMSE'], color=colors_rmse)
    axes[1].set_xlabel('RMSE', fontsize=12)
    axes[1].set_title('Erreur Quadratique Moyenne (RMSE)', fontsize=12)
    
    plt.suptitle('Comparaison des Performances des Modèles', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison_regression.png'), dpi=150)
    plt.close()
    print(f"✓ Comparaison des métriques sauvegardée: {OUTPUT_DIR}/metrics_comparison_regression.png")

def tracer_correlation_features(X, y, feature_names):
    """Trace la corrélation entre les features et la variable cible."""
    # Calculer les corrélations
    correlations = []
    for col in feature_names:
        corr = X[col].corr(y)
        correlations.append({'Feature': col, 'Corrélation': corr})
    
    corr_df = pd.DataFrame(correlations).sort_values('Corrélation', ascending=True)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(corr_df)))
    colors = [plt.cm.RdYlGn(0.5 + c/2) for c in corr_df['Corrélation']]
    
    plt.barh(corr_df['Feature'], corr_df['Corrélation'], color=colors)
    plt.xlabel('Corrélation avec nb_accidents', fontsize=12)
    plt.title('Corrélation des Features avec le Nombre d\'Accidents', fontsize=14)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_features.png'), dpi=150)
    plt.close()
    print(f"✓ Corrélation des features sauvegardée: {OUTPUT_DIR}/correlation_features.png")

def generer_conclusion(results_df, best_model_name, X, y):
    """Génère une conclusion textuelle de l'analyse."""
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    best_row = results_df[results_df['Modèle'] == best_model_name].iloc[0]
    
    # Trouver la feature la plus corrélée
    correlations = {col: X[col].corr(y) for col in X.columns}
    top_feature = max(correlations, key=lambda k: abs(correlations[k]))
    top_corr = correlations[top_feature]
    
    conclusion = f"""
ANALYSE DE RÉGRESSION - PRÉDICTION DU NOMBRE D'ACCIDENTS DE VÉLO
═══════════════════════════════════════════════════════════════════════

OBJECTIF:
Prédire le nombre d'accidents de vélo par commune en Île-de-France
en fonction des aménagements cyclables disponibles.

MEILLEUR MODÈLE: {best_model_name}

PERFORMANCES:
  • RMSE:     {best_row['RMSE']:.2f} accidents
  • MAE:      {best_row['MAE']:.2f} accidents
  • R²:       {best_row['R²']:.3f}
  • CV R²:    {best_row['CV R² (mean)']:.3f} (+/- {best_row['CV R² (std)']:.3f})

FEATURE LA PLUS IMPORTANTE:
  • {top_feature} (corrélation = {top_corr:.3f})

INTERPRÉTATION:
- Le modèle explique {best_row['R²']*100:.1f}% de la variance du nombre d'accidents.
- L'erreur moyenne de prédiction est de {best_row['MAE']:.1f} accidents par commune.
- Les communes avec plus d'aménagements cyclables ont généralement 
  plus d'accidents enregistrés (effet de trafic cycliste plus important).

INSIGHTS CLÉS:
1. Le nombre d'aménagements cyclables est fortement corrélé au nombre 
   d'accidents, mais cela reflète aussi une utilisation plus intensive.
2. Les communes parisiennes ont un profil spécifique avec beaucoup 
   d'aménagements ET beaucoup d'accidents.
3. Pour réduire les accidents, il faut améliorer la qualité des 
   aménagements, pas seulement leur quantité.

RECOMMANDATIONS:
1. Privilégier les pistes cyclables séparées de la circulation.
2. Améliorer l'éclairage sur les voies cyclables.
3. Cibler les communes à haut risque avec peu d'aménagements.
"""
    
    print(conclusion)
    
    # Sauvegarder la conclusion
    with open(os.path.join(OUTPUT_DIR, 'conclusion_regression.txt'), 'w') as f:
        f.write(conclusion)
    print(f"\n✓ Conclusion sauvegardée: {OUTPUT_DIR}/conclusion_regression.txt")

def main():
    """Fonction principale."""
    print("\n" + "█" * 70)
    print("  PRÉDICTION DU NOMBRE D'ACCIDENTS DE VÉLO - ÎLE-DE-FRANCE")
    print("█" * 70)
    
    # 1. Charger les données
    X, y, feature_names, df = charger_donnees()
    
    # 2. Préparer les données
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = \
        preparer_donnees(X, y)
    
    # 3. Définir les modèles
    models = definir_modeles()
    
    # 4. Évaluer les modèles
    results_df, trained_models, predictions = evaluer_modeles(
        models, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 5. Afficher le tableau comparatif
    results_sorted = afficher_tableau_comparatif(results_df)
    
    # 6. Identifier le meilleur modèle (par R²)
    best_model_name = results_sorted.iloc[0]['Modèle']
    print(f"\n🏆 Meilleur modèle: {best_model_name}")
    
    # 7. Visualisations
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    
    tracer_predictions_vs_reel(predictions, y_test, best_model_name)
    tracer_predictions_vs_reel_zoom(predictions, y_test, best_model_name)
    tracer_residus(predictions, y_test, best_model_name)
    tracer_importance_features(trained_models, feature_names)
    tracer_comparaison_metriques(results_sorted)
    tracer_correlation_features(X, y, feature_names)
    
    # 8. Conclusion
    generer_conclusion(results_sorted, best_model_name, X, y)
    
    print("\n" + "█" * 70)
    print("  ✅ ANALYSE DE RÉGRESSION TERMINÉE")
    print("█" * 70)
    
    return results_sorted, trained_models

if __name__ == "__main__":
    results, models = main()
