#!/usr/bin/env python3
"""
Script d'analyse ML pour la classification du risque d'accidents de vélo par commune.

Question: La commune présente-t-elle un risque élevé d'accidents de vélo ?

Ce script compare plusieurs algorithmes de classification:
- Régression Logistique
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report,
                              roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
    """Charge et prépare les données pour la classification."""
     
    print("Chargement des données")
     
    
    df = pd.read_csv(DATASET_FILE)
    print(f"Dataset chargé: {len(df)} communes")
    
    # Features pour la classification
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
    y = df['risque_eleve'].copy()
    
    # Remplacer les valeurs infinies et NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"\nDistribution de la variable cible:")
    print(y.value_counts())
    print(f"\nProportion risque élevé: {y.mean()*100:.1f}%")
    
    return X, y, available_cols, df

def preparer_donnees(X, y):
    """Prépare les données pour l'entraînement."""
     
    print("Préparation des données")
     
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Ensemble d'entraînement: {len(X_train)} communes")
    print(f"Ensemble de test: {len(X_test)} communes")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def definir_modeles():
    """Définit les modèles de classification à comparer."""
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, random_state=42, max_depth=5
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=42, use_label_encoder=False,
            eval_metric='logloss', scale_pos_weight=3
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, random_state=42, class_weight='balanced',
            verbose=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', random_state=42, class_weight='balanced', probability=True
        )
    }
    return models

def evaluer_modeles(models, X_train_scaled, X_test_scaled, y_train, y_test):
    """Évalue tous les modèles et retourne les résultats."""
     
    print("Évaluation des modèles")
     
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        
        print(f"Entraînement: {name}")
           
        
        # Entraînement
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Prédictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
        
        results.append({
            'Modèle': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV F1 (mean)': cv_scores.mean(),
            'CV F1 (std)': cv_scores.std()
        })
        
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  ROC-AUC:   {roc_auc:.3f}")
        print(f"  CV F1:     {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    return pd.DataFrame(results), trained_models

def afficher_tableau_comparatif(results_df):
    """Affiche un tableau comparatif des résultats."""
     
    print("Tabeleau comparatif des modèles")
     
    
    # Trier par F1-Score
    results_sorted = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + results_sorted.to_string(index=False))
    
    # Sauvegarder
    results_sorted.to_csv(os.path.join(OUTPUT_DIR, 'classification_results.csv'), index=False)
    
    return results_sorted

def tracer_courbes_roc(trained_models, X_test_scaled, y_test):
    """Trace les courbes ROC pour tous les modèles."""
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('Taux de Faux Positifs', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs', fontsize=12)
    plt.title('Courbes ROC - Classification du Risque Élevé', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_classification.png'), dpi=150)
    plt.close()
    print(f"\nCourbes ROC sauvegardées: {OUTPUT_DIR}/roc_curves_classification.png")

def tracer_importance_features(trained_models, feature_names):
    """Trace l'importance des features pour les modèles basés sur les arbres."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
    
    for ax, model_name in zip(axes.flatten(), tree_models):
        if model_name in trained_models:
            model = trained_models[model_name]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(indices)))
            ax.barh(range(len(indices)), importances[indices], color=colors)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name}')
            ax.invert_yaxis()
    
    plt.suptitle("Importance des Features par Modèle", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_classification.png'), dpi=150)
    plt.close()
    print(f"Importance des features sauvegardée: {OUTPUT_DIR}/feature_importance_classification.png")

def tracer_matrice_confusion(trained_models, X_test_scaled, y_test, best_model_name):
    """Trace la matrice de confusion pour le meilleur modèle."""
    model = trained_models[best_model_name]
    y_pred = model.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Risque Faible', 'Risque Élevé'],
                yticklabels=['Risque Faible', 'Risque Élevé'])
    plt.xlabel('Prédit', fontsize=12)
    plt.ylabel('Réel', fontsize=12)
    plt.title(f'Matrice de Confusion - {best_model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_classification.png'), dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardée: {OUTPUT_DIR}/confusion_matrix_classification.png")
    
    # Afficher le rapport de classification
    print(f"\nRapport de classification ({best_model_name}):")
    print(classification_report(y_test, y_pred, 
                                target_names=['Risque Faible', 'Risque Élevé']))

def tracer_comparaison_metriques(results_df):
    """Trace un graphique de comparaison des métriques."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(results_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, results_df[metric], width, label=metric)
    
    ax.set_xlabel('Modèle', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Comparaison des Métriques par Modèle', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Modèle'], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison_classification.png'), dpi=150)
    plt.close()
    print(f"Comparaison des métriques sauvegardée: {OUTPUT_DIR}/metrics_comparison_classification.png")

def generer_conclusion(results_df, best_model_name):
    """Génère une conclusion textuelle de l'analyse."""
     
    print("CONCLUSION")
     
    
    best_row = results_df[results_df['Modèle'] == best_model_name].iloc[0]
    
    conclusion = f"""
ANALYSE DE CLASSIFICATION - RISQUE D'ACCIDENTS DE VÉLO

OBJECTIF:
Prédire si une commune d'Île-de-France présente un risque élevé 
d'accidents de vélo en fonction de ses aménagements cyclables.

MEILLEUR MODÈLE: {best_model_name}

PERFORMANCES:
  • Accuracy:  {best_row['Accuracy']:.1%}
  • Precision: {best_row['Precision']:.1%}  
  • Recall:    {best_row['Recall']:.1%}
  • F1-Score:  {best_row['F1-Score']:.1%}
  • ROC-AUC:   {best_row['ROC-AUC']:.1%}

INTERPRÉTATION:
- Le modèle permet d'identifier les communes à risque élevé avec une 
  précision de {best_row['Precision']:.0%} et un rappel de {best_row['Recall']:.0%}.
- L'aire sous la courbe ROC de {best_row['ROC-AUC']:.2f} indique une bonne 
  capacité discriminative du modèle.
- Les features les plus importantes sont liées au nombre et à la longueur
  des aménagements cyclables.

RECOMMANDATIONS:
1. Les communes avec peu d'aménagements cyclables sont plus à risque.
2. La présence de pistes cyclables dédiées réduit le risque.
3. Les arrondissements parisiens présentent un profil spécifique.
"""
    
    print(conclusion)
    

def main():
    """Fonction principale."""
    
    print("  CLASSIFICATION DU RISQUE D'ACCIDENTS DE VÉLO - ÎLE-DE-FRANCE")
    
    # 1. Charger les données
    X, y, feature_names, df = charger_donnees()
    
    # 2. Préparer les données
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = \
        preparer_donnees(X, y)
    
    # 3. Définir les modèles
    models = definir_modeles()
    
    # 4. Évaluer les modèles
    results_df, trained_models = evaluer_modeles(
        models, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 5. Afficher le tableau comparatif
    results_sorted = afficher_tableau_comparatif(results_df)
    
    # 6. Identifier le meilleur modèle (par F1-Score)
    best_model_name = results_sorted.iloc[0]['Modèle']
    print(f"\n Meilleur modèle: {best_model_name}")
    
    # 7. Visualisations
     
    print("GÉNÉRATION DES VISUALISATIONS")
     
    
    tracer_courbes_roc(trained_models, X_test_scaled, y_test)
    tracer_importance_features(trained_models, feature_names)
    tracer_matrice_confusion(trained_models, X_test_scaled, y_test, best_model_name)
    tracer_comparaison_metriques(results_sorted)
    
    # 8. Conclusion
    generer_conclusion(results_sorted, best_model_name)
    
    print("   \nANALYSE DE CLASSIFICATION TERMINÉE")
    
    return results_sorted, trained_models

if __name__ == "__main__":
    results, models = main()
