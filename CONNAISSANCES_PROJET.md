# 📚 Guide des Connaissances - Projet Accidents Vélo IDF

**Projet Sciences des Données et Apprentissage 2025/2026**  
*Analyse et Prédiction des Accidents de Vélo en Île-de-France*

---

## 🎯 Vue d'ensemble du projet

Ce projet fusionne **4 datasets** (accidents, aménagements cyclables, comptages vélo, population INSEE) pour analyser et prédire les accidents de vélo en Île-de-France via 3 approches ML :
1. **Classification** : Identifier les communes à risque élevé
2. **Régression** : Prédire le nombre brut d'accidents
3. **Régression sur taux normalisés** : Prédire des taux de risque (par km, par habitant)

---

## 📂 PARTIE 1 : Recherche & Conception des Features
*David Chhoa & Jérémie Masnou*

### 1.1 Recherche de données sur data.gouv.fr

#### Compétences requises
- **Navigation efficace** sur les portails open data
- **Évaluation de la qualité** des datasets :
  - Complétude (% de valeurs manquantes)
  - Fraîcheur (date de mise à jour)
  - Documentation (métadonnées, description des colonnes)
  - Format (CSV, JSON, Excel)
  - Licence (Open Data, gratuit)
  
#### Datasets sélectionnés et justification

| Dataset | Taille | Intérêt | Difficultés |
|---------|--------|---------|-------------|
| **Accidents vélo** | 80k accidents (22k IDF) | Variable cible principale | Géolocalisation parfois imprécise |
| **Aménagements cyclables** | 143k infrastructures | Features principales d'exposition | Formats hétérogènes (OSM) |
| **Comptages vélo** | 933k mesures | Proxy du trafic cycliste | Couverture limitée (69 compteurs) |
| **Population INSEE** | 1287 communes IDF | Normalisation des taux | Nécessite jointure par code INSEE |

#### Pourquoi ces choix ?
- **Complémentarité** : accidents (sortie) + aménagements (features) + comptages (activité) + population (normalisation)
- **Granularité commune** : toutes les données agrégables au niveau communal via code INSEE
- **Période cohérente** : données récentes (2015-2023)

---

### 1.2 Conception des nouvelles features (Feature Engineering)

#### 1.2.1 Variables agrégées par commune

**Pourquoi agréger au niveau communal ?**
- Unité géographique administrative stable
- Code INSEE comme clé de jointure unique
- Échelle pertinente pour les politiques publiques

**Agrégations réalisées (fichier `01_preparation_donnees.py`)**

```python
# ACCIDENTS - Statistiques par commune
df_accidents_agg = df_accidents.groupby('code_insee').agg(
    nb_accidents=('Num_Acc', 'count'),                        # Total
    nb_accidents_graves=('grav', lambda x: (x <= 2).sum()),   # Tués + hospitalisés
    nb_accidents_mortels=('grav', lambda x: (x == 1).sum()),  # Décès uniquement
    gravite_moyenne=('grav', 'mean'),                         # Moyenne de gravité
    age_moyen_victimes=('age', 'mean'),                       # Profil démographique
    # Conditions environnementales
    nb_accidents_nuit=('lum', lambda x: (x.isin([2,3,4,5])).sum()),
    nb_accidents_pluie=('atm', lambda x: (x.isin([2,3,4,5,6,7])).sum()),
    nb_accidents_mouille=('surf', lambda x: (x == 2).sum())
)

# AMÉNAGEMENTS - Infrastructure cyclable
df_amenagements_agg = df_amenagements.groupby('code_insee').agg(
    nb_amenagements=('osm_id', 'count'),
    longueur_totale_amenagements=('longueur', 'sum'),         # Mètres totaux
    longueur_moyenne_amenagement=('longueur', 'mean'),
    # Types de voies (classification OSM)
    nb_voies_principales=('highway', lambda x: x.isin(['primary','secondary','tertiary']).sum()),
    nb_voies_residentielles=('highway', lambda x: (x == 'residential').sum()),
    nb_pistes_cyclables=('highway', lambda x: (x == 'cycleway').sum()),
    # Sens de circulation
    nb_double_sens=('sens_voit', lambda x: (x == 'DOUBLE').sum()),
    nb_sens_unique=('sens_voit', lambda x: (x == 'UNIQUE').sum()),
    # Qualité du revêtement
    nb_asphalt=('revetement', lambda x: (x == 'asphalt').sum())
)
```

**Pourquoi ces variables ?**
- `nb_accidents_graves` : discrimine la gravité, pas juste la quantité
- `nb_pistes_cyclables` : infrastructure dédiée vs voie partagée (sécurité différente)
- `nb_accidents_nuit` : facteur de risque connu (visibilité)
- `longueur_totale_amenagements` : exposition au risque (plus d'infra = plus d'usage)

---

#### 1.2.2 Taux de risque (Feature Engineering avancé)

**➡️ TAUX 1 : Accidents par km d'aménagement**

```python
taux_risque_par_km = nb_accidents / (longueur_totale_amenagements / 1000)
```

**Justification :**
- Normalise par l'**exposition** à l'infrastructure
- Une commune avec 100 km d'aménagements et 50 accidents est moins dangereuse qu'une commune avec 10 km et 30 accidents
- **Interprétation** : "Combien d'accidents pour 1 km de piste cyclable ?"
- **Limite** : ne tient pas compte du trafic réel (nombre de cyclistes)

**➡️ TAUX 2 : Accidents pour 10 000 habitants**

```python
taux_risque_par_habitant = (nb_accidents / population) * 10000
```

**Justification :**
- Normalise par la **population** (proxy de l'activité)
- Permet de comparer petites et grandes communes (Paris vs village)
- **Interprétation** : "Pour 10 000 habitants, combien d'accidents ?"
- **Échelle** : 10 000 pour avoir des nombres > 1 (lisibilité)
- **Limite** : assume que le nombre de cyclistes est proportionnel à la population

**➡️ VARIABLE BINAIRE : Risque élevé**

```python
seuil_risque = df['nb_accidents'].quantile(0.75)  # 75e percentile
risque_eleve = (nb_accidents >= seuil_risque).astype(int)  # 1 ou 0
```

**Justification du seuil (75e percentile) :**
- **Déséquilibre maîtrisé** : 25% de communes à risque élevé (équilibré pour ML)
- Évite les seuils arbitraires (ex: "5 accidents") qui ignorent la distribution
- **Approche data-driven** : le seuil s'adapte aux données
- Dans ce projet : seuil = 6 accidents (25% des communes ont ≥6 accidents)

**➡️ AUTRES FEATURES DÉRIVÉES**

```python
# Ratios (proportions)
ratio_pistes_cyclables = nb_pistes_cyclables / nb_amenagements
ratio_double_sens = nb_double_sens / nb_amenagements
ratio_amenagements_par_accident = nb_amenagements / nb_accidents

# Densité
densite_pop_amenagement = population / (longueur_totale_amenagements / 1000)

# Indicateurs binaires
est_paris = (departement == '75').astype(int)
```

**Pourquoi ces ratios ?**
- `ratio_pistes_cyclables` : qualité de l'infra (piste séparée = + sécurité)
- `densite_pop_amenagement` : congestion potentielle
- `est_paris` : Paris a un profil très différent (forte densité, tourisme)

---

### 1.3 Analyse critique des résultats

#### 1.3.1 Métriques de classification

**Matrice de confusion et métriques dérivées**

| Métrique | Formule | Interprétation | Quand optimiser |
|----------|---------|----------------|-----------------|
| **Accuracy** | `(TP + TN) / Total` | % de prédictions correctes | Équilibré |
| **Precision** | `TP / (TP + FP)` | % de vrais positifs parmi les prédits positifs | Coût élevé des faux positifs |
| **Recall** | `TP / (TP + FN)` | % de vrais positifs détectés | Coût élevé des faux négatifs |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Moyenne harmonique | Compromis P/R |
| **ROC-AUC** | Aire sous courbe ROC | Capacité à discriminer | Global (seuil flexible) |

**Résultats du projet (Classification du risque)**

| Modèle | Accuracy | F1-Score | ROC-AUC | Interprétation |
|--------|----------|----------|---------|----------------|
| Régression Logistique | 0.884 | **0.794** | **0.956** | ✅ Meilleur compromis |
| Random Forest | 0.898 | 0.793 | 0.951 | ✅ Accuracy légèrement supérieure |
| XGBoost | 0.884 | 0.780 | 0.941 | ⚠️ Moins bon sur F1 |

**Analyse critique :**
- 🟢 **ROC-AUC > 0.95** : excellente capacité de discrimination
- 🟢 **F1 ~ 0.79** : bon équilibre precision/recall malgré le déséquilibre de classes
- 🟡 **Pourquoi régression logistique gagne ?** 
  - Problème relativement **linéaire** (features bien construites)
  - `class_weight='balanced'` gère bien le déséquilibre
  - Moins d'overfitting que les modèles complexes

---

#### 1.3.2 Métriques de régression

**Interprétation des métriques**

| Métrique | Formule | Unité | Interprétation | Avantages | Inconvénients |
|----------|---------|-------|----------------|-----------|---------------|
| **RMSE** | `√(Σ(y - ŷ)² / n)` | Même que y | Erreur quadratique moyenne | Pénalise grandes erreurs | Sensible aux outliers |
| **MAE** | `Σ|y - ŷ| / n` | Même que y | Erreur absolue moyenne | Robuste aux outliers | Moins sensible aux grandes erreurs |
| **R²** | `1 - SS_res/SS_tot` | Sans unité [0,1] | % de variance expliquée | Intuitive (0-100%) | Peut être négatif si modèle très mauvais |
| **MAPE** | `100 × Σ|y - ŷ|/y / n` | % | Erreur en pourcentage | Interprétable en % | Undefined si y=0 |

**Résultats du projet (Prédiction nb accidents)**

| Modèle | RMSE | MAE | R² | Analyse |
|--------|------|-----|----|---------|
| XGBoost | 14.2 | 5.3 | **0.721** | ✅ Meilleure variance expliquée |
| Gradient Boosting | 14.5 | 5.5 | 0.712 | ✅ Très proche |
| Random Forest | 15.1 | 5.8 | 0.688 | ⚠️ Un peu moins bon |
| Linear Regression | 18.3 | 7.2 | 0.534 | ❌ Trop simple (non-linéarités) |

**Analyse critique :**
- 🟢 **R² ~ 0.72** : 72% de la variance expliquée (correct pour données réelles)
- 🟡 **RMSE = 14 accidents** : erreur moyenne de ±14 accidents (échelle : 0-300)
- 🔴 **Limites identifiées** :
  - Forte asymétrie (Paris = 300 accidents, villages = 0-5)
  - Comptages incomplets (seulement 69 compteurs pour 1124 communes)
  - Causalité complexe (facteurs non capturés : comportement, infrastructure urbaine)

---

#### 1.3.3 Analyse de distribution (Statistiques)

**Mesures d'asymétrie et d'aplatissement**

```python
# Dans 04_analyse_taux_risque.py
skewness = stats.skew(data)      # Asymétrie
kurtosis = stats.kurtosis(data)  # Aplatissement
cv = std / mean                  # Coefficient de variation
```

**Interprétation :**
- **Skewness** :
  - `> 0` : distribution asymétrique à droite (queue longue vers valeurs élevées)
  - `~ 0` : distribution symétrique (normale)
  - `< 0` : asymétrique à gauche
  - Dans le projet : **skew ~ 5-8** → fortement asymétrique (beaucoup de petites valeurs, quelques grandes)
  
- **Kurtosis** :
  - `> 0` : distribution leptokurtique (pic pointu, queues lourdes)
  - `~ 0` : normale
  - `< 0` : platykurtique (aplatie)
  - Dans le projet : **kurt ~ 30-50** → pics extrêmes (Paris vs villages)

**Transformation log pour normaliser :**
```python
y_log = np.log1p(y)  # log(1 + y) pour éviter log(0)
```
**Pourquoi ?** Réduit l'asymétrie, rend la distribution plus normale (meilleure performance des modèles linéaires)

---

#### 1.3.4 Corrélations

**Analyse de Pearson :**
```python
correlation = df['feature'].corr(df['target'])
```

**Résultats clés (corrélations avec nb_accidents) :**
| Feature | Corrélation | Interprétation |
|---------|-------------|----------------|
| `nb_amenagements` | **+0.82** | ✅ Forte : plus d'infra → plus d'accidents (causalité : + usage) |
| `population` | **+0.76** | ✅ Forte : villes → + accidents |
| `longueur_totale_amenagements` | **+0.79** | ✅ Exposition |
| `comptage_total_commune` | **+0.65** | 🟡 Modérée : données incomplètes |
| `ratio_pistes_cyclables` | **-0.12** | ⚠️ Faible négative : + de pistes séparées → - d'accidents ? |

**Analyse critique :**
- Corrélations fortes attendues (exposition)
- **Attention** : corrélation ≠ causalité
  - Ex : `nb_amenagements` corrélé car communes avec + d'infra ont + de cyclistes
  - Pas nécessairement que l'infra cause les accidents

---

### 1.4 Rédaction du rapport (LaTeX)

#### Structure d'un article scientifique

```latex
\section{Introduction}          % Contexte + objectifs
\section{Données et Méthodologie}  % Description datasets + fusion
\section{Analyse 1 : Classification}  % Question 1 + résultats
\section{Analyse 2 : Régression}      % Question 2 + résultats
\section{Analyse 3 : Taux de risque}  % Question 3 + résultats
\section{Discussion}             % Limites + interprétation
\section{Conclusion}             % Synthèse + perspectives
```

#### Bonnes pratiques LaTeX

**Tableaux :**
```latex
\begin{table}[H]
    \centering
    \caption{Performance des modèles}
    \label{tab:classification}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Modèle} & \textbf{Accuracy} & \textbf{F1} \\
        \midrule
        XGBoost & 0.884 & 0.780 \\
        \bottomrule
    \end{tabular}
\end{table}
```

**Figures :**
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{predictions_vs_reel.png}
    \caption{Prédictions vs valeurs réelles}
    \label{fig:predictions}
\end{figure}
```

**Références croisées :**
```latex
Voir Tableau~\ref{tab:classification} et Figure~\ref{fig:predictions}
```

#### Vocabulaire technique français

| Anglais | Français | Exemple |
|---------|----------|---------|
| Machine Learning | Apprentissage automatique | "modèles d'apprentissage automatique" |
| Feature Engineering | Ingénierie des variables | "création de nouvelles variables" |
| Overfitting | Surapprentissage | "risque de surapprentissage" |
| Train/Test split | Séparation entraînement/test | "ensemble d'entraînement" |
| Cross-validation | Validation croisée | "validation croisée à 5 plis" |
| Baseline | Modèle de référence | "modèle de base pour comparaison" |

---

---

## 🤖 PARTIE 2 : Fusion, Préparation & Modèles de Régression
*Nicolas Huyghe*

### 2.1 Fusion des datasets

#### 2.1.1 Stratégies de jointure (pandas)

**Types de merges et justifications :**

```python
# 1. Accidents + Aménagements : OUTER JOIN
df_final = pd.merge(
    df_accidents_agg,
    df_amenagements_agg,
    on='code_insee',
    how='outer',  # Garde TOUTES les communes (avec ou sans accidents)
    indicator='_merge_amenagements'
)
```
**Pourquoi `outer` ?**
- Beaucoup de communes ont des aménagements mais 0 accident (sécurité ?)
- On veut étudier ces cas aussi (prédire 0 accidents)
- `indicator` permet de diagnostiquer la fusion

**Statistiques de fusion du projet :**
```
Communes avec accidents ET aménagements : 758
Communes avec accidents uniquement      : 143 (rural, pas d'infra)
Communes avec aménagements uniquement   : 223 (très sûr ou sous-reporting ?)
```

```python
# 2. + Comptages : LEFT JOIN
df_final = pd.merge(
    df_final,
    df_comptages_communes,
    on='code_insee',
    how='left'  # Garde toutes les communes même sans compteur
)
```
**Pourquoi `left` ?**
- Seulement 69 compteurs pour 1124 communes (couverture 6%)
- Ne pas perdre les communes sans compteur (mettre NaN → fillna(0))

```python
# 3. + Population : LEFT JOIN
df_final = pd.merge(
    df_final,
    df_population,
    on='code_insee',
    how='left'
)
```

---

#### 2.1.2 Association géospatiale (compteurs → communes)

**Problème :** Les compteurs n'ont pas de code INSEE, seulement des coordonnées GPS.

**Solution :** Distance euclidienne entre coordonnées

```python
from scipy.spatial.distance import cdist

# Matrices de coordonnées
coords_compteurs = df_compteurs[['lat', 'long']].values  # (69, 2)
coords_communes = df_communes[['lat_moyenne', 'long_moyenne']].values  # (1124, 2)

# Matrice de distances (69 x 1124)
distances = cdist(coords_compteurs, coords_communes, metric='euclidean')

# Pour chaque compteur, trouver la commune la plus proche
idx_commune_proche = distances.argmin(axis=1)  # (69,)
df_compteurs['code_insee'] = communes.iloc[idx_commune_proche]['code_insee']
```

**Limites :**
- Distance euclidienne sur lat/long ≠ distance réelle (projection)
- Un compteur peut être à la frontière de 2 communes
- Mieux : utiliser `geopandas` avec vraies géométries (polygones communaux)

---

### 2.2 Préparation des données (Data Cleaning)

#### 2.2.1 Gestion des valeurs manquantes

**Stratégies selon le type de variable :**

```python
# 1. Variables de comptage : NaN → 0 (absence = 0)
accident_cols = ['nb_accidents', 'nb_accidents_graves']
df[accident_cols] = df[accident_cols].fillna(0).astype(int)

# 2. Ratios : NaN → 0 (si dénominateur = 0)
df['taux_accidents_graves'] = df['taux_accidents_graves'].fillna(0)

# 3. Population : NaN → garder NaN puis exclure si nécessaire
# (car 0 habitant n'a pas de sens, c'est vraiment une donnée manquante)
```

**Valeurs infinies :**
```python
# Division par 0 → inf
df['ratio'] = df['a'] / df['b']  # Si b=0 → inf

# Solution
df = df.replace([np.inf, -np.inf], np.nan)  # inf → NaN
df = df.fillna(0)  # ou fillna(valeur_appropriée)
```

---

#### 2.2.2 Conversion de types

```python
# Codes INSEE : forcer en string avec padding
df['code_insee'] = df['com'].str.zfill(5)  # '75' → '75000' (Paris)

# Dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['annee'] = df['date'].dt.year

# Numériques avec gestion d'erreurs
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')  # Invalides → NaN
```

**Pourquoi `errors='coerce'` ?**
- Données réelles contiennent souvent des valeurs invalides ('NA', 'N/A', '')
- `coerce` transforme en NaN au lieu de lever une erreur

---

### 2.3 Implémentation des modèles de régression

#### 2.3.1 Gradient Boosting (sklearn)

**Code :**
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,      # Nombre d'arbres
    max_depth=5,           # Profondeur max de chaque arbre
    learning_rate=0.1,     # Taux d'apprentissage (par défaut)
    random_state=42        # Reproductibilité
)
model.fit(X_train, y_train)
```

**Principe du Gradient Boosting :**
1. **Séquentiel** : chaque arbre corrige les erreurs du précédent
2. **Gradient descent** : optimise une fonction de perte (MSE pour régression)
3. **Weak learners** : arbres peu profonds (max_depth=5)

**Pourquoi ces hyperparamètres ?**

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `n_estimators=100` | 100 arbres | Compromis temps/performance (50 = sous-apprentissage, 500 = lent) |
| `max_depth=5` | Profondeur 5 | Évite l'overfitting (arbres simples = régularisation) |
| `learning_rate=0.1` | 0.1 (défaut) | Compromis stabilité/vitesse (0.01 = lent, 0.5 = instable) |

**Avantages :**
- ✅ Gère bien les non-linéarités
- ✅ Robuste aux outliers
- ✅ Pas besoin de normalisation

**Inconvénients :**
- ❌ Sensible à l'overfitting si `max_depth` trop grand
- ❌ Plus lent que Random Forest (séquentiel)

---

#### 2.3.2 XGBoost (Extreme Gradient Boosting)

**Code :**
```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0           # Silence les logs
)
model.fit(X_train, y_train)
```

**Différences avec GB sklearn :**
| Aspect | Gradient Boosting | XGBoost |
|--------|-------------------|---------|
| **Régularisation** | Non | Oui (L1/L2 sur poids des feuilles) |
| **Parallélisation** | Non (séquentiel) | Oui (construction d'arbres parallèle) |
| **Gestion NaN** | Non (erreur) | Oui (natif) |
| **Optimisation** | Standard | Cache-aware, sparsity-aware |
| **Vitesse** | Référence | **2-10x plus rapide** |

**Pourquoi XGBoost est meilleur dans ce projet ?**
- ✅ **Régularisation automatique** → moins d'overfitting
- ✅ **Plus rapide** : 100 arbres entraînés en ~2s vs 10s pour GB
- ✅ **Gère mieux les features peu importantes** (pruning intelligent)
- ✅ Très utilisé en compétition (Kaggle)

**Hyperparamètres spécifiques XGBoost :**
```python
XGBRegressor(
    reg_alpha=0,       # Régularisation L1 (Lasso) - 0 = pas de L1
    reg_lambda=1,      # Régularisation L2 (Ridge) - 1 = défaut
    subsample=1.0,     # % d'échantillons par arbre (1.0 = tous)
    colsample_bytree=1.0  # % de features par arbre
)
```

---

#### 2.3.3 LightGBM (Light Gradient Boosting Machine)

**Code :**
```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    verbose=-1  # Silence complet
)
```

**Différence clé : Leaf-wise vs Level-wise**

| Gradient Boosting / XGBoost | LightGBM |
|----------------------------|----------|
| **Level-wise** : construit l'arbre niveau par niveau | **Leaf-wise** : développe la feuille avec plus de gain |
| Plus équilibré | Plus profond, plus rapide |
| Moins d'overfitting | Risque d'overfitting si `max_depth` trop grand |

**Quand utiliser LightGBM ?**
- ✅ Très grands datasets (millions de lignes) → + rapide que XGBoost
- ✅ Features catégorielles (gestion native)
- ⚠️ Sur petits datasets : XGBoost souvent meilleur

---

#### 2.3.4 Comparaison finale des modèles de régression

| Modèle | Principe | Vitesse | Performance | Quand utiliser |
|--------|----------|---------|-------------|----------------|
| **Linear Regression** | Régression linéaire simple | ⚡⚡⚡ Très rapide | ⭐⭐ Faible (relations linéaires seulement) | Baseline, problèmes simples |
| **Ridge** | Régression linéaire + régularisation L2 | ⚡⚡⚡ | ⭐⭐ | Features corrélées |
| **Lasso** | Régression linéaire + régularisation L1 | ⚡⚡ | ⭐⭐ | Sélection de features (met certains coefs à 0) |
| **Random Forest** | Ensemble d'arbres indépendants (bagging) | ⚡⚡ | ⭐⭐⭐ | Bon baseline non-linéaire |
| **Gradient Boosting** | Boosting séquentiel d'arbres | ⚡ Lent | ⭐⭐⭐⭐ | Référence sklearn |
| **XGBoost** | GB optimisé + régularisation | ⚡⚡ Rapide | ⭐⭐⭐⭐⭐ | **Meilleur choix général** |
| **LightGBM** | GB leaf-wise | ⚡⚡⚡ Très rapide | ⭐⭐⭐⭐ | Très grands datasets |

**Dans ce projet :**
- 🥇 **XGBoost** : R² = 0.721, RMSE = 14.2 → meilleur modèle
- 🥈 **Gradient Boosting** : R² = 0.712 → très proche
- 🥉 **Random Forest** : R² = 0.688 → bon baseline

---

### 2.4 Préparation ML (Workflow complet)

#### 2.4.1 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% en test
    random_state=42,    # Reproductibilité
    stratify=y          # Pour classification : garde les proportions de classes
)
```

**Pourquoi 80/20 ?**
- Compromis biais-variance :
  - Plus de train → modèle apprend mieux
  - Plus de test → évaluation fiable
- Convention : 80/20 pour ~1000 échantillons, 90/10 pour > 10k

**Pourquoi `random_state=42` ?**
- Reproductibilité : même split à chaque exécution
- 42 = convention (référence "Hitchhiker's Guide to the Galaxy")

**Pourquoi `stratify=y` (classification) ?**
```python
# Sans stratify
y_train: [0:750, 1:250]  # Peut être déséquilibré
y_test:  [0:150, 1:50]

# Avec stratify=y
y_train: [0:800, 1:200]  # Garde la proportion 75/25
y_test:  [0:100, 1:25]
```

---

#### 2.4.2 Normalisation (Standardization)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
X_test_scaled = scaler.transform(X_test)        # Transform seulement
```

**Formule :**
```
X_scaled = (X - mean) / std
```

**Pourquoi normaliser ?**
- **Modèles sensibles à l'échelle** : régression logistique, SVM, réseaux de neurones
  - Ex : `population` (0-2M) vs `ratio_pistes_cyclables` (0-1)
  - Sans normalisation : population domine
- **Pas nécessaire** pour arbres (Random Forest, XGBoost) : insensibles à l'échelle

**⚠️ IMPORTANT : fit sur train, transform sur test**
```python
# ❌ ERREUR (data leakage)
scaler.fit(X)  # Utilise les stats du test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ CORRECT
scaler.fit(X_train)  # Stats du train uniquement
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

#### 2.4.3 Validation croisée (Cross-Validation)

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, 
    X_train, 
    y_train, 
    cv=5,              # 5-fold CV
    scoring='r2'       # Métrique
)

print(f"R² moyen: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

**Principe (5-fold) :**
```
Train: [====|====|====|====|----]  Test: [----]  → Score 1
Train: [====|====|====|----|----|  Test: [====]  → Score 2
Train: [====|====|----|----|====]  Test: [====]  → Score 3
Train: [====|----|----|====|====]  Test: [====]  → Score 4
Train: [----|----|====|====|====]  Test: [====]  → Score 5

Score final = mean(5 scores) ± std
```

**Pourquoi CV ?**
- Évalue la **stabilité** du modèle
- Détecte l'**overfitting** : si `score_train >> score_cv`, overfitting
- Utilise **toutes les données** pour validation

**Dans le projet :**
```
XGBoost : CV R² = 0.698 (+/- 0.042)
→ Modèle stable (faible std)
```

---

### 2.5 Évaluation et visualisation

#### 2.5.1 Graphiques de prédictions

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Parfait')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title(f'XGBoost - R² = {r2:.3f}')
```

**Interprétation :**
- Points sur la ligne rouge : prédictions parfaites
- Points au-dessus : surestimation
- Points en-dessous : sous-estimation
- Dispersion : incertitude du modèle

---

#### 2.5.2 Feature Importance (arbres)

```python
importances = model.feature_importances_  # Modèles à arbres uniquement
indices = np.argsort(importances)[::-1][:10]  # Top 10

plt.barh(feature_names[indices], importances[indices])
plt.xlabel('Importance')
```

**Calcul (Gini importance) :**
- Pour chaque split d'un arbre : gain = réduction de l'erreur
- Importance d'une feature = somme des gains pour cette feature

**Dans le projet (XGBoost) :**
```
1. longueur_totale_amenagements : 0.352  → Très important (exposition)
2. population                    : 0.198  → Important (activité)
3. comptage_total_commune        : 0.142  → Modéré (trafic)
4. nb_amenagements               : 0.089  → Modéré
```

---

### 2.6 Gestion du déséquilibre (Classification)

**Problème :**
```
Classe 0 (risque faible)  : 843 communes (75%)
Classe 1 (risque élevé)   : 281 communes (25%)
```

**Solutions implémentées :**

#### 1. Pondération des classes
```python
LogisticRegression(class_weight='balanced')
RandomForestClassifier(class_weight='balanced')
```
**Effet :** Pénalise plus les erreurs sur la classe minoritaire

**Formule :**
```python
weight_class_0 = n_samples / (2 * n_class_0)  # 1124 / (2*843) = 0.67
weight_class_1 = n_samples / (2 * n_class_1)  # 1124 / (2*281) = 2.00
```

#### 2. Scale pos weight (XGBoost)
```python
XGBClassifier(scale_pos_weight=3)
```
**Effet :** Multiplie le poids des exemples positifs par 3
**Calcul :** `ratio = n_class_0 / n_class_1 = 843/281 = 3`

---

## 🎓 Concepts ML Transversaux

### Overfitting vs Underfitting

| Concept | Définition | Symptômes | Solutions |
|---------|------------|-----------|-----------|
| **Overfitting** | Modèle trop complexe, apprend le bruit | `score_train >> score_test` | ↓ Complexité, régularisation, + données |
| **Underfitting** | Modèle trop simple | `score_train` et `score_test` faibles | ↑ Complexité, + features |
| **Good fit** | Compromis optimal | `score_train ≈ score_test` | ✅ |

**Dans le projet :**
```
XGBoost : Train R² = 0.756, Test R² = 0.721
→ Légèrement overfitté mais acceptable (écart < 5%)
```

---

### Boosting vs Bagging

| Aspect | Bagging (Random Forest) | Boosting (XGBoost) |
|--------|-------------------------|---------------------|
| **Principe** | Arbres indépendants en parallèle | Arbres séquentiels corrigeant les erreurs |
| **Biais** | Modéré | Faible |
| **Variance** | Faible (moyennage) | Modérée |
| **Overfitting** | Résistant | Sensible si mal réglé |
| **Vitesse** | Rapide (parallèle) | Lent (séquentiel) |
| **Performance** | Bonne | Excellente |

---

## 📊 Résumé des choix techniques

### Hyperparamètres finaux

| Modèle | Hyperparamètres | Justification |
|--------|-----------------|---------------|
| **XGBoost Régression** | `n_estimators=100, max_depth=5, lr=0.1` | Compromis temps/performance |
| **XGBoost Classification** | `n_estimators=100, scale_pos_weight=3` | Gestion déséquilibre |
| **Gradient Boosting** | `n_estimators=100, max_depth=5` | Évite overfitting (arbres simples) |
| **Random Forest** | `n_estimators=100, max_depth=10` | + profond car bagging = régularisation |
| **Régression Logistique** | `class_weight='balanced', max_iter=1000` | Convergence + équilibrage |

### Split et validation

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `test_size` | 0.2 (20%) | Convention 80/20 pour ~1000 échantillons |
| `random_state` | 42 | Reproductibilité |
| `cv` | 5 folds | Bon compromis variance/biais |
| `stratify` | `y` (classification) | Garde proportions des classes |

---

## 🎤 Conseils pour la présentation

### Parler des features
> "Nous avons créé deux **taux de risque normalisés** : le premier divise le nombre d'accidents par les kilomètres d'aménagements pour mesurer le risque par exposition à l'infrastructure, le second normalise par la population pour comparer des communes de tailles différentes."

### Parler des modèles
> "Nous avons testé 7 algorithmes de régression. **XGBoost** s'est révélé le plus performant avec un **R² de 0.72**, expliquant 72% de la variance des accidents. Ce modèle utilise un **gradient boosting optimisé** avec régularisation L2 pour éviter le surapprentissage."

### Parler des hyperparamètres
> "Nous avons fixé `n_estimators=100` pour un compromis entre temps de calcul et performance, et `max_depth=5` pour limiter la complexité de chaque arbre et éviter l'overfitting sur notre dataset de 1124 communes."

### Parler des limites
> "Notre analyse présente certaines limites : les données de comptage couvrent seulement 6% des communes, et nous n'avons pas capturé certains facteurs comportementaux. Néanmoins, un **R² de 0.72** reste satisfaisant pour des données réelles avec une forte hétérogénéité."

---

## 📚 Références et ressources

### Librairies Python
- **pandas** : manipulation de données tabulaires
- **numpy** : calcul numérique
- **scikit-learn** : modèles ML, preprocessing, métriques
- **xgboost** : gradient boosting optimisé
- **lightgbm** : gradient boosting rapide
- **matplotlib/seaborn** : visualisation
- **scipy** : statistiques, distances

### Concepts clés à maîtriser
1. ✅ Feature engineering (agrégation, ratios, normalisation)
2. ✅ Fusion de datasets (merges, jointures spatiales)
3. ✅ Préprocessing (gestion NaN, outliers, types)
4. ✅ Train/test split + validation croisée
5. ✅ Métriques (R², RMSE, F1, ROC-AUC)
6. ✅ Modèles de régression (linéaires, arbres, boosting)
7. ✅ Hyperparamètres (justification des choix)
8. ✅ Analyse critique (limites, biais, causalité)

---

**Bonne présentation ! 🚀**
