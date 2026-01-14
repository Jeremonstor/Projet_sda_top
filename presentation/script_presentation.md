# Script de Présentation - Accidents de Vélo en Île-de-France
## Durée totale : 12 minutes + 5 minutes de questions

---

## Slide 1 : Page de titre (30 sec)
**Contenu :**
- Titre : "Analyse et Prédiction des Accidents de Vélo en Île-de-France"
- Noms : Nicolas Huyghe, David Chhoa, Jérémie Masnou
- Projet SDA 2025/2026

**À dire :**
> Bonjour à tous. Nous sommes Nicolas, David et Jérémie, et nous allons vous présenter notre projet sur l'analyse et la prédiction des accidents de vélo en Île-de-France. Ce projet s'inscrit dans le cadre du cours de Science des Données et Apprentissage.

---

## Slide 2 : Contexte et Problématique (1 min)
**Contenu :**
- Essor du vélo en Île-de-France (graphique évolution si possible)
- Politiques de mobilité durable, pistes cyclables, Vélib'
- Enjeu de sécurité routière : 22 609 accidents de vélo enregistrés en IDF
- Question centrale : peut-on prédire et prévenir ces accidents ?

**À dire :**
> Le vélo connaît un essor considérable en Île-de-France ces dernières années. Les politiques de mobilité durable, le développement des pistes cyclables et des services comme Vélib' ont contribué à cette croissance. Cependant, cette augmentation du nombre de cyclistes s'accompagne d'un enjeu majeur de sécurité routière. En effet, plus de 22 000 accidents impliquant des cyclistes ont été enregistrés dans la région. Notre question centrale est donc : peut-on utiliser le machine learning pour mieux comprendre ce risque et potentiellement aider à le prévenir ?

---

## Slide 3 : Questions de recherche (45 sec)
**Contenu :**
- Question 1 : **Classification** - Peut-on identifier les communes à risque élevé ?
- Question 2 : **Régression** - Peut-on prédire le nombre d'accidents par commune ?
- Question 3 : **Taux de risque** - Peut-on prédire des taux normalisés (par habitant, par km) ?

**À dire :**
> Nous avons structuré notre étude autour de trois questions de recherche. Premièrement, un problème de classification : peut-on identifier automatiquement les communes présentant un risque élevé d'accidents ? Deuxièmement, un problème de régression : peut-on prédire le nombre exact d'accidents dans une commune donnée ? Et troisièmement, après avoir identifié des limites dans notre approche initiale, nous avons ajouté une troisième question : peut-on prédire des taux de risque normalisés, c'est-à-dire rapportés à la population ou à la longueur des aménagements ?

---

## Slide 4 : Sources de données (1 min 30)
**Contenu :**
- 4 jeux de données de **data.gouv.fr** :
  1. **Accidents vélo** : 80 022 accidents nationaux → 22 609 en IDF
     - Localisation, date, gravité, conditions
  2. **Aménagements cyclables** : 143 060 infrastructures en IDF
     - Type de voie, longueur, revêtement
  3. **Comptages vélo** : 933 757 mesures (69 compteurs)
     - Trafic cycliste horaire
  4. **Population INSEE** : 1 287 communes
     - Données démographiques 2021

**À dire :**
> Nous avons utilisé quatre jeux de données, tous issus du portail data.gouv.fr. Le premier dataset contient les accidents de vélo au niveau national, soit plus de 80 000 accidents, dont 22 609 en Île-de-France. Pour chaque accident, on a la localisation GPS, la date, l'heure, la gravité, et les conditions comme la météo ou l'éclairage.
> 
> Le deuxième dataset recense les aménagements cyclables en Île-de-France : plus de 143 000 infrastructures, avec le type de voie (piste séparée, bande cyclable, voie partagée), la longueur en mètres et le revêtement.
> 
> Le troisième dataset contient des données de comptage horaire provenant de 69 compteurs automatiques, soit près d'un million de mesures. Cela permet d'estimer le trafic cycliste, même si la couverture reste limitée.
> 
> Enfin, nous avons ajouté les données de population INSEE 2021 pour pouvoir normaliser nos analyses par le nombre d'habitants.

---

## Slide 5 : Fusion et création de features (1 min)
**Contenu :**
- Schéma de fusion des 4 datasets
- Clé de jointure : **code INSEE** de la commune
- Dataset final : **1 124 communes**, **44 variables**
- Nouvelles features créées :
  - `nb_accidents`, `nb_accidents_graves`, `taux_accidents_graves`
  - `longueur_totale_amenagements`, `ratio_pistes_cyclables`
  - `taux_risque_par_km`, `taux_risque_par_habitant`
  - `risque_eleve` (variable binaire pour la classification)

**À dire :**
> L'étape de préparation des données a été cruciale. Nous avons fusionné les quatre sources au niveau communal en utilisant le code INSEE comme clé de jointure. Après nettoyage et agrégation, nous obtenons un dataset de 1 124 communes caractérisées par 44 variables.
> 
> Nous avons créé plusieurs nouvelles features. Pour les accidents : le nombre total, le nombre d'accidents graves, et le taux de gravité. Pour les aménagements : la longueur totale en mètres et le ratio de pistes cyclables séparées. Et pour l'analyse normalisée : le taux de risque par kilomètre d'aménagement et le taux pour 10 000 habitants. Enfin, nous avons créé une variable binaire "risque élevé" pour la classification.

---

## Slide 6 : Classification - Méthodologie (1 min)
**Contenu :**
- Définition : **risque élevé** si nb_accidents ≥ 75e percentile (≥ 6 accidents)
- 6 modèles testés :
  - Régression Logistique (avec pondération des classes)
  - Random Forest
  - SVM (noyau RBF)
  - Gradient Boosting
  - XGBoost
  - LightGBM
- Validation croisée 5-fold stratifiée
- Métriques : Accuracy, F1-Score, ROC-AUC

**À dire :**
> Pour la classification, nous avons défini une commune comme étant "à risque élevé" si son nombre d'accidents dépasse le 75e percentile de la distribution, soit 6 accidents ou plus. Cela représente environ 25% des communes.
> 
> Nous avons testé six algorithmes de classification : la régression logistique avec pondération des classes pour gérer le déséquilibre, le Random Forest, le SVM avec noyau RBF, et trois méthodes de boosting : Gradient Boosting, XGBoost et LightGBM.
> 
> Pour évaluer les modèles, nous avons utilisé une validation croisée 5-fold stratifiée et trois métriques principales : l'accuracy, le F1-score qui équilibre précision et rappel, et le ROC-AUC qui mesure la capacité à distinguer les deux classes.

---

## Slide 7 : Classification - Résultats (45 sec)
**Contenu :**
- Tableau des résultats :
  | Modèle | Accuracy | F1-Score | ROC-AUC |
  |--------|----------|----------|---------|
  | Rég. Logistique | 0.884 | **0.794** | **0.956** |
  | Random Forest | 0.898 | 0.793 | 0.951 |
  | SVM | 0.880 | 0.791 | 0.953 |
- **Meilleur : Régression Logistique** (AUC 95.6%)

**À dire :**
> Voici les résultats de la classification. La régression logistique obtient les meilleures performances avec un F1-score de 79.4% et surtout un ROC-AUC de 95.6%. Ce résultat peut sembler surprenant : un modèle simple surpasse des modèles plus complexes comme le boosting ou les forêts aléatoires.
> 
> Cela s'explique par la nature du problème : la relation entre les features et le risque est essentiellement linéaire. Les communes à risque élevé sont celles avec beaucoup d'aménagements, car elles attirent plus de cyclistes.

---

## Slide 8 : Courbe ROC (30 sec)
**Contenu :**
- Image : roc_curves_classification.png
- Tous les modèles > 0.94 AUC
- Courbes très proches du coin supérieur gauche

**À dire :**
> Voici les courbes ROC des différents modèles. On observe que tous les modèles atteignent un AUC supérieur à 0.94, ce qui indique une excellente capacité à distinguer les communes à risque élevé des autres. Les courbes sont très proches du coin supérieur gauche, signe d'un bon compromis entre sensibilité et spécificité.

---

## Slide 9 : Régression - Résultats initiaux (45 sec)
**Contenu :**
- Objectif : prédire le **nombre exact d'accidents** par commune
- Tableau des résultats :
  | Modèle | RMSE | R² | CV R² |
  |--------|------|-----|-------|
  | Gradient Boosting | 37.06 | **0.892** | 0.897 |
  | XGBoost | 37.53 | 0.889 | 0.900 |
  | Random Forest | 37.80 | 0.888 | 0.886 |
- Résultat apparemment **excellent** : R² = 89%

**À dire :**
> Notre deuxième objectif était de prédire le nombre exact d'accidents par commune, un problème de régression classique. Le Gradient Boosting obtient les meilleures performances avec un R² de 89.2%, ce qui signifie que le modèle explique près de 90% de la variance.
> 
> À première vue, ce résultat semble excellent. Cependant, une analyse plus approfondie nous a conduits à remettre en question cette performance.

---

## Slide 10 : Analyse critique - Problème 1 (1 min)
**Contenu :**
- **Corrélation paradoxale** : plus d'aménagements → plus d'accidents
- Graphique de corrélation (r = 0.60)
- Explication : biais de confusion
  - Plus d'aménagements → plus de cyclistes → plus d'accidents en absolu
  - Ce n'est PAS que les aménagements sont dangereux

**À dire :**
> Premier problème : nous avons observé une corrélation paradoxale. Les communes avec plus d'aménagements cyclables ont plus d'accidents, avec un coefficient de corrélation de 0.60. Autrement dit, plus une commune investit dans les pistes cyclables, plus elle a d'accidents.
> 
> Est-ce que cela signifie que les aménagements sont dangereux ? Non, bien sûr. C'est un biais de confusion classique : les communes bien équipées attirent plus de cyclistes, ce qui augmente mécaniquement le nombre d'accidents en valeur absolue, même si le risque individuel diminue peut-être. Le modèle apprend cette corrélation, mais elle n'est pas causale.

---

## Slide 11 : Analyse critique - Problèmes 2 et 3 (1 min)
**Contenu :**
- **Effet Paris** : 61% des accidents concentrés dans 20 arrondissements
  - Le modèle "apprend" surtout à identifier Paris
  - Gonfle artificiellement le R²
- **MAPE élevé** : 83% malgré R² de 89%
  - R² bon pour les grandes valeurs (Paris)
  - Mauvais pour les petites communes (majorité du dataset)
- **Distribution asymétrique** : 33.6% des communes sans accident

**À dire :**
> Deuxième problème : l'effet Paris. Les 20 arrondissements de Paris concentrent à eux seuls 61% des accidents de toute l'Île-de-France, soit près de 14 000 accidents sur 22 600. Le modèle "apprend" donc essentiellement à identifier Paris, ce qui gonfle artificiellement le R².
> 
> Troisième problème : malgré ce R² impressionnant de 89%, le MAPE, l'erreur absolue moyenne en pourcentage, atteint 83%. Cela signifie que les prédictions sont en moyenne à 83% d'écart par rapport à la réalité pour les petites communes, qui sont majoritaires dans notre dataset.
> 
> Enfin, la distribution est très asymétrique : un tiers des communes n'ont aucun accident enregistré.

---

## Slide 12 : Graphiques prédictions vs réel (45 sec)
**Contenu :**
- Image : predictions_vs_reel.png (vue globale)
- Image : predictions_vs_reel_zoom.png (zoom < 100 accidents)
- Commentaire : bonne prédiction pour Paris, dispersion pour les petites communes

**À dire :**
> Ces graphiques illustrent parfaitement le problème. Sur la vue globale à gauche, on voit que le modèle prédit bien les valeurs élevées, notamment les arrondissements parisiens qui sont les points en haut à droite.
> 
> Mais quand on zoome sur les communes avec moins de 100 accidents, qui représentent la grande majorité, on observe une dispersion importante autour de la diagonale. Le modèle est donc imprécis là où on en aurait le plus besoin.

---

## Slide 13 : Solution - Taux de risque normalisés (1 min)
**Contenu :**
- **Intégration des données de population INSEE**
- Deux nouvelles métriques :
  - **Taux par km** = Accidents / Longueur aménagements (km)
    - Compare les communes indépendamment de leur équipement
  - **Taux pour 10k habitants** = (Accidents / Population) × 10 000
    - Métrique classique en épidémiologie
- Objectif : éliminer l'effet de taille

**À dire :**
> Face à ces biais, nous avons décidé de changer d'approche. Nous avons intégré les données de population INSEE et défini deux nouvelles métriques de risque normalisées.
> 
> Le taux par kilomètre divise le nombre d'accidents par la longueur totale des aménagements. Cela permet de comparer les communes indépendamment de leur niveau d'équipement : une commune avec 10 km de pistes et 10 accidents a le même taux qu'une commune avec 1 km et 1 accident.
> 
> Le taux pour 10 000 habitants est une métrique classique en épidémiologie. Elle normalise par la population et permet de comparer le risque entre communes de tailles différentes.

---

## Slide 14 : Taux de risque - Résultats (1 min)
**Contenu :**
- Tableaux des résultats :
  - Taux par km : Ridge R² = 30%, MAPE = 70%
  - Taux par habitant : Random Forest R² = 19%, MAPE = 34%
- Performances **nettement inférieures** mais **plus honnêtes**
- Le modèle ne peut plus "tricher" avec l'effet de taille

**À dire :**
> Les résultats avec les taux normalisés sont nettement inférieurs : un R² maximum de 30% pour le taux par kilomètre avec Ridge, et 19% pour le taux par habitant avec Random Forest.
> 
> Loin d'être un échec, ces résultats sont en réalité plus honnêtes. Le modèle ne peut plus exploiter l'effet de taille des communes. Paris n'a plus un poids disproportionné dans l'apprentissage.
> 
> Ces performances modestes reflètent la complexité réelle du phénomène : le risque d'accident dépend de nombreux facteurs que nous n'avons pas dans nos données, comme le comportement des usagers, la densité du trafic automobile, ou la qualité de l'éclairage.

---

## Slide 15 : Synthèse comparative (1 min)
**Contenu :**
- Tableau récapitulatif :
  | Analyse | Modèle | Performance | Validité |
  |---------|--------|-------------|----------|
  | Classification | Rég. Log. | AUC 96% | ✓ Élevée |
  | Régression brute | Grad. Boost. | R² 89% | ⚠ Limitée |
  | Taux par km | Ridge | R² 30% | ✓ Élevée |
  | Taux par hab | Random Forest | R² 19% | ✓ Élevée |
- **Paradoxe** : meilleure performance ≠ meilleur modèle

**À dire :**
> Ce tableau résume l'ensemble de nos résultats. La classification fonctionne bien avec un AUC de 96%. La régression sur le nombre brut affiche un R² impressionnant de 89%, mais sa validité est limitée par les biais que nous avons identifiés. Les approches par taux normalisés ont des performances plus modestes mais une validité élevée.
> 
> Le paradoxe de notre étude est que la meilleure performance correspond à l'approche la moins valide. C'est un rappel important : en science des données, un bon score ne garantit pas un bon modèle. L'analyse critique est indispensable.

---

## Slide 16 : Limites de l'étude (45 sec)
**Contenu :**
- **Sous-déclaration** : tous les accidents ne sont pas signalés
- **Couverture limitée** : seulement 69 compteurs pour toute l'IDF
- **Variables manquantes** : trafic automobile, urbanisme, météo détaillée
- **33.6% de zéros** : nécessiterait des modèles Zero-Inflated

**À dire :**
> Notre étude présente plusieurs limites. Tous les accidents ne sont pas déclarés aux autorités, notamment les accidents mineurs. Nous n'avons que 69 compteurs pour estimer le trafic cycliste de toute l'Île-de-France. Il nous manque des variables importantes comme le trafic automobile ou les caractéristiques urbanistiques. Enfin, un tiers des communes n'ont aucun accident, ce qui pourrait nécessiter des modèles spécifiques comme les modèles Zero-Inflated.

---

## Slide 17 : Conclusion (45 sec)
**Contenu :**
- ✓ Classification efficace pour identifier les communes prioritaires
- ⚠ Régression brute biaisée par l'effet de taille
- ✓ Taux normalisés : approche méthodologiquement rigoureuse
- **Enseignement principal** : Performance ≠ Validité
- L'analyse critique des données est essentielle

**À dire :**
> En conclusion, la classification binaire du risque fonctionne bien et peut servir à identifier les communes prioritaires pour des interventions de sécurité. La régression sur le nombre brut d'accidents, malgré son R² élevé, est biaisée et doit être interprétée avec prudence. L'approche par taux normalisés est plus rigoureuse mais révèle la complexité du problème.
> 
> Le principal enseignement de ce projet est qu'une bonne performance ne garantit pas la validité d'un modèle. L'analyse critique des données et des résultats est essentielle pour éviter des conclusions erronées. Merci de votre attention.

---

## Slide 18 : Questions ?
**Contenu :**
- "Merci pour votre attention"
- "Questions ?"
- Contacts / GitHub (optionnel)

---

# Questions potentielles et réponses

**Q : Pourquoi la régression logistique surpasse les modèles complexes ?**
> La relation entre les features et le risque est essentiellement linéaire. Les communes à risque élevé sont simplement celles avec beaucoup d'aménagements, car elles attirent plus de cyclistes.

**Q : Comment avez-vous géré le déséquilibre des classes ?**
> Nous avons utilisé la pondération des classes (class_weight='balanced') et la stratification lors du split train/test.

**Q : Pourquoi ne pas exclure Paris de l'analyse ?**
> C'est une piste intéressante. On pourrait développer un modèle spécifique pour Paris et un autre pour le reste de l'IDF.

**Q : Quelles données supplémentaires amélioreraient les modèles ?**
> Le trafic automobile, les conditions météorologiques détaillées, les caractéristiques urbanistiques (densité du bâti, présence de commerces), et le trafic cycliste réel.

**Q : Comment interprétez-vous la corrélation paradoxale ?**
> C'est un biais de confusion classique. Les communes qui investissent dans les aménagements cyclables attirent plus de cyclistes. Plus de cyclistes = plus d'accidents en valeur absolue, même si le risque individuel diminue.
