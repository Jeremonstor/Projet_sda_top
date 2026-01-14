# Script de Présentation - Accidents de Vélo en Île-de-France
## Durée totale : 12 minutes + 5 minutes de questions

---

# RÉPARTITION DES INTERVENANTS

| Intervenant | Slides | Durée | Thèmes |
|-------------|--------|-------|--------|
| **Nicolas Huyghe** | 1-7 | ~4 min | Introduction, données, fusion, début classification |
| **David Chhoa** | 8-14 | ~4 min | Fin classification, régression, analyse critique |
| **Jérémie Masnou** | 15-22 | ~4 min | Graphiques, taux normalisés, conclusion |

---

# NICOLAS HUYGHE (Slides 1-7) — ~4 minutes

---

## Slide 1 : Page de titre (30 sec)
**Contenu :**
- Titre : "Analyse et Prédiction des Accidents de Vélo en Île-de-France"
- Noms : Nicolas Huyghe, David Chhoa, Jérémie Masnou
- Projet SDA 2025/2026

**À dire :**
> Bonjour à tous. Nous sommes Nicolas, David et Jérémie, et nous allons vous présenter notre projet d'analyse et de prédiction des accidents de vélo en Île-de-France, réalisé dans le cadre du cours de Science des Données et Apprentissage. Je vais commencer par vous présenter le contexte et les données, puis David vous parlera de nos modèles et de l'analyse critique, et enfin Jérémie conclura avec notre approche améliorée.

---

## Slide 2 : Contexte et Problématique (45 sec)
**Contenu :**
- Essor du vélo en Île-de-France (Vélib', pistes cyclables)
- Politiques de mobilité durable en expansion
- Enjeu de sécurité routière majeur
- 22 609 accidents de vélo enregistrés en Île-de-France
- Question : peut-on prédire et prévenir ces accidents ?

**À dire :**
> Le vélo connaît un essor considérable en Île-de-France. Avec le développement de Vélib', des pistes cyclables et des politiques de mobilité durable, de plus en plus de franciliens utilisent le vélo au quotidien. Cependant, cette croissance s'accompagne d'un enjeu majeur de sécurité routière : plus de 22 600 accidents impliquant des cyclistes ont été enregistrés dans la région. Notre objectif est donc d'utiliser le machine learning pour mieux comprendre et potentiellement prédire ce risque.

---

## Slide 3 : Questions de Recherche (30 sec)
**Contenu :**
- Classification : Identifier les communes à risque élevé
- Régression : Prédire le nombre d'accidents par commune
- Taux de risque : Prédire des taux normalisés (par habitant, par km)

**À dire :**
> Nous avons structuré notre étude autour de trois questions. Premièrement, un problème de classification : peut-on identifier les communes à risque élevé ? Deuxièmement, un problème de régression : peut-on prédire le nombre d'accidents ? Et troisièmement, après avoir identifié des limites, nous avons ajouté une question sur les taux de risque normalisés.

---

## Slide 4 : Sources de Données (1 min)
**Contenu :**
- Accidents vélo : 80 022 nationaux → 22 609 en IDF (localisation, date, gravité, conditions)
- Aménagements cyclables : 143 060 infrastructures (type de voie, longueur, revêtement)
- Comptages vélo : 933 757 mesures (69 compteurs)
- Population INSEE : 1 287 communes (2021)

**À dire :**
> Nous avons utilisé quatre jeux de données, tous issus de data.gouv.fr. Le premier contient les accidents de vélo au niveau national, dont 22 600 en Île-de-France, avec la localisation, la date, la gravité et les conditions. Le deuxième recense plus de 143 000 infrastructures cyclables avec le type de voie et la longueur. Le troisième contient des données de comptage horaire provenant de 69 compteurs automatiques. Et enfin, les données de population INSEE 2021 que nous avons ajoutées pour normaliser nos analyses.

---

## Slide 5 : Fusion et Création de Features (45 sec)
**Contenu :**
- Fusion au niveau communal (code INSEE)
- Dataset final : 1 124 communes
- Features créées : nb_accidents, nb_accidents_graves, longueur_totale_amenagements, ratio_pistes_cyclables, taux_risque_par_km, taux_risque_par_habitant, risque_eleve

**À dire :**
> J'ai ensuite fusionné ces quatre sources de données au niveau communal en utilisant le code INSEE comme clé de jointure. Après nettoyage et agrégation, on obtient un dataset de 1 124 communes. Nous avons créé plusieurs nouvelles features : le nombre d'accidents et d'accidents graves, la longueur totale des aménagements, le ratio de pistes cyclables séparées, et des taux de risque normalisés par kilomètre et par habitant.

---

## Slide 6 : Classification - Méthodologie (45 sec)
**Contenu :**
- Risque élevé : nb_accidents ≥ 75e percentile (≥ 6 accidents)
- 6 modèles : Régression Logistique, Random Forest, SVM (RBF), Gradient Boosting, XGBoost, LightGBM
- Validation croisée 5-fold
- Métriques : Accuracy, F1-Score, ROC-AUC

**À dire :**
> Pour la classification, nous avons défini une commune comme étant à risque élevé si son nombre d'accidents dépasse le 75e percentile, soit 6 accidents ou plus. Nous avons testé six algorithmes : la régression logistique, le Random Forest, le SVM avec noyau RBF, et trois méthodes de boosting. L'évaluation se fait par validation croisée 5-fold avec trois métriques : l'accuracy, le F1-score et le ROC-AUC.

---

## Slide 7 : Classification - Résultats (30 sec)
**Contenu :**
- Meilleur modèle : Régression Logistique (AUC 95.6%)
- Surprenant : modèle simple > modèles complexes
- Explication : relation essentiellement linéaire

**À dire :**
> Les résultats montrent que la régression logistique obtient les meilleures performances avec un AUC de 95.6%. C'est surprenant car un modèle simple surpasse les modèles complexes comme XGBoost. Cela s'explique par la nature linéaire de la relation : les communes à risque élevé sont celles avec beaucoup d'aménagements. Je passe maintenant la parole à David pour la suite des résultats.

---

# DAVID CHHOA (Slides 8-14) — ~4 minutes

---

## Slide 8 : Classification - Courbes ROC (30 sec)
**Contenu :**
- Image des courbes ROC
- Tous les modèles > 0.94 AUC
- Excellente séparation des classes

**À dire :**
> Merci Nicolas. Voici les courbes ROC de nos modèles de classification. On observe que tous les modèles atteignent un AUC supérieur à 0.94, ce qui indique une excellente capacité à distinguer les communes à risque élevé des autres. Les courbes sont très proches du coin supérieur gauche.

---

## Slide 9 : Classification - Matrice de Confusion (30 sec)
**Contenu :**
- Image de la matrice de confusion

**À dire :**
> Voici la matrice de confusion du meilleur modèle. On voit que le modèle fait peu d'erreurs, avec un bon équilibre entre les faux positifs et les faux négatifs. La précision est particulièrement bonne sur la classe majoritaire.

---

## Slide 10 : Classification - Features importantes (30 sec)
**Contenu :**
- Image des features importantes

**À dire :**
> Ce graphique montre l'importance des features pour la classification. Sans surprise, les variables liées aux aménagements cyclables et à la taille de la commune sont les plus importantes. Cela nous amène justement à notre analyse de régression.

---

## Slide 11 : Régression - Résultats Initiaux (45 sec)
**Contenu :**
- Objectif : prédire le nombre exact d'accidents
- Meilleur : Gradient Boosting (R² = 89.2%)
- Résultat apparemment excellent... mais attention !

**À dire :**
> Notre deuxième objectif était de prédire le nombre exact d'accidents par commune. Le Gradient Boosting, que Nicolas a implémenté, obtient un R² de 89.2%, ce qui signifie que le modèle explique près de 90% de la variance. À première vue, c'est un résultat excellent. Mais une analyse plus approfondie nous a conduits à remettre en question cette performance.

---

## Slide 12 : Régression - Corrélations (30 sec)
**Contenu :**
- Image de la matrice de corrélation

**À dire :**
> Cette matrice de corrélation révèle un premier problème. On observe une forte corrélation positive entre le nombre d'aménagements et le nombre d'accidents. Plus une commune a d'aménagements, plus elle a d'accidents. C'est contre-intuitif.

---

## Slide 13 : Analyse Critique - Corrélation Paradoxale (1 min)
**Contenu :**
- Plus d'aménagements → plus d'accidents (r = 0.60)
- Est-ce que les aménagements sont dangereux ? NON !
- Biais de confusion : Plus d'aménagements → plus de cyclistes → plus d'accidents
- Le modèle apprend cette corrélation, mais elle n'est pas causale

**À dire :**
> Analysons cette corrélation paradoxale. Avec un coefficient de 0.60, plus une commune investit dans les pistes cyclables, plus elle a d'accidents. Est-ce que cela signifie que les aménagements sont dangereux ? Bien sûr que non ! C'est un biais de confusion classique : les communes bien équipées attirent plus de cyclistes, ce qui augmente mécaniquement le nombre d'accidents en valeur absolue, même si le risque individuel diminue peut-être. Le modèle apprend cette corrélation, mais elle n'est pas causale.

---

## Slide 14 : Analyse Critique - Effet Paris et MAPE (1 min)
**Contenu :**
- Effet Paris : 20 arrondissements = 61% des accidents (13 853 sur 22 609)
- Le modèle apprend surtout à identifier Paris
- MAPE élevé : R² = 89% mais MAPE = 83%
- Prédictions imprécises pour les petites communes
- Distribution asymétrique : 33.6% des communes sans accident

**À dire :**
> Deuxième problème : l'effet Paris. Les 20 arrondissements de Paris concentrent à eux seuls 61% des accidents de toute l'Île-de-France, soit près de 14 000 accidents. Le modèle apprend donc essentiellement à identifier Paris, ce qui gonfle artificiellement le R².
> 
> Troisième problème : malgré ce R² de 89%, le MAPE, l'erreur moyenne en pourcentage, atteint 83%. Les prédictions sont donc très imprécises pour les petites communes, qui sont majoritaires. Enfin, un tiers des communes n'ont aucun accident enregistré. Je laisse maintenant Jérémie vous présenter notre solution à ces problèmes.

---

# JÉRÉMIE MASNOU (Slides 15-22) — ~4 minutes

---

## Slide 15 : Prédictions vs Valeurs Réelles (30 sec)
**Contenu :**
- Image : graphique global des prédictions

**À dire :**
> Merci David. Ce graphique illustre parfaitement les problèmes qu'il vient de décrire. On voit que le modèle prédit bien les valeurs élevées, notamment les arrondissements parisiens qui sont les points en haut à droite du graphique.

---

## Slide 16 : Prédictions vs Valeurs Réelles - Zoom (30 sec)
**Contenu :**
- Image : zoom sur les petites communes

**À dire :**
> Mais quand on zoome sur les communes avec moins de 100 accidents, qui représentent la grande majorité, on observe une dispersion importante autour de la diagonale. Le modèle est donc imprécis là où on en aurait le plus besoin.

---

## Slide 17 : Solution - Taux de Risque Normalisés (1 min)
**Contenu :**
- Intégration des données de population INSEE
- Taux par km = Accidents / Longueur aménagements (km) → Compare indépendamment de l'équipement
- Taux pour 10k hab = (Accidents / Population) × 10 000 → Métrique épidémiologique
- Objectif : éliminer l'effet de taille

**À dire :**
> Face à ces biais, j'ai proposé une nouvelle approche. En intégrant les données de population INSEE, nous avons défini deux nouvelles métriques de risque normalisées. Le taux par kilomètre divise le nombre d'accidents par la longueur totale des aménagements, ce qui permet de comparer les communes indépendamment de leur niveau d'équipement. Le taux pour 10 000 habitants est une métrique classique en épidémiologie qui normalise par la population. L'objectif est d'éliminer l'effet de taille qui biaisait nos résultats.

---

## Slide 18 : Taux de Risque - Résultats (45 sec)
**Contenu :**
- Performances nettement inférieures mais plus réalistes
- Le modèle ne peut plus exploiter l'effet de taille
- Paris n'a plus un poids disproportionné
- Reflète la complexité du problème

**À dire :**
> Les résultats avec les taux normalisés sont nettement inférieurs : un R² maximum de 30% pour le taux par kilomètre et 19% pour le taux par habitant. Mais loin d'être un échec, ces résultats sont plus réalistes. Le modèle ne peut plus "tricher" en identifiant simplement les grandes communes. Paris n'a plus un poids disproportionné. Ces performances modestes reflètent la complexité réelle du risque cycliste.

---

## Slide 19 : Synthèse Comparative (45 sec)
**Contenu :**
- Tableau récapitulatif avec validité
- Paradoxe : meilleure performance ≠ meilleur modèle
- Un R² élevé ne garantit pas la validité
- L'analyse critique est indispensable

**À dire :**
> Ce tableau résume l'ensemble de nos résultats. La classification fonctionne bien avec un AUC de 96%. La régression brute affiche un R² de 89%, mais sa validité est limitée. Les approches par taux normalisés ont des performances plus modestes mais une validité élevée. Le paradoxe de notre étude est que la meilleure performance correspond à l'approche la moins valide. C'est un rappel important : un bon score ne garantit pas un bon modèle.

---

## Slide 20 : Limites de l'Étude (30 sec)
**Contenu :**
- Sous-déclaration des accidents
- Couverture limitée : 69 compteurs pour toute l'IDF
- Variables manquantes : trafic automobile, urbanisme, météo
- 33.6% de communes sans accident → modèles Zero-Inflated ?

**À dire :**
> Notre étude présente plusieurs limites. Tous les accidents ne sont pas déclarés, notamment les accidents mineurs. Nous n'avons que 69 compteurs pour estimer le trafic cycliste. Il nous manque des variables importantes comme le trafic automobile ou l'urbanisme. Et un tiers des communes sans accident pourrait nécessiter des modèles spécifiques comme les modèles Zero-Inflated.

---

## Slide 21 : Conclusion (45 sec)
**Contenu :**
- ✓ Classification efficace pour identifier les communes prioritaires
- ⚠ Régression brute biaisée par l'effet de taille
- ✓ Taux normalisés : approche plus réaliste
- Enseignement principal : Performance ≠ Validité
- L'analyse critique des données est essentielle

**À dire :**
> En conclusion, la classification binaire du risque fonctionne bien et peut servir à identifier les communes prioritaires pour des interventions de sécurité. La régression sur le nombre brut, malgré son R² élevé, est biaisée par l'effet de taille. Notre approche par taux normalisés est plus réaliste mais révèle la complexité du problème. Le principal enseignement de ce projet est qu'une bonne performance ne garantit pas la validité d'un modèle. L'analyse critique des données est essentielle.

---

## Slide 22 : Questions (15 sec)
**Contenu :**
- Merci pour votre attention
- Questions ?

**À dire :**
> Merci pour votre attention. Nous sommes maintenant disponibles pour répondre à vos questions.

---

# QUESTIONS POTENTIELLES ET RÉPONSES

**Q : Pourquoi la régression logistique surpasse les modèles complexes ?**
> La relation entre les features et le risque est essentiellement linéaire. Les communes à risque élevé sont celles avec beaucoup d'aménagements, car elles attirent plus de cyclistes. Un modèle linéaire capture bien cette relation.

**Q : Comment avez-vous géré le déséquilibre des classes ?**
> Nous avons utilisé la pondération des classes (class_weight='balanced') et la stratification lors du split train/test pour maintenir les proportions.

**Q : Pourquoi ne pas exclure Paris de l'analyse ?**
> C'est une piste intéressante qu'on aurait pu explorer. On pourrait développer un modèle spécifique pour Paris et un autre pour le reste de l'IDF.

**Q : Quelles données supplémentaires amélioreraient les modèles ?**
> Le trafic automobile serait crucial, ainsi que les caractéristiques urbanistiques (densité du bâti, présence de commerces), la météo détaillée, et surtout le trafic cycliste réel qui n'est mesuré que par 69 compteurs.

**Q : Comment interprétez-vous la corrélation paradoxale ?**
> C'est un biais de confusion classique en épidémiologie. Les communes qui investissent dans les aménagements attirent plus de cyclistes, donc plus d'accidents en absolu. Le risque individuel pourrait même diminuer.

**Q : Pourquoi utiliser le 75e percentile pour définir le risque élevé ?**
> C'est un seuil courant qui permet d'identifier le quartile supérieur de la distribution. Cela donne environ 25% de communes à risque élevé, un ratio raisonnable pour la classification.

**Q : Les modèles Zero-Inflated auraient-ils amélioré les résultats ?**
> Probablement oui, car 33.6% des communes n'ont aucun accident. Ces modèles traitent séparément la probabilité d'avoir zéro et la distribution des valeurs positives.

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
