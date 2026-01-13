# Analyse Critique des Résultats

## 🔴 Problèmes Majeurs

### 1. MAPE extrêmement élevé (83-145%)

Le MAPE de 83% même pour le meilleur modèle signifie qu'en moyenne, on se trompe de 83% sur la valeur réelle.

**Cause** : beaucoup de communes ont 0-1 accidents (médiane = 1), donc une erreur de 2 accidents sur une commune avec 1 accident = 200% d'erreur.

### 2. Distribution très déséquilibrée de la cible

```
mean = 20, median = 1, max = 1086
75% des communes ont ≤ 6 accidents
```

- Le R² de 89% est gonflé par les outliers (Paris avec 1086 accidents)
- Le RMSE de 37 accidents est énorme comparé à la médiane de 1

### 3. Corrélation paradoxale aménagements ↔ accidents (+0.84)

Plus d'aménagements = plus d'accidents ? C'est contre-intuitif !

**Explication** : c'est un effet de confusion. Plus de vélos → plus d'aménagements → plus d'accidents en valeur absolue. Le modèle prédit en fait le **trafic cycliste**, pas le **risque**.

---

## 🟠 Données Manquantes

| Variable manquante | Impact |
|-------------------|--------|
| **Population** par commune | Impossible de calculer un taux |
| **Trafic cycliste** | Seulement 69 compteurs pour 1124 communes |
| **Surface/densité** | Communes rurales vs urbaines non différenciées |
| **Trafic automobile** | Facteur de risque majeur ignoré |

---

## 🟡 Choix Méthodologiques Discutables

### 1. Variable cible mal choisie

On prédit le **nombre brut** d'accidents, pas le **taux de risque**.

Il faudrait : 
- accidents / km d'aménagement
- accidents / 1000 cyclistes

### 2. Seuil de "risque élevé" arbitraire

Le 75e percentile = 6 accidents → pas de justification métier.

Une commune avec 5 accidents sur 500 cyclistes est plus dangereuse qu'une avec 10 accidents sur 10 000.

### 3. Pas de dimension temporelle

- Accidents cumulés sur ~20 ans vs aménagements actuels
- Un aménagement de 2024 ne peut pas expliquer un accident de 2005

### 4. Autocorrélation spatiale ignorée

- Les communes voisines sont probablement similaires
- Le split train/test aléatoire peut surestimer les performances

---

## 🟢 Ce qui fonctionne

- Les modèles convergent (pas d'overfitting flagrant, CV proche du test)
- La méthodologie de comparaison est rigoureuse
- Les visualisations sont claires

---

## 💡 Améliorations Possibles

1. **Normaliser la cible** : accidents par km d'aménagement ou par habitant
2. **Log-transformer** nb_accidents pour gérer l'asymétrie
3. **Ajouter des données externes** : population INSEE, densité
4. **Créer un vrai indicateur de risque** : `(nb_accidents / longueur_amenagements) × 1000`
5. **Validation spatiale** : train sur certains départements, test sur d'autres
6. **Régression Poisson/Négative Binomiale** : plus adaptée aux comptages
