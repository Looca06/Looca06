# Modèle pression du vent sur porte vitrée

Ce dépôt contient un petit modèle Python pour estimer :

- la **pression dynamique du vent** (en Pascal),
- la **force totale exercée** sur une porte vitrée (en Newton).

## Formules utilisées

- Pression : `q = 0.5 × ρ × V²`
- Force : `F = q × C_d × A`

Avec :

- `ρ` : masse volumique de l'air (kg/m³),
- `V` : vitesse du vent (m/s),
- `C_d` : coefficient de traînée,
- `A` : surface de la porte (m²).

## Fichier principal

- `modele_pression_vent.py`

## Exécution

```bash
python3 modele_pression_vent.py
```

Exemple de sortie :

```text
Surface porte : 1.89 m²
Pression du vent : 551.2 Pa
Force sur la porte : 1250.2 N
```
