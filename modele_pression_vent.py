"""Modèle de calcul de la pression du vent sur une porte vitrée.

Le modèle utilise la pression dynamique:
    q = 0.5 * rho * V^2
et la force:
    F = q * C_d * A

Où:
- rho: masse volumique de l'air (kg/m³)
- V: vitesse du vent (m/s)
- C_d: coefficient de traînée (sans unité)
- A: surface exposée de la porte (m²)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PorteVitree:
    """Caractéristiques géométriques d'une porte vitrée."""

    largeur_m: float
    hauteur_m: float

    @property
    def surface_m2(self) -> float:
        """Retourne la surface de la porte en m²."""
        return self.largeur_m * self.hauteur_m


@dataclass(frozen=True)
class ConditionsVent:
    """Paramètres physiques du vent."""

    vitesse_m_s: float
    masse_volumique_air: float = 1.225
    coefficient_trainee: float = 1.2


def calculer_pression_vent(conditions: ConditionsVent) -> float:
    """Calcule la pression exercée par le vent en Pascal (N/m²)."""
    return 0.5 * conditions.masse_volumique_air * conditions.vitesse_m_s**2


def calculer_force_sur_porte(porte: PorteVitree, conditions: ConditionsVent) -> float:
    """Calcule la force totale du vent sur la porte en Newton (N)."""
    pression = calculer_pression_vent(conditions)
    return pression * conditions.coefficient_trainee * porte.surface_m2


def exemple_utilisation() -> None:
    """Exemple simple d'utilisation du modèle."""
    porte = PorteVitree(largeur_m=0.9, hauteur_m=2.1)
    vent = ConditionsVent(vitesse_m_s=30.0)

    pression_pa = calculer_pression_vent(vent)
    force_n = calculer_force_sur_porte(porte, vent)

    print(f"Surface porte : {porte.surface_m2:.2f} m²")
    print(f"Pression du vent : {pression_pa:.1f} Pa")
    print(f"Force sur la porte : {force_n:.1f} N")


if __name__ == "__main__":
    exemple_utilisation()
