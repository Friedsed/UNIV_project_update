# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

@author: C. Airiau
@date: 30/10/2023

Partie 1: apprentissage sur le modèle analytique, modèle 4

"""
from Ballistic.analytical_model_example import AnalyticalModelExample


# 1 - Initialisation des paramètres et instanciation
v_0, h, alpha = 20, 20, 40
modele_analytique = AnalyticalModelExample(v_0=v_0, alpha=alpha, h=h)
# 2 - Calcul des quantités à l'impact et affichages
impact = modele_analytique.set_impact_values()
print("impact values  :", impact)                # mauvaise méthode d'affichage
modele_analytique.get_impact_values()            # bonne méthode d'affichage
# 3 - Modification des paramètres, deux approches
modele_analytique.v_0 = 100                     # pas conseillé
modele_analytique.update_parameters({"h": 100, "alpha": 30})  # mieux
modele_analytique.set_impact_values()     # nouveau calcul d'impact
modele_analytique.get_impact_values()

print("normal end of execution")
