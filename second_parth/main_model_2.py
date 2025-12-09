# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

@author: C. Airiau
@date: 30/10/2023

Partie 2: résolution ODE du problème balistique numériquement
         ODE simple, uniquement avec la force de gravité
         Modèle 2

task :
    1 : test de la classe Analytical Model
    2 : ajout du tracé de la trajectoire, de la validation, du calcul d'impact précis


"""
import numpy as np                      # module de math
import matplotlib.pyplot as plt         # module graphique
from model_2 import Model_2

task = 0

if task == 0:         # test of AnalyticalModel
    t_end, alpha_ref = 3, 20
    model_2 = Model_2({"v_0": 20, "h": 20, "alpha": 40, "npt": 101})
    model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    print("solution:", model_2.x, model_2.z)
    plt.figure(0)
    model_2.validation(3, 101)
    plt.figure(1)
    model_2.plot_trajectory(3,40)

elif task == 1:         # test of AnalyticalModel
    t_end = 3
    alpha_ref = 20
    model_2 = Model_2({"v_0": 20, "h": 20, "alpha": 40, "npt": 101})
    model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    # Décommenter au fur et à mesure des implémentations.
    # model_2.plot_trajectory()
    # model_2.validation()
    # impact = model_2.set_impact_values()
    # model_2.get_impact_values()

plt.show()
print("normal end of execution")
