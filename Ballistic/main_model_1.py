# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

@author: C. Airiau
@date: 30/10/2023

Partie 1: Travail sur le modèle analytique

task :
    0 : graphiques: trajectoire, v_z et v, f(alpha)
        valeurs à l'impact
    1 : ajout fonction plot_trajectories, plot_maximun_distance, plot_maximum_height dans le script principal
        affichage de la portée maximale et de l'angle correspondant + valeur théorique
"""

import numpy as np  # module de math
import matplotlib.pyplot as plt  # module graphique
from scipy.constants import g    # constante en m/s^2.
from Ballistic.model_1 import Model_1
#from .import colored_messages as cm
#from .import constantes as cs

task = 0

if task == 0:

    model = Model_1(h=20, v_0=50, alpha=40)
        


    immpact_values = model.set_impact_values()
    model.set_trajectory(immpact_values["t_i"],50)     # reponse a la question 9.2)2 
   # figure1
    model.plot_component()                  # reponse a la question 9.2)3
    model.plot_trajectory()
    h=model.set_trajectories()               # reponse a la question 10.1)1
    print("les valeurs sont",h[0][0])        # reponse a la question 10.1)1
    model.plot_trajectories(model.set_trajectories())           # reponse a la question 10.1)2


    
   

print("\nGraphique de la trajectoire", model.set_impact_values()    )
#     pass

# elif task == 1:
#     pass
# plt.show()
# print("normal end of execution")
