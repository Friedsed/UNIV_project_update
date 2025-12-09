
<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from model_1 import Model_1
#from Ballistic.colored_messages import *
#from Ballistic.constantes import *
=======
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
from model_1 import Model_1
#from .import colored_messages as cm
#from .import constantes as cs
>>>>>>> fc8536589f26adf32b86ec3f4b0ba62d1694c50d

task = 0

if task == 0:

    model = Model_1(h=20, v_0=50, alpha=40)
        

#__________________________________________________________________________________________________
    impact_values = model.set_impact_values()
    model.set_trajectory(impact_values["t_i"],50)     # reponse a la question 9.2)2 

#__________________________________________________________________________________________________
   # figure1
    model.plot_component()                  # reponse a la question 9.2)3
    model.plot_trajectory()


task=1

if task==1:

    model = Model_1(h=20, v_0=50, alpha=40)
    alphas = np.arange(20, 71, 5)  # Angles de 20Â° Ã  70Â° par pas de 5Â°
#_________________________________________________________________________________________________
        # Trajectoires
    trajectoriess=model.set_trajectories(alphas)
#__________________________________________________________________________________________________
    x_list, z_list = trajectoriess[0], trajectoriess[1]  # reponse a la question 10.1)1

#_________________________________________________________________________________________________
        # TracÃ© des trajectoires
    model.plot_trajectories(x_list,z_list,alphas)  # reponse a la question 10.1)2

#_________________________________________________________________________________________________                      
     # PortÃ©e maximale
    model.plot_maximum_distance(alphas)
#_________________________________________________________________________________________________
        # Angle optimal
    optimal_alpha, optimal_distance = model.find_optimal_angle(alphas)
    print(f"Angle optimal: {optimal_alpha}Â°, Portée maximale: {optimal_distance:.2f} m")

#_________________________________________________________________________________________________
        # Altitude maximale
    model.plot_maximum_height(alphas)

    
   