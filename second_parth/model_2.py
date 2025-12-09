# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

@author: C. Airiau
@date: 30/10/2023

Partie 2: résolution ODE du problème balistique
Implémentation du modèle 1

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g    # constante en m/s^2.
from scipy.integrate import odeint

#from Ballistic.colored_messages import *
#from Ballistic.constantes import *


class Model_2(object):
    def __init__(self, params):
        """ Le constructeur de classe est lancé dès la création de la classe"""
        self.h = params["h"]
        self.v_0 = params["v_0"]
        self.alpha = np.deg2rad(params["alpha"])
        self.npt = params["npt"]
        #self.initial_message()

        self.t, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.impact_values = None
        self.g=9.8

    @staticmethod
    #def initial_message():
     #   set_title("Création d'une instance du modèle  ODE (exemple d'apprentissage)")

    def ode( y,t):
        #ODE to solve
        #dy/dt = f(t, y)
        #y = [self.x, self.z, self.v_x, self.v_z]
        dy = np.zeros(4)
        dy[0] = y[2]  # dx / dt = v_x
        dy[1] = y[3]  # dz / dt = v_z
        dy[2] = 0     # dv_x / dt = 0
        dy[3] = -10   # dv_z / dt = - g

        return dy


    def solve_trajectory(self, alpha=30, t_end=1):
        """
        trajectory in a gravity field
        t_end is an approximation of the trajectory duration

        On préfère ici mettre alpha et t_end en paramètres plutôt que d'utiliser l'attribut self.alpha.
        c'est plus clair pour les études paramétriques ultérieures
        """
        self.t = np.linspace(0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)
        # initial condition
        y_init = [0, self.h, self.v_0 * np.cos(self.alpha), self.v_0 * np.sin(self.alpha)]
        y = odeint (self.ode, y_init, t=self.t)       # résolution de l'ode
       # y = odeint(lambda y, t: self.ode(y, t), y_init, self.t, full_output=False)
        self.x, self.z, self.v_x, self.v_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    def plot_trajectory(self, t_end, alpha):
        """
        dessin de la trajectoire
        Ajout des étudiants
        """
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()
        #print("Récupérer la méthode de la partie 1 et la recopier")

    def validation(self,t_end,npt):
        """
        on se contente de calculer l'erreur sur la position du dernier point.
        ajout des étudiants
        """
        #set_msg("Validation")
        print("analytical solution at t = %f" % self.t[-1])
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x, z                       : %f  %f" % (x_ref, z_ref))
        print("numerical solution at the same time:")
        print("x, z                       : %f  %f" % (self.x[-1], self.z[-1]))
        
        # Ajouter le calcul de l'erreur et l'affichage 
         

        ################################### COMPLETER PAR MOI################################################
        self.time=np.linspace(0,t_end,npt)
        x = self.v_0*np.cos(self.alpha)*self.time
        z= -(g*0.5*(self.time)**2) + self.v_0 * self.time*np.sin(self.alpha) + self.h
        v_x= self.v_0*np.cos(self.alpha)
        v_z= self.v_0 * np.sin(self.alpha) - g*self.time
        v= np.sqrt(self.v_x**2 + self.v_z**2)
        #solanal=[x,z,v_x,v_z,v]
        #solnum=[self.x, self.z, self.v_x, self.v_z]
        ecart=[ np.max( np.abs(x-self.x) ) , np.max( np.abs (z-self.z) ) ] # np.max( np.abs(v_x-self.v_x) ), np.max( np.abs(v_z-self.v_z) )   ]
       
        if np.max(ecart) < 10e-7:
            print(" La validation est vrai")
        else:
            print(" Pas de validation")

        print("l erreur max est ", np.max(ecart))
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.plot(x, z, marker="+", color="green",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()

        ################################## FIN ################################################################################
      

    def set_reference_solution(self, t):
        x = self.v_0 * np.cos(self.alpha) * t
        z = - g / 2 * t ** 2 + self.v_0 * np.sin(self.alpha) * t + self.h
        return x, z

    def set_impact_values(self):
        """
        partie à modifier par les étudiants

        méthode: trouver le temps d'impact t_i tel que v_z est juste au dessus de 0.
        v_z(t_{i+1}) < 0. Puis remplir v_x, v_z, v, theta_i, x_i à cet instant d'impact.
        """

        def interpo(a, n, u):
            return u[n] + a * (u[n + 1] - u[n])

        n = 0
        
        # partie à coder
        
        # résultat à conserver:
        self.impact_values = {"t_i": t_i, "p": x_i, "angle": np.rad2deg(theta_i), "v": [v_x, v_z, v]}
        return self.impact_values

    def get_parameters(self):
        """
        Affichage formatté des paramètres
        """
        set("Parameters:")
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))

    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])
