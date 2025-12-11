# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g
from scipy.integrate import odeint

import colored_messages as cm
import constantes as cs 


class Model_3(object):
    def __init__(self, params):
        self.h = params["h"]
        self.v_0 = params["v_0"]
        self.alpha = np.deg2rad(params["alpha"])
        self.npt = params["npt"]

        self.mass = params["mass"]
        self.rho = params["rho"]
        self.Cd = params["Cd"]
        self.Cl = params["Cl"]
        self.area = params["area"]
        self.a = params["a"]

        self.t, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.Cx, self.Cz = None, None
        self.impact_values = None

        self.T0 = self.a * self.mass * g
        self.beta = (self.rho * self.area) / (2 * self.mass)
        self.Ct = self.T0 / self.mass

        

    @staticmethod
    def initial_message():
       cm.set_title("Création d'une instance du modèle  ODE (exemple d'apprentissage)")

    def ode(self, y, t):
        dy = np.zeros(4)
        dy[0] = y[2]
        dy[1] = y[3]
        v2 = y[2]**2 + y[3]**2
        theta = np.arctan2(y[3], y[2])

        Cx = -self.Cd*np.cos(theta) - self.Cl*np.sin(theta)
        Cz =  self.Cl*np.cos(theta) - self.Cd*np.sin(theta)

        dy[2] = self.beta * v2 * Cx + self.Ct * np.cos(theta)
        dy[3] = -g + self.beta * v2 * Cz + self.Ct * np.sin(theta)
        return dy


    def solve_trajectory(self, alpha=30, t_end=1):
       
        self.t = np.linspace(0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)
        # initial condition
        y_init = [0, self.h, self.v_0 * np.cos(self.alpha), self.v_0 * np.sin(self.alpha)]
        y = odeint (self.ode, y_init, t=self.t)       # résolution de l'ode
       # y = odeint(lambda y, t: self.ode(y, t), y_init, self.t, full_output=False)
        self.x, self.z, self.v_x, self.v_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    def plot_trajectory(self):
      
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()
        #print("Récupérer la méthode de la partie 1 et la recopier")



#_____________________________________________________________________________________________ LA VALIDATION AFFICE LA SOLUTION ANALYTIQUE ET LA SOLUTION NUMERIQUE, AFFICHE L ERREU PUIS TRACE LA COURBE ANALYTIQUE SUPPERSPOSE AVEC LA COURBE  NUMERIQUE 
    def validation(self,t_end,npt):
      
        #set_msg("Validation")

        #______________________________________________________________________________________________________________________________________ AFFICE LA SOLUTION ANALYTIQUE ET LA SOLUTION NUMERIQUE JUSTE LES COORDONNES DES POINTS D IMPACT
        cm.set_msg("Validation")
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
       

        ecart=[ np.max( np.abs(x-self.x) ) , np.max( np.abs (z-self.z) ) ] 
       
        if np.max(ecart) < 10e-7:
            print(" La validation est vrai")
        else:
            print(" Pas de validation")
        #_______________________________________________________________________________________________________________________________________ AFFICHE L ERREUR
        print("l erreur max est ", np.max(ecart))

        #_______________________________________________________________________________________________________________________________________ TRACE LA COURBE ANALYTIQUE SUPPERSPOSE AVEC LA COURBE  NUMERIQUE 
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.plot(x, z, marker="+", color="green",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.grid()
        plt.show ()

        ################################## FIN ################################################################################
      

    def set_reference_solution(self, t):
        x = self.v_0 * np.cos(self.alpha) * t
        z = - g / 2 * t ** 2 + self.v_0 * np.sin(self.alpha) * t + self.h
        return x, z

    def set_impact_values(self):
      
        def interpo(a, n, u):
            return u[n] + a * (u[n + 1] - u[n])

        n = 0
        
        # partie à coder
        
        # résultat à conserver:
        self.impact_values = {"t_i": t_i, "p": x_i, "angle": np.rad2deg(theta_i), "v": [v_x, v_z, v]}
        return self.impact_values

    def get_parameters(self):
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))
        print("mass       : %.2f kg" % self.mass)
        print("rho        : %.2f kg/m^3" % self.rho)
        print("area       : %.2f m^2" % self.area)
        print("Cd         : %.2f" % self.Cd)
        print("Cl         : %.2f" % self.Cl)

    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])
