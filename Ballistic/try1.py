












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

import colored_messages as cm
import constantes as cs 

class Model_3(object):
    def __init__(self, params):
        """ Le constructeur de classe est lancé dès la création de la classe"""
        self.h = params["h"]
        self.v_0 = params["v_0"]
        self.alpha = np.deg2rad(params["alpha"])
        self.npt = params["npt"]
        #______________________________________________________ ADD BY ME ______________________________________________
        self.mass=params["mass"]
        self.rho=params["rho"]
        self.Cd= params["Cd"]
        self.area= params["area"]
        self.Cl= params["Cl"]
        self.a= params["a"]
        #______________________________________________________ END
        #self.initial_message()

        self.t, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.Cx, self.Cz = None, None
        self.impact_values = None
    #_________________________________________________________________________________________________________________________+++++++++++++++++++++++++++++++++++++++probleme a corriger avec l angle teta +++++++++++++++++ semble corrige dernier minute
        self.T0= self.a*self.mass * g
        self.beta= (self.rho*self.area)/(2*self.mass)
        self.Ct= self.T0 / self.mass
    #_________________________________________________________ END
        

    @staticmethod
    def initial_message():
       cm.set_title("Création d'une instance du modèle  ODE (exemple d'apprentissage)")

    def ode(self, y,t):
        #ODE to solve
        #dy/dt = f(t, y)
        #y = [self.x, self.z, self.v_x, self.v_z]
        dy = np.zeros(4)
        dy[0] = y[2]  # dx / dt = v_x
        dy[1] = y[3]  # dz / dt = v_z
        v2= y[2]**2 + y[3]**2
        teta=np.arctan(y[3]/y[2])
        self.Cx=-self.Cd*np.cos(teta)- self.Cl* np.sin(teta)
        self.Cz= self.Cl*np.cos(teta)-self.Cd* np.sin(teta)
        dy[2] = self.beta*v2* self.Cx +  self.Ct * np.cos(teta)    # dv_x / dt = 0
        dy[3] = -g + self.beta * v2 * self.Cz +  self.Ct * np.sin(teta)   # dv_z / dt = - g
  

        return dy
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

    def solve_trajectory(self, alpha=30, t_end=1):
        
        self.t = np.linspace(0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)

        # initial condition
        y_init = [  0   , self.h    ,   self.v_0 * np.cos(self.alpha)   ,   self.v_0 * np.sin(self.alpha)   ]
 
        y = odeint(self.ode, y_init, self.t)     

        self.x  ,   self.z  ,   self.v_x    ,   self.v_z    =       y[:, 0]   ,   y[:, 1] ,   y[:, 2] ,   y[:, 3]

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

    def plot_trajectory(self):
    
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
    def set_reference_solution(self, t):
        x = self.v_0 * np.cos(self.alpha) * t
        z = - g / 2 * t ** 2 + self.v_0 * np.sin(self.alpha) * t + self.h
        return x, z
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

    def validation(self,t_end,npt):

        cm.set_msg("Validation")
        print("analytical solution at t = %f" % self.t[-1])
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x, z                       : %f  %f" % (x_ref, z_ref))
        print("numerical solution at the same time:")
        print("x, z                       : %f  %f" % (self.x[-1], self.z[-1]))


        ################################### COMPLETER PAR MOI     ################################################
        self.time=np.linspace(0,t_end,npt)
        x , z= self.set_reference_solution (self.time)
     
        ecart=[ np.max( np.abs(x-self.x) ) , np.max( np.abs (z-self.z) ) ] # np.max( np.abs(v_x-self.v_x) ), np.max( np.abs(v_z-self.v_z) )   ]
       
        if np.max(ecart) < 1e-7:
            print(" La validation est vrai")
        else:
            print(" Pas de validation")

        print("l erreur max est ", np.max(ecart))
        plt.plot(self.x, self.z, marker="+", color="red", markersize = 12, linewidth=3)
        plt.plot(x, z, marker="+", color="green",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()

        ################################## FIN ################################################################################
      

    

    def set_impact_values(self):
        """
        partie à modifier par les étudiants

        méthode: trouver le temps d'impact t_i tel que v_z est juste au dessus de 0.
        v_z(t_{i+1}) < 0. Puis remplir v_x, v_z, v, theta_i, x_i à cet instant d'impact.
        """
        # partie à coder
        ######################################################## COMPLETER PAR MOI     ################################################
        

        # 1) Trouver le premier indice où z devient négatif
        n = None
        for i in range(len(self.z) -1):
            if self.z[i] > 0 and self.z[i+1] < 0:
                n = i
                break

        if n is None:
            raise ValueError("Aucun impact détecté : z ne devient jamais négatif.")

            # 2) Calcul du coefficient d’interpolation

        def interpo(a, n, u):
            return u[n] + a * (u[n + 1] - u[n])

        a = - self.z[n] / (self.z[n+1] - self.z[n])


        t_i  = interpo(a, n, self.t)
        x_i  = interpo(a, n, self.x)
        z_i  = interpo(a, n, self.z)
        vx_i = interpo(a, n, self.v_x)
        vz_i = interpo(a, n,self. v_z)
        #v_i = interpo(a, n, self.v)

        # angle d’impact
        theta_i = np.rad2deg(np.arctan(vz_i / vx_i))

        # 4) Stockage des valeurs
        self.impact_values = {"t_i": t_i, "p": x_i, "angle": np.rad2deg(theta_i), "v": [ vx_i, vz_i]}


        # ##########################################################################################################################


        # résultat à conserver:
        
        return self.impact_values

    def get_parameters(self):
        """
        Affichage formatté des paramètres
        """
        set("Parameters:")
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))
        print("mass       : %.2f kg" % self.mass)
        print("rho        : %.2f kg/m^3" % self.rho)
        print("area       : %.2f m^2" % self.area)
        print("Cd         : %2f "% self.Cd)
        print("Cl         : %2f "% self.Cl)

    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])
