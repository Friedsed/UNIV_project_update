# -*- coding: utf-8 -*-
"""
L3 ME
Projet "balistique"

@author: C. Airiau
@date: 30/10/2023

Partie 1: apprentissage sur le modèle analytique, Model_1

"""
import numpy as np  # module de math
import matplotlib.pyplot as plt  # module graphique
from scipy.constants import g    # constante en m/s^2.
from .import colored_messages as cm
from .import constantes as cs

class Model_1(object):
    """ Class of the analytical model"""
    def __init__(self, h=10, v_0=10, alpha=30):
        """ Le constructeur de classe est lancé dès la création de la classe"""
        self.h = h
        self.v_0 = v_0
        self.alpha = np.deg2rad(alpha)  # on met alpha en radians directement
        #self.initial_message()
        self.impact_values = None
        self.time, self.x, self.z = None, None, None
        self.v_x, self.v_z, self.v = None, None, None
        self.t_1, self.h_1 = None, None


    def initial_message():
        set_title("Création d'une instance du modèle analytique (class initiation)")

    # SETTERS

    def update_parameters(self, params_dict):
        """ set new values for parameters from a dictionary"""
        if params_dict is None:
            params_dict = dict(v_0=10, h=20, alpha=30)
        else:
            for key in params_dict.keys():
                if key == "v_0":
                    self.v_0 = params_dict[key]
                elif key == "h":
                    self.h = params_dict[key]
                elif key == "alpha":
                    self.alpha = np.deg2rad(params_dict[key])
        self.get_parameters()

    def set_velocity(self, t):
        v_z = self.v_0 * np.sin(self.alpha) - g * t
        v_x = self.v_0 * np.cos(self.alpha)
        v = np.sqrt(v_x ** 2 + v_z ** 2)
        return v_x, v_z, v

    def set_impact_values(self):
        """
        retourne les valeurs temps, portée, vitesse et angle à l'impact
        """
        var = self.v_0 * np.sin(self.alpha)
        t_i = (var + np.sqrt(var ** 2 + 2 * g * self.h)) / g
        v_x, v_z, v = self.set_velocity(t_i)
        x_i = v_x * t_i  # ici des variables locales, pas de self devant
        theta_i = np.rad2deg(np.arctan(v_z / v_z))
        self.impact_values = {"t_i": t_i, "p": x_i, "angle": theta_i, "v": [v_x, v_z, v]}
        return self.impact_values

    # GETTERS
    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        set_info("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])

    def get_parameters(self):
        """
        Affichage formatté des paramètres
        """
        set_info("Parameters:")
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))

##############################################################JE COMPLETE ICI###########################################################    



    def set_trajectory (self,t_end,npt ):    # t_end is a nombers and npt is number
        self.time=np.linspace(0,t_end,npt)
        self.x = self.v_0*np.cos(self.alpha)*self.time
        self.z= -(g*0.5*(self.time)**2) + self.v_0 * self.time*np.sin(self.alpha) + self.h
        self.v_x= self.v_0*np.cos(self.alpha)
        self.v_z= self.v_0 * np.sin(self.alpha) - g*self.time
        self.v= np.sqrt(self.v_x**2 + self.v_z**2)


    def plot_trajectory(self):
        plt.plot(self.x, self.z, marker="+", color="red",linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.show ()

    def plot_component(self):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)        
        fig.suptitle("Composantes")
        ax1.plot(self.time, self.v_z, marker="+",color="red",linewidth=3)
        ax1.fill_between(self.time, self.v_z, alpha=0.3, color='red')
        ax1.set_ylabel("$v_z$")
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(self.time, self.v, marker="o",color="green",linewidth=3)
        ax2.fill_between(self.time,self.v, alpha=0.3, color='green')
        ax2.set_ylabel("$v$")
        ax2.set_xlabel("time (s)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # ajuste les marges
        plt.show()



    def set_trajectories(self):

        list_alpha=[]
        list_x=[]
        list_z=[]
        list_v_x=[]
        list_v_z=[]
        list_v=[]
        x=z=v_x=v_z=v=0

        for i in range (20,71):
            if i%5==0:
                list_alpha.append(i)
        list_alpha=np.deg2rad(list_alpha)

        for i in list_alpha:
            
            x = self.v_0*np.cos(i)*self.time
            z= -(g*0.5*(self.time)**2) + self.v_0 * self.time*np.sin(i) + self.h
            v_x= self.v_0*np.cos(i)
            v_z= self.v_0 * np.sin(i) - g*self.time
            v= np.sqrt(self.v_x**2 + self.v_z**2)
            list_x.append(x)
            list_z.append(z)
            list_v_x.append(v_x)
            list_v_z.append(v_z)
            list_v.append(v)

        return [list_x , list_z, list_v_x, list_v_z, list_v]
    
    def plot_trajectories(self, liste):
         
        plt.figure(figsize=(10, 6)) # Crée une nouvelle figure
        xliste= liste[0]
        zliste= liste[1] 
        for i in range(len(xliste)):
            X=xliste[i]
            Z=zliste[i]
            plt.plot(X, Z)

        # --- Mise en forme finale du graphique ---
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Z (m)")
        plt.title("FIGURE 3 – Différentes trajectoires en fonction de α")
        plt.legend(title="Angle de lancement", loc='upper right')
        plt.grid(True, linestyle='--')
        # plt.ylim(bottom=0) # Assure que l'axe Z commence à zéro
        plt.show()




