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

        # listes pour stocker les trajectoires multiples
        self.list1 = [None, None]
        self.list2 = [None, None]
        self.list3 = [None, None]

        self.T0 = self.a * self.mass * g
        self.beta = (self.rho * self.area) / (2 * self.mass)
        self.Ct = self.T0 / self.mass

    #____________________________________________________________________________________________

    def update_param(self, param1):
        self.h = param1["h"]
        self.v_0 = param1["v_0"]
        self.alpha = np.deg2rad(param1["alpha"])
        self.npt = param1["npt"]
        self.mass = param1["mass"]
        self.rho = param1["rho"]
        self.Cd = param1["Cd"]
        self.Cl = param1["Cl"]
        self.area = param1["area"]
        self.a = param1["a"]

        self.T0 = self.a * self.mass * g
        self.beta = (self.rho * self.area) / (2 * self.mass)
        self.Ct = self.T0 / self.mass

    #____________________________________________________________________________________________

    @staticmethod
    def initial_message():
        cm.set_title("Création d'une instance du modèle ODE (exemple d'apprentissage)")

    #____________________________________________________________________________________________

    def ode(self, y, t):
        dy = np.zeros(4)
        dy[0] = y[2]
        dy[1] = y[3]
        v2 = y[2]**2 + y[3]**2
        theta = np.arctan2(y[3], y[2])

        Cx = -self.Cd * np.cos(theta) - self.Cl * np.sin(theta)
        Cz = self.Cl * np.cos(theta) - self.Cd * np.sin(theta)

        dy[2] = self.beta * v2 * Cx + self.Ct * np.cos(theta)
        dy[3] = -g + self.beta * v2 * Cz + self.Ct * np.sin(theta)

        return dy

    #____________________________________________________________________________________________

    def solve_trajectory(self, alpha, t_end):
        self.t = np.linspace(0, t_end, self.npt)
        self.alpha = np.deg2rad(alpha)

        y_init = [0,
                  self.h,
                  self.v_0 * np.cos(self.alpha),
                  self.v_0 * np.sin(self.alpha)]
        y = odeint(self.ode, y_init, t=self.t)

        self.x, self.z, self.v_x, self.v_z = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    #____________________________________________________________________________________________

    def plot_trajectory(self):
        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position X"], fontsize=12)
        plt.grid()
        plt.show()

    #____________________________________________________________________________________________

    def plot_trajectories(self, param1, param2, param3, listC, t_end):
        # listC = [Cl4, Cd4, Cl5, Cd5, Cl7, Cd7]

        # 1ère trajectoire
        self.update_param(param1)
        self.solve_trajectory(param1["alpha"], t_end)
        self.list1[0], self.list1[1] = self.x, self.z

        # 2ème trajectoire
        self.update_param(param2)
        self.solve_trajectory(param2["alpha"], t_end)
        self.list2[0], self.list2[1] = self.x, self.z

        # 3ème trajectoire
        self.update_param(param3)
        self.solve_trajectory(param3["alpha"], t_end)
        self.list3[0], self.list3[1] = self.x, self.z

        lab4 = r"$C_l, C_d$ : " + f"{listC[0]:.2f}, {listC[1]:.2f}"
        lab5 = r"$C_l, C_d$ : " + f"{listC[2]:.2f}, {listC[3]:.2f}"
        lab7 = r"$C_l, C_d$ : " + f"{listC[4]:.2f}, {listC[5]:.2f}"

        plt.plot(self.list1[0], self.list1[1], marker="+", color="blue",
                 linewidth=3, label=lab4)
        plt.plot(self.list2[0], self.list2[1], marker="+", color="red",
                 linewidth=3, label=lab5)
        plt.plot(self.list3[0], self.list3[1], marker="+", color="green",
                 linewidth=3, label=lab7)

        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()

    #____________________________________________________________________________________________

    def validation(self, t_end, npt):
        cm.set_msg("Validation")

        print("analytical solution at t = %f" % self.t[-1])
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x, z                       : %f  %f" % (x_ref, z_ref))
        print("numerical solution at the same time:")
        print("x, z                       : %f  %f" % (self.x[-1], self.z[-1]))

        # Courbe analytique
        self.time = np.linspace(0, t_end, npt)
        x = self.v_0 * np.cos(self.alpha) * self.time
        z = -(g * 0.5 * (self.time)**2) + self.v_0 * self.time * np.sin(self.alpha) + self.h

        ecart = [np.max(np.abs(x - self.x)),
                 np.max(np.abs(z - self.z))]

        if np.max(ecart) < 1e-7:
            print("La validation est vraie")
        else:
            print("Pas de validation")

        print("L'erreur max est ", np.max(ecart))

        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3)
        plt.plot(x, z, marker="+", color="green", linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Numérique", "Analytique"], fontsize=12)
        plt.grid()
        plt.show()

    #____________________________________________________________________________________________

    def set_reference_solution(self, t):
        x = self.v_0 * np.cos(self.alpha) * t
        z = - g / 2 * t ** 2 + self.v_0 * np.sin(self.alpha) * t + self.h
        return x, z

    #____________________________________________________________________________________________

    def set_impact_values(self):
        def interpo(a, n, u):
            # interpolation linéaire entre u[n] et u[n+1]
            return u[n] + a * (u[n + 1] - u[n])

        # on cherche le premier passage de z de >0 à <=0
        n = 0
        for i in range(len(self.z) - 1):
            if self.z[i] > 0 and self.z[i + 1] <= 0:
                n = i
                break

        # paramètre d'interpolation a tel que z_i = 0
        a = - self.z[n] / (self.z[n + 1] - self.z[n])

        # temps et position à l'impact
        t_i = interpo(a, n, self.t)
        x_i = interpo(a, n, self.x)
        z_i = 0.0  # par définition de l’impact

        # vitesses à l'impact
        v_x = interpo(a, n, self.v_x)
        v_z = interpo(a, n, self.v_z)
        v = np.sqrt(v_x**2 + v_z**2)

        # angle de la vitesse
        theta_i = np.arctan2(v_z, v_x)

        # résultat conservé
        self.impact_values = {
            "t_i": t_i,
            "p": x_i,
            "angle": np.rad2deg(theta_i),
            "v": [v_x, v_z, v],
        }
        return self.impact_values

    #____________________________________________________________________________________________

    def get_parameters(self):
        print("v_0        : %.2f m/s" % self.v_0)
        print("h          : %.2f m" % self.h)
        print("alpha      : %.2f °" % np.rad2deg(self.alpha))
        print("mass       : %.2f kg" % self.mass)
        print("rho        : %.2f kg/m^3" % self.rho)
        print("area       : %.2f m^2" % self.area)
        print("Cd         : %.2f" % self.Cd)
        print("Cl         : %.2f" % self.Cl)

    #____________________________________________________________________________________________

    def get_impact_values(self):
        """
        Joli affichage pour les valeurs d'impact
        """
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % self.impact_values["v"][2])

# Cdc et alphaC pour dire Cd pour la focntion de contour et alpha pouhr la fonction de contour
    def plot_contour (self , alphaC, CdC, param_base, t_end):

        R = np.zeros((len(alphaC), len(CdC)))

        for i, alphai in enumerate(alphaC):
            for j,Cdj in enumerate (CdC):
                param_new={}
                param_new["v_0"], param_new["h"],  param_new["npt"],param_new["a"] =  param_base["v_0"], param_base["h"],  param_base["npt"], param_base["a"]
                param_new["area"], param_new["mass"] , param_new["Cl"], param_new["rho"] =  param_base["area"], param_base["mass"], param_base["Cl"], param_base["rho"]
                param_new["alpha"], param_new["Cd"]  =  alphai, Cdj
                self.update_param(param_new)       # mettre a jour les parametre du model pour avoir un nouveau model
                self.solve_trajectory(alphai, t_end)     # resoudre pour avoir les valeurs de position en x, z les vitesse et autre 
                impact= self.set_impact_values()         # recuperer les valeurs d'impact
                R[i,j]=impact["p"]                        #  Sauvegarder les valeurs de l impact en fonction de l angle alpha et puis de Cdj


        CD, A = np.meshgrid(CdC, alphaC)
        fig, ax = plt.subplots()
        cont = ax.contourf(A, CD, R, levels=20, cmap="jet")

        cbar= fig.colorbar(cont,ax=ax)
        cbar.set_label("Portee R[m]")

        ax.set_ylabel(r"$c_d$")
        ax.set_xlabel(r"$\alpha$ [deg]")
        ax.set_title("Contours de la portée dans le plan ($\\alpha$, $c_d$)")
        plt.show()
