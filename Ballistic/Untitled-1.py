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

        y_init = [0, self.h, self.v_0*np.cos(self.alpha),
                          self.v_0*np.sin(self.alpha)]

        y = odeint(self.ode, y_init, self.t)

        self.x  = y[:, 0]
        self.z  = y[:, 1]
        self.v_x = y[:, 2]
        self.v_z = y[:, 3]


    def plot_trajectory(self):
        plt.plot(self.x, self.z, marker="+", color="red", linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.grid()
        plt.show()

    def plot_trajectories(self, list1, list2,list3):
        plt.plot(self.list1[0], self.list2[1], marker="+", color="red", linewidth=3)
        plt.plot(self.list2[0], self.list2[1] marker="+", color="red", linewidth=3)
        plt.plot(self.list3[0], self.list3[1] marker="+", color="red", linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["Position Z en fonction de la position z "], fontsize=12)
        plt.grid()
        plt.show()



    def set_reference_solution(self, t):
        x = self.v_0 * np.cos(self.alpha) * t
        z = - g/2 * t**2 + self.v_0 * np.sin(self.alpha) * t + self.h
        return x, z


    def validation(self, t_end, npt):

        cm.set_msg("Validation")
        print("analytical solution at t = %f" % self.t[-1])
        x_ref, z_ref = self.set_reference_solution(self.t[-1])
        print("x, z                       : %f  %f" % (x_ref, z_ref))
        print("numerical solution at the same time:")
        print("x, z                       : %f  %f" % (self.x[-1], self.z[-1]))

        self.time = np.linspace(0, t_end, npt)
        x, z = self.set_reference_solution(self.time)

        x_num = np.interp(self.time, self.t, self.x)
        z_num = np.interp(self.time, self.t, self.z)

        ecart = [np.max(np.abs(x - x_num)), np.max(np.abs(z - z_num))]

        if np.max(ecart) < 1e-7:
            print(" La validation est vrai")
        else:
            print(" Pas de validation")

        print("l erreur max est ", np.max(ecart))

        plt.plot(self.x, self.z, marker="+", color="red", markersize=12, linewidth=3)
        plt.plot(x, z, marker="+", color="green", linewidth=3)
        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(["numérique", "analytique"], fontsize=12)
        plt.grid()
        plt.show()


    def set_impact_values(self):

        n =0
        for i in range(len(self.z) - 1):
            if self.z[i] > 0 and self.z[i+1] <= 0:
                n = i
                break

        #if n is None:
           # raise ValueError("Aucun impact détecté : z ne devient jamais négatif.")

        def interpo(a, n, u):
            return u[n] + a * (u[n+1] - u[n])

        a = - self.z[n] / (self.z[n+1] - self.z[n])

        t_i  = interpo(a, n, self.t)
        x_i  = interpo(a, n, self.x)
        z_i  = interpo(a, n, self.z)
        vx_i = interpo(a, n, self.v_x)
        vz_i = interpo(a, n, self.v_z)

        theta_i = np.rad2deg(np.arctan2(vz_i, vx_i))

        self.impact_values = {"t_i": t_i, "p": x_i,
                              "angle": theta_i, "v": [vx_i, vz_i]}

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
        print("Impact:")
        print("time       : %.2f s" % self.impact_values["t_i"])
        print("length     : %.2f m" % self.impact_values["p"])
        print("angle      : %.2f °" % self.impact_values["angle"])
        print("|v|        : %.2f m/s" % np.sqrt(self.impact_values["v"][0]**2 + self.impact_values["v"][1]**2))
