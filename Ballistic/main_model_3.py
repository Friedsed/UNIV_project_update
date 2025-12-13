



import numpy as np                      # module de math
import matplotlib.pyplot as plt         # module graphique
from scipy.constants import g           # constante en m/s^2.
from model_3 import Model_3

task = 1

if task == 0:         # test of AnalyticalModel
    t_end, alpha_ref = 3, 20
    model_2 = Model_3({"v_0": 20, "h": 20, "alpha": 36, "npt": 101, "area": 0.01, "rho":1.3 , "mass": 0.1, "Cl":0.18, "Cd": 0.01, "a":0.3})
    model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    print("solution:", model_2.x, model_2.z)
    plt.figure(0)
    #______________________________________________________________________________  ICI C EST LA VALIDATION PAR RAPPORTAL ERREUR 
    model_2.validation(t_end, 101)
    #plt.figure(1)
    #______________________________________________________________________________
    model_2.plot_trajectory()

elif task == 1:         # test of AnalyticalModel
    t_end = 20
    alpha_ref = 36
    
    model_4 = Model_3({"v_0": 40, "h": 20, "alpha": 36, "npt": 101, "area": 0.01, "rho":1.3 , "mass": 0.1, "Cl": 0, "Cd": 0, "a":0.3})
    model_5 = Model_3({"v_0": 40, "h": 20, "alpha": 36, "npt": 101, "area": 0.01, "rho":1.3 , "mass": 0.1, "Cl":0, "Cd": 0.01, "a":0.3})
    model_7 = Model_3({"v_0": 40, "h": 20, "alpha": 36, "npt": 101, "area": 0.01, "rho":1.3 , "mass": 0.1, "Cl":0.18, "Cd": 0.01, "a":0.3})
    
    model_4.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    model_5.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    model_7.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    listC=[model_4.Cl   ,   model_4.Cd  ,   model_5.Cl,  model_5.Cd,  model_7.Cl,     model_7.Cd   ]

    def plot_trajectories(list1, list2, list3, listC):
        # listC = [Cl4, Cd4, Cl5, Cd5, Cl7, Cd7]

        lab4 = r"$C_l, C_d$ : " + f"{listC[0]:.2f}, {listC[1]:.2f}"
        lab5 = r"$C_l, C_d$ : " + f"{listC[2]:.2f}, {listC[3]:.2f}"
        lab7 = r"$C_l, C_d$ : " + f"{listC[4]:.2f}, {listC[5]:.2f}"

        plt.plot(list1[0], list1[1], marker="+", color="blue",
                linewidth=3, label=lab4)
        plt.plot(list2[0], list2[1], marker="+", color="red",
                linewidth=3, label=lab5)
        plt.plot(list3[0], list3[1], marker="+", color="green",
                linewidth=3, label=lab7)

        plt.xlabel("Position X")
        plt.ylabel("Position Z")
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()

    plot_trajectories([model_4.x,model_4.z], [model_5.x, model_5.z], [ model_7.x , model_7.z], listC)


    alphaC= np.linspace(20,60,11)
    CdC=np.linspace(0,0.1,11)

    
  

    def plot_contour(params_base):
        """
        params_base : dict contenant tous les paramètres fixes de Model_3
                    (v0, h, alpha_ref, npt, mass, rho, area, cl, ...).
                    On NE met pas 'alpha' ni 'cd' dedans, ils seront balayés.
        """

        # grille 11x11 : alpha entre 20° et 60°, cd entre 0 et 0.1
        alpha_vec = np.linspace(20.0, 60.0, 11)
        cd_vec    = np.linspace(0.0,  0.1, 11)

        # matrice des portées (distance d'impact)
        R = np.zeros((len(alpha_vec), len(cd_vec)))
        t_end=20

        for i, alpha in enumerate(alpha_vec):
            for j, Cd in enumerate(cd_vec):

                # on crée une copie du dictionnaire de paramètres
                params = params_base.copy()
                params["alpha"] = alpha
                params["Cd"]    = Cd

                # instance du modèle 3
                model = Model_3(params)

                # on résout la trajectoire (temps final et pas de temps dans params)
                model.solve_trajectory(alpha, t_end)

                # valeurs à l'impact (méthode que tu as déjà codée)
                impact = model.set_impact_values()   # [ti, xi, zi, vxi, vzi, theta_i]

                # portée = abscisse à l'impact
                R[i, j] = impact["p"]

        # tracé des isocontours
        CD, A = np.meshgrid(cd_vec, alpha_vec)

        fig, ax = plt.subplots()
        cont = ax.contourf(CD, A, R, levels=20, cmap="jet")

        cbar = fig.colorbar(cont, ax=ax)
        cbar.set_label("Portée R [m]")

        ax.set_xlabel(r"$c_d$")
        ax.set_ylabel(r"$\alpha$ [deg]")
        ax.set_title("Contours de la portée dans le plan ($\\alpha$, $c_d$)")

        plt.show()

    plot_contour({"v_0": 40, "h": 20,  "npt": 101, "area": 0.01, "rho":1.3 , "mass": 0.1, "Cl": 0, "a":0.3, })



"""
    def plot_contour(alphaC,CdC):
        X = np.zeros((11, 11))
        for i in  alphaC : 
            for j in CdC:
                model_8 = Model_3({"v_0": 40, "h": 20, "alpha": i, "npt": 101, "area": 0.01, "rho":1.3 , "mass": 0.1, "Cl": 0, "Cd": j, "a":0.3})
                model_8.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut

                impact_para = model_8.set_impact_values
                X[i][j]=impact_para["p"]


    X=plot_contour(alphaC,CdC)
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(alphaC,CdC, X)
    fig.colorbar(pc)

    plt.show


    
 #______________________________________________________________________________
    model_4.plot_trajectory()
    model_4.validation(t_end, 101)

    
    #______________________________________________________________________________
    model_5.plot_trajectory()
    model_5.validation(t_end, 101)

    
    #______________________________________________________________________________
    model_7.plot_trajectory()
    model_7.validation(t_end, 101)


    #model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    # Décommenter au fur et à mesure des implémentations.
    # model_2.plot_trajectory()
    # model_2.validation()
    # impact = model_2.set_impact_values()
    # model_2.get_impact_values()()

plt.show()
print("normal end of execution")"""