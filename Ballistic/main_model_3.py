import numpy as np                      # module de math
import matplotlib.pyplot as plt         # module graphique
from scipy.constants import g           # constante en m/s^2.
from model_3 import Model_3


task = 1


if task == 0:         # test of AnalyticalModel
    t_end, alpha_ref = 3, 20
    model_2 = Model_3({
        "v_0": 20, "h": 20, "alpha": 36, "npt": 101,
        "area": 0.01, "rho": 1.3, "mass": 0.1,
        "Cl": 0.18, "Cd": 0.01, "a": 0.3
    })

    model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)
    print("solution:", model_2.x, model_2.z)

    plt.figure(0)
    # Validation par rapport à l’erreur
    model_2.validation(t_end, 101)
    model_2.plot_trajectory()


elif task == 1:       # test of AnalyticalModel
    t_end = 20
    alpha_ref = 36

    model_4 = Model_3({ "v_0": 40, "h": 20, "alpha": 36, "npt": 101,"area": 0.01, "rho": 1.3, "mass": 0.1,"Cl": 0, "Cd": 0, "a": 0.3})
    model_5 = Model_3({"v_0": 40, "h": 20, "alpha": 36, "npt": 101,"area": 0.01, "rho": 1.3, "mass": 0.1,"Cl": 0, "Cd": 0.01, "a": 0.3})
    model_7 = Model_3({"v_0": 40, "h": 20, "alpha": 36, "npt": 101,"area": 0.01, "rho": 1.3, "mass": 0.1,"Cl": 0.18, "Cd": 0.01, "a": 0.3})
    """
    model_4.solve_trajectory(alpha=alpha_ref, t_end=t_end)
    model_5.solve_trajectory(alpha=alpha_ref, t_end=t_end)
    model_7.solve_trajectory(alpha=alpha_ref, t_end=t_end)
    """
    listC = [model_4.Cl, model_4.Cd, model_5.Cl, model_5.Cd,model_7.Cl, model_7.Cd]

    param1 = {"v_0": 40, "h": 20, "alpha": 36, "npt": 101,"area": 0.01, "rho": 1.3, "mass": 0.1,"Cl": 0, "Cd": 0, "a": 0.3}
    param2 = {"v_0": 40, "h": 20, "alpha": 36, "npt": 101,"area": 0.01, "rho": 1.3, "mass": 0.1,"Cl": 0, "Cd": 0.01, "a": 0.3}
    param3 = {"v_0": 40, "h": 20, "alpha": 36, "npt": 101, "area": 0.01, "rho": 1.3, "mass": 0.1, "Cl": 0.18, "Cd": 0.01, "a": 0.3 }

    # on utilise une instance pour tracer les 3 trajectoires
    model_plot = Model_3(param1)
    model_plot.plot_trajectories(param1, param2, param3, listC, t_end)


    alphaC = np.linspace(20, 60, 11)
    CdC = np.linspace(0, 0.1, 11)
    model_plot.plot_contour(alphaC, CdC, param1,t_end)