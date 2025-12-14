

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# mpl.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
mpl.rcParams['font.size'] = 16

colours = {
    "green": "#00B828",
    "yellow": "#FFD900",
    "purple": "#800FF2",
    "blue": "#0073FF",
    "orange": "#FF5000",
    "grey": "#B3B3B3",
}
plt.rcParams.update({
    'xtick.major.width': 2,     # x-tick thickness
    'ytick.major.width': 2,     # y-tick thickness
    'xtick.major.size': 5,        # x-tick length
    'ytick.major.size': 5,        # y-tick length
    'axes.linewidth': 2,         # Thickness of axis border (applies to spines)
    'lines.linewidth': 2
})

def effective_mass(k_reduced, energies, a, k_position=0.5, n_points=11,
                   hbar=1.0545718e-34, m_e=9.10938356e-31):

    if n_points < 3 or n_points % 2 == 0:
        raise ValueError("n_points must be odd and >= 3")

    # Reflect so k=0.5 sits in the middle
    k_reflected = np.concatenate([k_reduced, k_reduced[-2:0:-1]])
    E_reflected = np.concatenate([energies, energies[-2:0:-1]])

    idx = np.argmin(np.abs(k_reflected - k_position))
    half_window = n_points // 2
    if idx < half_window or idx >= len(k_reflected) - half_window:
        raise ValueError(f"k_position too close to boundary")

    indices = np.arange(idx - half_window, idx + half_window + 1)
    k_local = k_reflected[indices]
    E_local = E_reflected[indices]

    # Center k around k_position
    k_centered = k_local - k_position

    # Units
    k_SI = k_centered * (2 * np.pi / (a * 1e-10))   # 1/m
    E_J = E_local * 1.60218e-19                     # J

    # Fit parabola: E = E0 + a*k²
    def parabola(k, E0, curvature_half):
        return E0 + curvature_half * k**2
    
    popt, _ = curve_fit(parabola, k_SI, E_J)
    curvature = 2 * popt[1]  # d²E/dk² = 2a
    
    if curvature == 0:
        raise ZeroDivisionError("Curvature is zero; cannot compute.")

    m_eff = hbar**2 / curvature

    return m_eff / m_e


def plot_dos(dos, fermi_level, range):
  plt.plot(dos[:, 0], dos[:, 1], color = 'k') # plot the dos as a black line
  plt.axvline(fermi_level, linestyle='dashed', color = "k") # fermi energy
  # the next two lines make a fill with a colour depending on occupancy
  plt.fill_between(dos[:, 0], dos[:, 1], where=(dos[:, 0] < fermi_level),
                  facecolor=colours["blue"], alpha=0.5, label='occupied')
  plt.fill_between(dos[:, 0], dos[:, 1], where=(dos[:, 0] >= fermi_level),
                  facecolor=colours["orange"], alpha=0.5, label='unoccupied')
  plt.xlabel("energy (eV)")
  plt.ylabel("density of states")
  plt.xlim(fermi_level - range, fermi_level + range)
  plt.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.05, 1))
  plt.ylim(0, 1.1*np.max(dos[:, 1]))
  plt.show()