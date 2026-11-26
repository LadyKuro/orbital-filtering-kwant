"""
Batch computation of the transmitted L_z^max (energy, theta) maps for a
series of lead widths W. For every width the full 2D scan is run and the
result is stored in a single compressed archive (plus one .npz per width,
so a crashed run does not lose finished maps).

Axis convention matches fig6_map_Lz.py:
    Lz_map shape = (len(thetas), len(energies))
    -> rows = angle, columns = energy,
       ready for pcolormesh(energies, thetas, Lz_map).

Run from the repository root:  python scripts/fig7_compute_widths.py
Then plot with:               python scripts/fig7_plot_widths.py
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe (server / tmux runs)
import matplotlib.pyplot as plt

from fig6_map_Lz import map_Lz_energy_theta, plot_Lz_map

# =======================================================================
# Configuration
# =======================================================================
WIDTHS = [2, 3, 4, 5, 6, 10, 15, 20, 50, 100]      # Fig. 7 panels

BASE_PARAMS = dict(
    onsite_energy=0.0,
    Vpi=-0.3,          # <-- your values
    Vsigma=1.0,
    L=440,             # 2*20 buffer + 400 central
    L_central=400,
    a=1.0,
    coupling=0.5,
    theta=0.0,
)

THETAS = np.linspace(0.0, np.pi / 2, 401)
ENERGIES = np.linspace(-1.5, 1.5, 401)   # adjust to your bandwidth

DATA_DIR = "data"
PLOT_DIR = "plots"
N_JOBS = -1


# =======================================================================
# Batch run
# =======================================================================
def run_width(width):
    """Compute, save and plot the map for a single width. Returns Lz_map."""
    params = dict(BASE_PARAMS, width=width)

    t0 = time.time()
    Lz_map = map_Lz_energy_theta(params, ENERGIES, THETAS, n_jobs=N_JOBS)
    dt = time.time() - t0
    print(f"[W={width:2d}] done in {dt/60:.1f} min, "
          f"max Lz = {np.nanmax(Lz_map):.4f}")

    # per-width file (safe against crashes mid-batch)
    np.savez_compressed(
        os.path.join(DATA_DIR,
                     f"Lz_map_w{width}_L{params['L_central']}.npz"),
        thetas=THETAS, energies=ENERGIES, Lz_map=Lz_map,
        axes="rows=thetas, cols=energies",
        params=np.array(str(params)))

    fig, _ = plot_Lz_map(ENERGIES, THETAS, Lz_map, params,
                         save_path=os.path.join(PLOT_DIR,
                                                f"Lz_map_w{width}.png"))
    plt.close(fig)
    return Lz_map


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    maps = {}
    for w in WIDTHS:
        maps[f"Lz_map_w{w}"] = run_width(w)

    # one combined archive with everything
    np.savez_compressed(
        os.path.join(DATA_DIR,
                     f"Lz_maps_widths_L{BASE_PARAMS['L_central']}.npz"),
        thetas=THETAS, energies=ENERGIES,
        widths=np.array(WIDTHS),
        axes="rows=thetas, cols=energies",
        base_params=np.array(str(BASE_PARAMS)),
        **maps)
    print("All widths finished; combined archive written.")


if __name__ == "__main__":
    main()
