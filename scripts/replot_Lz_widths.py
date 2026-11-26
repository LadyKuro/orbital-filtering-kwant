"""
Re-plot the transmitted L_z^max (energy, theta) maps for a series of lead
widths W, using the data already computed and saved by batch_Lz_widths.py.
No Kwant computation is performed here - this script only reads the
per-width .npz files from DATA_DIR and re-generates the plots.

Expects files of the form:
    data/Lz_map_w{width}_L{L_central}.npz
as written by batch_Lz_widths.py's run_width(), each containing:
    thetas, energies, Lz_map, axes, params

Run:  python replot_Lz_widths.py
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless-safe (server / tmux runs)
import matplotlib.pyplot as plt

from map_Lz_energy_theta import plot_Lz_map, plot_Lz_phase_diagram

# =======================================================================
# Configuration
# =======================================================================
WIDTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20,50,100]
#WIDTHS = [4]
L_CENTRAL = 400          # must match BASE_PARAMS['L_central'] used originally
DATA_DIR = "data/data"
PLOT_DIR = "plots"


# =======================================================================
# Helpers
# =======================================================================
def load_width_map(width, L_central=L_CENTRAL, data_dir=DATA_DIR):
    """Load the saved (thetas, energies, Lz_map, params) for one width."""
    path = os.path.join(data_dir, f"Lz_map_w{width}_L{L_central}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved data for width={width}: {path}")

    with np.load(path, allow_pickle=True) as d:
        thetas = d["thetas"]
        energies = d["energies"]
        Lz_map = d["Lz_map"]
        # params was saved via np.array(str(params)) -> 0-d object array
        params_str = str(d["params"])

    # params was stored as a stringified dict; recover it safely
    params = eval(params_str, {"__builtins__": {}}, {"array": np.array})
    return thetas, energies, Lz_map, params


def replot_width(width, L_central=L_CENTRAL):
    """Load one width's map and regenerate its plot."""
    thetas, energies, Lz_map, params = load_width_map(width, L_central)
    fig, _ = plot_Lz_map(
        energies, thetas, Lz_map, params,
        save_path=os.path.join(PLOT_DIR, f"Lz_map_w{width}.png"),
    )
    plt.close(fig)
    print(f"[W={width:2d}] re-plotted, max Lz = {np.nanmax(Lz_map):.4f}")
    return thetas, energies, Lz_map, params


# =======================================================================
# Main
# =======================================================================
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    maps = {}
    thetas_ref = energies_ref = None

    for w in WIDTHS:
        path = os.path.join(DATA_DIR, f"Lz_map_w{w}_L{L_CENTRAL}.npz")
        if not os.path.exists(path):
            print(f"[W={w:2d}] SKIPPED - file not found: {path}")
            continue

        thetas, energies, Lz_map, params = replot_width(w, L_CENTRAL)
        maps[f"Lz_map_w{w}"] = Lz_map
        thetas_ref, energies_ref = thetas, energies

    if not maps:
        print("No saved width maps found in "
              f"'{DATA_DIR}' - nothing to plot.")
        return

    print(f"Re-plotted {len(maps)}/{len(WIDTHS)} width maps "
          f"from existing data in '{DATA_DIR}'.")


if __name__ == "__main__":
    main()
