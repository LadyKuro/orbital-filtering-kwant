"""
Fig. 4, right panel: zoom on the interface showing the p_x, p_y orbitals
in the leads and the rotated p_x', p_y' orbitals in the central region.

Pure matplotlib drawing; no Kwant computation.
Moved unchanged from notebooks/visualisations.ipynb (cell 3); only the
output paths were changed (now figures/fig4_orbital_zoom.*).

Run from the repository root:  python scripts/fig4_orbital_zoom.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import cKDTree
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
})


# ----------------------------------------------------------------------
#  p-orbital glyph
# ----------------------------------------------------------------------
def _p_orbital_polys(center, angle, length, squeeze=0.70, n=100):
    """Two lobe polygons of a p orbital centred at `center`, axis along `angle`."""
    phi = np.linspace(-np.pi / 2, np.pi / 2, n)
    r = np.cos(phi)
    xl = r * np.cos(phi) * length
    yl = r * np.sin(phi) * length * squeeze

    out = []
    for sign in (+1, -1):
        ca, sa = np.cos(angle), np.sin(angle)
        X = sign * (ca * xl - sa * yl) + center[0]
        Y = sign * (sa * xl + ca * yl) + center[1]
        out.append(np.column_stack([X, Y]))
    return out  # [positive lobe, negative lobe]


def draw_p_orbital(ax, center, angle, length, color,
                   alpha=0.75, zorder=3, lw=0.9):
    """One p orbital: filled lobe (+), hollow lobe (-)."""
    pos, neg = _p_orbital_polys(center, angle, length)
    ax.add_patch(Polygon(pos, closed=True, facecolor=color,
                         edgecolor=color, lw=lw, alpha=alpha, zorder=zorder))
    ax.add_patch(Polygon(neg, closed=True, facecolor='white',
                         edgecolor=color, lw=lw, alpha=min(1.0, alpha + 0.2),
                         zorder=zorder))


# ----------------------------------------------------------------------
#  zoomed schematic of the upper-left interface
# ----------------------------------------------------------------------
def draw_interface_zoom(
    theta=np.pi / 6,
    a=0.5,
    n_lead=4,
    n_central_x=8,
    width=6,
    interface_cutoff=2.0,
    # --- zoom window, in lattice cells ---
    n_lead_cols=2,        # lead columns kept left of the interface
    n_cent_cols=3,        # central columns kept right of the interface
    n_rows=1,             # rows kept below the top row
    # --- which sites carry orbitals ---
    orb_rows=2,           # rows from the top
    orb_cols_lead=2,      # lead columns counted back from the interface
    orb_cols_cent=2,      # central columns counted from the interface
    # --- appearance ---
    orb_len=0.45,         # lobe length in units of a
    c_px="#E07A1F",
    c_py="#7B3FB5",
    label_orbitals=True,
    show_iface_line=False,
    figsize=(6.4, 4.6),
    save_path=None,
    title=r'Rotation of the local $p_x$, $p_y$ orbital basis',
):
    fig, ax = plt.subplots(figsize=figsize)

    c_lead = "#1565C0"
    c_central = "#088C34"
    c_interface = "#752C0A"

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    a1 = R @ np.array([a, 0])
    a2 = R @ np.array([0, a])

    L1, L2, L3 = n_lead, n_central_x, n_lead
    x_iface1 = (L1 + 1) * a
    x_iface2 = (L1 + L2) * a
    y_min, y_max = 0.0, (width - 1) * a

    # ---------------- sites (identical construction to the overview) ----
    left_sites = [np.array([ix * a, iy * a])
                  for ix in range(L1 + 1) for iy in range(width)]

    central_sites = []
    for i in range(-L2 - 4, L2 + 5):
        for j in range(-width, width + 3):
            pos = np.array([x_iface1, 0.0]) + i * a1 + j * a2
            x, y = pos
            if (x_iface1 - 0.6 * a <= x < x_iface2 - 0.4 * a and
                    y_min - 0.5 * a <= y < y_max + 0.5 * a):
                central_sites.append(pos)
    central_sites = np.array(central_sites)

    # ---------------- zoom window --------------------------------------
    x_lo = a * L1 - n_lead_cols * a - 0.55 * a
    x_hi = x_iface1 + n_cent_cols * a + 0.15 * a
    y_hi = y_max + 0.60 * a
    y_lo = y_max - n_rows * a - 0.55 * a

    def inside(p, pad=0.25 * a):
        return (x_lo - pad <= p[0] <= x_hi + pad and
                y_lo - pad <= p[1] <= y_hi + pad)

    # ---------------- bonds --------------------------------------------
    arr_left = np.array(left_sites)
    for i, j in cKDTree(arr_left).query_pairs(r=a * 1.05):
        p1, p2 = arr_left[i], arr_left[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=c_lead, lw=3.0, alpha=0.85, zorder=1)

    for i, j in cKDTree(central_sites).query_pairs(r=a * 1.05):
        p1, p2 = central_sites[i], central_sites[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=c_central, lw=3.0, alpha=0.85, zorder=1)

    # interface hoppings: same algorithm as add_interface_coupling
    sq_sites = [s for s in left_sites if np.isclose(s[0], a * L1)]
    rot_sites = [s for s in central_sites
                 if abs(s[0] - x_iface1) < interface_cutoff * a]
    tree_r = cKDTree(rot_sites)
    for sq in sq_sites:
        idxs = tree_r.query_ball_point(sq, r=1.4 * a)
        if len(idxs) == 0:
            idxs = [tree_r.query(sq)[1]]
        for idx in idxs:
            rot = rot_sites[idx]
            ax.plot([sq[0], rot[0]], [sq[1], rot[1]],
                    color=c_interface, lw=3.0, alpha=0.95, zorder=2)

    if show_iface_line:
        ax.plot([a * L1 + 0.5 * a] * 2, [y_lo, y_hi], color="0.55",
                lw=1.0, ls='--', alpha=0.7, zorder=0)

    # ---------------- which sites get orbitals --------------------------
    y_orb = y_max - (orb_rows - 1) * a - 0.30 * a          # top `orb_rows` rows
    x_orb_lo = a * L1 - (orb_cols_lead - 1) * a - 0.30 * a  # lead side
    x_orb_hi = x_iface1 + (orb_cols_cent - 1) * a + 0.55 * a  # central side

    def carries_orbital(p):
        return (p[1] >= y_orb) and (x_orb_lo <= p[0] <= x_orb_hi)

    Lorb = orb_len * a
    decorated = {'lead': [], 'cent': []}

    def decorate(sites, angle, c_site, key):
        for s in sites:
            if not inside(s):
                continue
            if carries_orbital(s):
                draw_p_orbital(ax, s, angle, Lorb, c_px, zorder=3)           # p_x
                draw_p_orbital(ax, s, angle + np.pi / 2, Lorb, c_py, zorder=3)  # p_y
                decorated[key].append(s)
            ax.plot(*s, 'o', color=c_site, ms=6.0, zorder=6,
                    markeredgecolor='black', markeredgewidth=1.1)

    decorate(left_sites, 0.0, c_lead, 'lead')
    decorate(central_sites, theta, c_central, 'cent')

    # ---------------- minimal p_x / p_y labels --------------------------
    if label_orbitals:
        off = 1.30 * Lorb
        box = dict(facecolor='white', edgecolor='none', alpha=0.95, pad=0.8)

        # lead: top-left decorated site, labels pointing out of the lattice
        if decorated['lead']:
            s = min(decorated['lead'], key=lambda p: (p[0], -p[1]))
            ax.text(s[0] - off, s[1] + 0.10 * a, r'$p_x$', color=c_px,
                    fontsize=25, fontweight = "bold", ha='right', va='center', zorder=7, bbox=box)
            ax.text(s[0], s[1] + off, r'$p_y$', color=c_py, fontsize=25, fontweight = "bold",
                    ha='center', va='bottom', zorder=7, bbox=box)

        # central: rightmost/lowest decorated site, labels along the rotated axes
        if decorated['cent']:
            s = max(decorated['cent'], key=lambda p: p[0] - p[1])
            ax.text(*(s + R @ np.array([off, 0.0])), r"$p_{x'}$", color=c_px,
                    fontsize=25, fontweight = "bold", ha='left', va='center', zorder=7, bbox=box)
            ax.text(*(s + R @ np.array([0.0, off])), r"$p_{y'}$", color=c_py,
                    fontsize=25, fontweight = "bold", ha='center', va='bottom', zorder=7, bbox=box)

    if title:
        ax.set_title(title, fontsize=20, pad=9, color='0.12')

    ax.set_xlim(x_lo+0.2, x_hi-0.5)
    ax.set_ylim(y_lo-0.3, y_hi  + (0.30 * a if label_orbitals else 0.0))
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
    return fig, ax


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    draw_interface_zoom(theta=np.pi / 6 + 0.01, n_lead=4, n_central_x=8,
                        width=6, save_path='figures/fig4_orbital_zoom.pdf')
    plt.savefig('figures/fig4_orbital_zoom.png', dpi=170, bbox_inches='tight')
