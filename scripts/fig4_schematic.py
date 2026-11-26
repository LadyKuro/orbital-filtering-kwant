"""
Fig. 4: schematic of the 2D rotated-junction tight-binding model
(square-lattice leads, central region rotated by theta, interface bonds).

Pure matplotlib drawing; no Kwant computation.
Moved from notebooks/visualisations.ipynb (cell 2). Changes: output path
(figures/fig4_schematic.pdf), and interface bonds drawn with the model's
rule (r_c = 1.2 a on both sides, fallback to the nearest site).

Run from the repository root:  python scripts/fig4_schematic.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from scipy.spatial import cKDTree

def draw_lattice_schematic(
    theta=np.pi/6,
    a=0.5,
    n_lead=4,
    n_central_x=7,
    width=4,
    interface_cutoff=2.0,   # same logic as abs(s.pos[0] - x_rot) < 2*a
    figsize=(9, 3.8),
    save_path=None
):
    fig, ax = plt.subplots(figsize=figsize)

    c_lead      = "#1565C0"   # deep royal blue
    c_central   = "#088C34"   # deep red
    c_interface = "#752C0A"   # golden amber
    c_grid      = "#757575"   # neutral gray

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    a1 = R @ np.array([a, 0])
    a2 = R @ np.array([0, a])

    L1 = n_lead
    L2 = n_central_x
    L3 = n_lead

    x_iface1  = (L1 +1) * a
    x_iface2  = (L1 + L2) * a
    x_right   = (L1 + L2 + L3) * a
    y_min, y_max = 0.0, (width - 1) * a

    # --- generate lead sites ---
    def make_lead_sites(x_range):
        sites = []
        for ix in x_range:
            for iy in range(width):
                sites.append(np.array([ix * a, iy * a]))
        return sites

    left_sites  = make_lead_sites(range(L1 + 1))
    right_sites = make_lead_sites(range(L1 + L2, L1 + L2 + L3 + 1))

    # --- generate rotated central sites ---
    central_sites = []
    for i in range(-L2-4, L2 + 5):
        for j in range(-width, width + 3):
            pos = np.array([x_iface1, 0.0]) + i * a1 + j * a2
            x, y = pos
            if (x_iface1 - 0.6*a <= x < x_iface2 - 0.4*a and
                    y_min - 0.5*a <= y < y_max + 0.5*a):
                central_sites.append(pos)
    central_sites = np.array(central_sites)

    # --- draw lead bonds ---
    def draw_lead_bonds(sites_list, x_range):
        site_arr = np.array(sites_list)
        tree = cKDTree(site_arr)
        pairs = tree.query_pairs(r=a * 1.05)
        for i, j in pairs:
            p1, p2 = site_arr[i], site_arr[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=c_lead, lw=1.6, alpha=0.8, zorder=1)

    draw_lead_bonds(left_sites,  None)
    draw_lead_bonds(right_sites, None)

    # --- draw central bonds ---
    if len(central_sites) > 0:
        tree_c = cKDTree(central_sites)
        pairs_c = tree_c.query_pairs(r=a * 1.05)
        for i, j in pairs_c:
            p1, p2 = central_sites[i], central_sites[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=c_central, lw=1.6, alpha=0.8, zorder=1)

    # --- interface hoppings: same algorithm as add_interface_coupling ---
    def draw_interface(sq_sites_all, x_sq, x_rot, bond_reach=None):
        # mirror of your Kwant code
        sq_sites = [s for s in sq_sites_all
                    if np.isclose(s[0], x_sq)]
        rot_sites = [s for s in central_sites
                     if abs(s[0] - x_rot) < interface_cutoff *a]

        if len(rot_sites) == 0:
            print(f"Warning: no rotated sites near x_rot={x_rot:.2f}")
            return

        tree_r = cKDTree(rot_sites)
        for sq in sq_sites:
            if bond_reach is None:
                dist, idx = tree_r.query(sq)
                idxs = [idx]
            else:
                idxs = tree_r.query_ball_point(sq, r=bond_reach)
                if len(idxs) == 0:
                    _, idx = tree_r.query(sq)
                    idxs = [idx]
            for idx in idxs:
                rot = rot_sites[idx]
                ax.plot([sq[0], rot[0]], [sq[1], rot[1]],
                    color=c_interface, lw=1.6,
                    #ls=(0, (4, 2.5)),   # dashed
                    alpha=0.9, zorder=2)

    draw_interface(left_sites,  x_sq=a * (L1), x_rot=x_iface1, bond_reach=1.2*a)
    draw_interface(right_sites, x_sq=a * (L1 + L2), x_rot=x_iface2, bond_reach=1.2*a)

    # --- draw sites (on top of all bonds) ---
    for s in left_sites + right_sites:
        ax.plot(*s, 'o', color=c_lead, ms=9, zorder=4, markeredgecolor='black', markeredgewidth=1.4)
    if len(central_sites) > 0:
        ax.plot(central_sites[:,0], central_sites[:,1],
                'o', color=c_central, ms=9, zorder=4, markeredgecolor='black', markeredgewidth=1.4)

    # --- interface boundary lines ---
    y_lo, y_hi = y_min - 0.4*a, y_max + 0.4*a
    for xi in [x_iface1-a, x_iface2]:
        ax.plot([xi, xi], [y_lo, y_hi], color=c_grid, lw=1.0,
                ls='--', alpha=0.8, zorder=2)


    # --- theta annotation ---
    origin    = np.array([x_iface1, y_min])
    r_arc     = 0.7 * a
    ax.annotate('', xy=origin + np.array([r_arc*1.5, 0]),
                xytext=origin,
                arrowprops=dict(arrowstyle='->', color="darkred", lw=1.5))
    rot_vec = R @ np.array([r_arc*1.3, 0])
    ax.annotate('', xy=origin + rot_vec,
                xytext=origin,
                arrowprops=dict(arrowstyle='->', color="darkred", lw=1.5, zorder = 6))
    arc = Arc(origin, 2*r_arc, 2*r_arc, angle=0,
              theta1=0, theta2=np.degrees(theta),
              color="darkred", lw=0.9, ls='--')
    ax.add_patch(arc)
    mid = theta / 2
    lp  = origin + (r_arc + 0.1) * np.array([np.cos(mid), np.sin(mid)])
    ax.text(lp[0]-0.8*a, lp[1], r'$\theta$',
            color="darkred", fontsize=17, fontweight='bold', ha='left', va='bottom')

    # --- region labels ---
    y_lab = y_min - 1.1 * a
    ax.text(x_iface1 / 2 - 0.7*a, y_lab, 'Left lead',
            ha='center', va='top', fontsize=15, color="black")
    ax.text((x_iface1 - a + x_iface2) / 2, y_lab,
            r'Central region (rotated by $\theta$)',
            ha='center', va='top', fontsize=15, color="black")
    ax.text((x_iface2 + x_right + 0.5*a) / 2, y_lab, 'Right lead',
            ha='center', va='top', fontsize=15, color="black")


    # --- dimension lines ---
    y_dim = y_min - 1.85 * a

    # --- transport arrow ---
    y_top = y_max + 0.75 * a
    ax.annotate('', xy=(x_right, y_top), xytext=(0, y_top),
                arrowprops=dict(arrowstyle='->', color="black",
                                lw=1.5, alpha=0.8))
    ax.text(x_right / 2, y_top + 0.08,
            r'transport direction $x$',
            ha='center', va='bottom', fontsize=15, color="black")

    # --- semi-infinite indicators ---
    y_mid = (y_min + y_max) / 2
    for x0, dx in [(0, -0.4), (x_right, 0.4)]:
        ax.annotate('', xy=(x0 + dx, y_mid), xytext=(x0, y_mid),
                    arrowprops=dict(arrowstyle='->', color=c_lead,
                                   lw=1.6, alpha=0.9))

    # --- width indicator (vertical, left of lattice) ---
    x_w = -1.1 * a
    ax.annotate('', xy=(x_w, y_max+0.2*a), xytext=(x_w, y_min-0.2*a),
                arrowprops=dict(arrowstyle='<->', color="black", lw=1.0))
    ax.text(x_w - 0.15, y_mid, '$W$',
            ha='right', va='center', fontsize=15, color="black")

    # --- force axis limits to include annotate-only elements ---
    # ax.annotate arrows aren't picked up by matplotlib's autoscaling
    # (only plot/scatter data is), so without this the W arrow can get
    # clipped by the axes viewport even though its text label survives
    # (Text defaults to clip_on=False, arrow patches don't).
    ax.set_xlim(x_w - 0.6 * a, x_right + 0.8 * a)
    ax.set_ylim(y_dim, y_top + 0.6 * a)

    # --- legend ---
    
    '''
    handles = [
        plt.Line2D([0],[0], color=c_lead,  lw=1.6,    label='Lead hoppings'),
        plt.Line2D([0],[0], color=c_central, lw=1.6,  label='Rotated hoppings'),
        plt.Line2D([0],[0], color=c_interface, lw=1.6,
                   #ls='--',
                    label='Interface hoppings'),
    ]
    ax.legend(handles=handles, fontsize=12, loc='lower right',
              framealpha=0.95, edgecolor='none')
    '''

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight',
                    transparent=True)
    plt.show()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    draw_lattice_schematic(
        theta=np.pi/6,
        n_lead=2,
        n_central_x=8,
        width=6,
        save_path='figures/fig4_schematic.pdf'
    )
