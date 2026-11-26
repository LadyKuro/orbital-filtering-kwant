import numpy as np
import kwant
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

Lz = np.array([[0, 0, 0],
               [0, 0, -1j],
               [0, 1j, 0]])


class Rotated2D:
    def __init__(self, onsite_energy, Vpi, Vsigma, L, L_central, width, a, theta, coupling):
        # store physical parameters
        self.width = width
        self.L = L
        self.L_central = L_central
        self.a = a
        self.theta = theta
        self.e = onsite_energy
        self.Vpi = Vpi
        self.Vsigma = Vsigma
        self.norb = 3
        self.coupling = coupling

        # precompute regions
        self.L1 = L // 2 - L_central // 2
        self.L2 = L_central
        self.L3 = L // 2 - L_central // 2

        # rotation matrix
        self.R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # bounding box
        self.x_min = self.L1 * a
        self.x_max = (self.L1 + self.L2) * a
        self.y_min = 0
        self.y_max = width * a

        self.syst = None

        self.binning = 1.0  # default binning coefficient
        self.energy = 0.0  # default energy for calculations


    # ============================================================
    #  Hopping matrices
    # ============================================================
    def t0(self):
        """On-site potential."""
        return self.e * np.eye(3)

    def t1(self, angle):
        """Standard SK hopping along x direction."""
        v1, v2 = self.Vpi, self.Vsigma
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [v1, 0, 0],
            [0, v1*c*c + v2*s*s, -(v1 - v2)*c*s],
            [0, -(v1 - v2)*c*s, v1*s*s + v2*c*c]
        ])

    def t2(self, angle):
        """Standard SK hopping along y direction."""
        v1, v2 = self.Vpi, self.Vsigma
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [v1, 0, 0],
            [0, v2*c*c + v1*s*s, -(v2 - v1)*c*s],
            [0, -(v2 - v1)*c*s, v2*s*s + v1*c*c]
        ])

    def t_direction(self, phi, dist):
        """General Slater-Koster hopping for arbitrary direction."""
        v1, v2 = self.Vpi, self.Vsigma
        c, s = np.cos(phi), np.sin(phi)

        t = np.zeros((3, 3))
        t[0, 0] = v1
        t[1,1] = v1*c*c + v2*s*s
        t[2,2] = v1*s*s + v2*c*c

        t_xy = (v1 - v2) * c * s
        t[1, 2] = t_xy
        t[2, 1] = t_xy

        # distance correction
        t *= self.coupling*(self.a / dist)**2
        return t


    # ============================================================
    #  Geometry helpers
    # ============================================================
    def rot(self, i, j):
        return self.R @ np.array([i, j])

    def inv_rot(self, x, y):
        return self.R.T @ np.array([x, y])


    # ============================================================
    #  Build steps
    # ============================================================
    def add_square_region(self, syst, lat):
        """Insert left and right square regions."""
        for x in list(range(self.L1)) + list(range(self.L1 + self.L2, self.L1 + self.L2 + self.L3)):
            for y in range(self.width):
                syst[lat(x, y)] = self.t0()

        syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = self.t1(0)
        syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = self.t2(0)

    def add_rotated_region(self, syst, lat_rot):
        """Insert region defined on rotated lattice."""
        a = self.a
        rotated_sites = []

        # find minimal index extent enclosing bounding box
        indices = [
            lat_rot.closest([x, y])
            for x in [self.x_min, self.x_max]
            for y in [self.y_min, self.y_max]
        ]
        indices = np.array(indices)
        i_min, j_min = indices.min(axis=0)
        i_max, j_max = indices.max(axis=0)

        # fill the rotated region
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                x, y = self.rot(i, j)
                if (self.x_min-0.5*a <= x < self.x_max-0.5*a 
                    and self.y_min-0.5*a <= y < self.y_max - 0.5*a):
                    site = lat_rot(i, j)
                    syst[site] = self.t0()
                    rotated_sites.append(site)

        # nearest-neighbor hoppings (rotated orientation)
        syst[kwant.builder.HoppingKind((1, 0), lat_rot, lat_rot)] = self.t1(self.theta)
        syst[kwant.builder.HoppingKind((0, 1), lat_rot, lat_rot)] = self.t2(self.theta)


        return rotated_sites
    
    def add_interface_coupling(self, syst, lat, rotated_sites):
        a = self.a

        def build_interface(x_sq, x_rot):
            sq_sites = [s for s in syst.sites() if s.family == lat and np.isclose(s.pos[0], x_sq)]
            rot_sites = [s for s in rotated_sites if abs(s.pos[0] - x_rot) < 2*a]

            if not rot_sites:
                print(f"Warning: No interface rotated sites found at x_rot={x_rot}.")
                return

            tree = cKDTree([s.pos for s in rot_sites])
            for sq in sq_sites:
                dist, idx = tree.query(sq.pos)
                rot_site = rot_sites[idx]
                dx, dy = (rot_site.pos - sq.pos) / a
                phi = np.arctan2(dy, dx)
                hop = self.t_direction(phi, dist)
                syst[sq, rot_site] = hop

        # Build left and right interfaces
        build_interface(x_sq=a*(self.L1 - 1), x_rot=self.L1*a)
        build_interface(x_sq=a*(self.L1 + self.L2), x_rot=(self.L1 + self.L2)*a)
    

    def attach_leads(self, syst, lat):
        """Attach left and right square leads."""
        a = self.a

        # left lead
        sym_left = kwant.TranslationalSymmetry((-a, 0))
        lead_left = kwant.Builder(sym_left)
        for y in range(self.width):
            lead_left[lat(0, y)] = self.t0()
        lead_left[kwant.builder.HoppingKind((1, 0), lat, lat)] = self.t1(0)
        lead_left[kwant.builder.HoppingKind((0, 1), lat, lat)] = self.t2(0)
        syst.attach_lead(lead_left)

        # right lead
        sym_right = kwant.TranslationalSymmetry((a, 0))
        lead_right = kwant.Builder(sym_right)
        for y in range(self.width):
            lead_right[lat(self.L1 + self.L2 + self.L3, y)] = self.t0()
        lead_right[kwant.builder.HoppingKind((1, 0), lat, lat)] = self.t1(0)
        lead_right[kwant.builder.HoppingKind((0, 1), lat, lat)] = self.t2(0)
        syst.attach_lead(lead_right)


    # ============================================================
    #  Main builder method
    # ============================================================
    def build_system_2D(self):
        a = self.a
        lat = kwant.lattice.square(a, norbs=3)
        lat_rot = kwant.lattice.Monatomic((self.R @ lat.prim_vecs).T, norbs=3)

        syst = kwant.Builder()

        # Build square parts
        self.add_square_region(syst, lat)

        # Build rotated region
        rotated_sites = self.add_rotated_region(syst, lat_rot)

        # Interfaces
        self.add_interface_coupling(syst, lat, rotated_sites)

        # Leads
        self.attach_leads(syst, lat)

        return syst.finalized()
    
    def build_system_1D(self):
        syst = kwant.Builder()
        lat = kwant.lattice.chain(norbs=3)
        #Add sites
        for x in range(self.L):
            syst[lat(x)] = self.t0()
        #Add hoppings
        for x in range(self.L1):
            syst[lat(x), lat(x + 1)] = self.t1(0)
        for x in range(self.L1, self.L1 + self.L2):
            syst[lat(x), lat(x + 1)] = self.t1(self.theta)
        for x in range(self.L1 + self.L2, self.L-1):
            syst[lat(x), lat(x + 1)] = self.t1(0)
        #Attach leads
        # Left lead (periodic unit cell defined at origin)
        sym_left = kwant.TranslationalSymmetry((-self.a, ))
        lead_left = kwant.Builder(sym_left)
        lead_left[lat(0)] = self.t0()
        lead_left[lat(0), lat(1)] = self.t1(0)
        syst.attach_lead(lead_left)

        # Right lead: define the lead unit cell at origin and let Kwant
        # attach it to the system boundary automatically.
        sym_right = kwant.TranslationalSymmetry((self.a, ))
        lead_right = kwant.Builder(sym_right)
        lead_right[lat(0)] = self.t0()
        lead_right[lat(0), lat(1)] = self.t1(0)
        syst.attach_lead(lead_right)
        return syst.finalized()
    
    # ============================================================

    def build(self):
        if self.width > 1:
            self.syst = self.build_system_2D()
        else:
            self.syst = self.build_system_1D()
        return self.syst

    def plot(self, path=None):
        if self.syst is None:
            raise ValueError("System not built yet. Call build() before plotting.")
        kwant.plot(self.syst)
        if path is not None:
            plt.savefig(path, dpi=600)
        plt.show()

    # -------------------------------------------------------------
    # Helper: returns wavefunctions reshaped into (modes, sites, norb)
    # -------------------------------------------------------------
    def _get_modes(self, energy, lead):
        wf = kwant.wave_function(self.syst, energy)
        psi = wf(lead)
        norb = 3  # adjust if your model changes later
        n_sites = len(list(self.syst.sites))
        modes = psi.reshape(-1, n_sites, norb)
        return modes
    
    def get_lead_modes(self, energy=0.0, lead=0):
        """
        Return propagating modes of a given lead at specified energy.

        Returns
        -------
        modes : dict with keys
            'wavefunctions' : array, shape (n_modes, n_sites, norb)
            'velocities'    : array, shape (n_modes,)
            'momenta'       : array, shape (n_modes,)
            'directions'    : list of 'in' / 'out'
        """
        if self.syst is None:
            raise RuntimeError("System not built. Call build() first.")

        lead_sys = self.syst.leads[lead]

        # Compute modes
        prop_modes, _ = lead_sys.modes(energy)

        wf = prop_modes.wave_functions      # (norb * n_sites_uc, n_modes)
        velocities = prop_modes.velocities
        momenta = prop_modes.momenta

        norb = self.norb

        # number of sites in the lead unit cell
        n_sites_uc = len(lead_sys.sites)

        # reshape wavefunctions
        wf = wf.T.reshape(-1, n_sites_uc, norb)

        # classify directions
        directions = [
            "in" if v > 0 else "out"
            for v in velocities
        ]

        return {
            "wavefunctions": wf,
            "velocities": velocities,
            "momenta": momenta,
            "directions": directions,
        }

    # -------------------------------------------------------------
    #  Spectrum open and closed
    # -------------------------------------------------------------

    def build_middle_finite(self):
        a = self.a
        lat_rot = kwant.lattice.Monatomic(
            (self.R @ kwant.lattice.square(a).prim_vecs).T,
            norbs=3
        )

        syst = kwant.Builder()

        # bounding box of middle region
        x_min = self.L1 * a
        x_max = (self.L1 + self.L2) * a
        y_min = 0
        y_max = self.width * a

        # determine index range covering box
        indices = [
            lat_rot.closest([x, y])
            for x in [x_min, x_max]
            for y in [y_min, y_max]
        ]
        indices = np.array(indices)
        i_min, j_min = indices.min(axis=0)
        i_max, j_max = indices.max(axis=0)

        # fill region
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                x, y = self.rot(i, j)
                if (x_min-0.5*a <= x < x_max-0.5*a and
                    y_min-0.5*a <= y < y_max-0.5*a):
                    syst[lat_rot(i, j)] = self.t0()

        # hoppings
        syst[kwant.builder.HoppingKind((1, 0), lat_rot, lat_rot)] = self.t1(self.theta)
        syst[kwant.builder.HoppingKind((0, 1), lat_rot, lat_rot)] = self.t2(self.theta)

        return syst.finalized()
    
    def middle_spectrum(self):
        syst_mid = self.build_middle_finite()
        H = syst_mid.hamiltonian_submatrix(sparse=False)
        evals = np.linalg.eigvalsh(H)
        return evals

    def build_middle_infinite(self):
        a = self.a
        lat = kwant.lattice.square(a, norbs=3)

        sym = kwant.TranslationalSymmetry((a, 0))
        syst = kwant.Builder(sym)

        for y in range(self.width):
            syst[lat(0, y)] = self.t0()

        syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = self.t1(self.theta)
        syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = self.t2(self.theta)

        return syst.finalized()
    
    def middle_bands(self, k_vals):
        syst_inf = self.build_middle_infinite()
        bands = kwant.physics.Bands(syst_inf)

        energies = np.array([bands(k) for k in k_vals])
        return energies

    # -------------------------------------------------------------
    #  Spatially resolved ⟨O(x)⟩ using binned averaging
    # -------------------------------------------------------------
    def compute_operator_binned(
            self,
            operator,
            energy=0,
            lead=0,
            binning_coeff=1.0,
            summed_over_modes=False,
            return_wfs = False,
            normed=True):

        """
        Compute ⟨O⟩(x) for an arbitrary orbital operator O using
        density-weighted binning along x.

        Per bin b and mode m:
            O_m(b) = Σ_{sites∈b} ⟨ψ|O|ψ⟩ / Σ_{sites∈b} ⟨ψ|ψ⟩   (normed=True)
            O_m(b) = Σ_{sites∈b} ⟨ψ|O|ψ⟩                        (normed=False)

        With summed_over_modes=True, numerators and norms are summed
        over modes *before* dividing, i.e. the polarization of an
        equal mixture of all incoming modes (cf. Fig. 3e of the paper).
        """
                
        self.binning = binning_coeff
        self.energy = energy

        # --- extract site positions ---
        sites = list(self.syst.sites)
        positions = np.array([s.pos for s in sites])
        x = positions[:, 0]
        x_min, x_max = x.min(), x.max()

        # --- compute number of bins (binning_coeff=1 -> ~1 bin per column) ---
        default_bins = max(1, int(round((x_max - x_min) / self.a)) + 1)
        n_bins = max(1, int(default_bins * binning_coeff))

        edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        # clip so sites at x == x_max land in the last bin instead of index n_bins
        bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

        # --- get modes (modes, sites, norb) ---
        modes = self._get_modes(energy, lead)
        n_modes, n_sites, norb = modes.shape

        operator = np.asarray(operator)
        if operator.shape != (norb, norb):
            raise ValueError(
                f"Operator has wrong shape {operator.shape}, expected {(norb, norb)}")

        # --- per-site numerators ⟨ψ|O|ψ⟩ and norms ⟨ψ|ψ⟩, all sites at once ---
        numer_sites = np.einsum(
            "msi,ij,msj->ms", modes.conj(), operator, modes).real   # (m, sites)
        norm_sites = np.einsum(
            "msi,msi->ms", modes.conj(), modes).real                # (m, sites)

        # --- accumulate into bins (density-weighted) ---
        numer_bins = np.zeros((n_modes, n_bins))
        norm_bins = np.zeros((n_modes, n_bins))
        for b in range(n_bins):
            site_list = np.where(bin_idx == b)[0]
            if len(site_list) == 0:
                continue
            numer_bins[:, b] = numer_sites[:, site_list].sum(axis=1)
            norm_bins[:, b] = norm_sites[:, site_list].sum(axis=1)

        if n_modes == 0:
            print("Warning: no propagating modes at this energy.")

        # --- optionally form the unpolarized mixture over modes FIRST ---
        if summed_over_modes:
            numer_bins = numer_bins.sum(axis=0, keepdims=True)  # (1, n_bins)
            norm_bins = norm_bins.sum(axis=0, keepdims=True)

        # --- divide (or not) ---
        if normed:
            O_total = np.zeros_like(numer_bins)
            valid = norm_bins > 1e-12
            O_total[valid] = numer_bins[valid] / norm_bins[valid]
        else:
            O_total = numer_bins

        if return_wfs:
            # binned probability density (summed over sites in bin)
            wfs_binned = norm_bins.copy()
            return bin_centers, O_total, wfs_binned
        
        return bin_centers, O_total

    # -------------------------------------------------------------
    # Optional: plotting helper
    # -------------------------------------------------------------
    def plot_operator_binned(self, bin_centers, O_total, operator_name="O", path = None):
        import matplotlib.pyplot as plt

        n_modes = O_total.shape[0]
        fig, axes = plt.subplots(n_modes, 1, figsize=(5, 2*n_modes), sharex=True)
        if n_modes == 1:
            axes = [axes]

        for m, ax in enumerate(axes):
            ax.plot(bin_centers, O_total[m], label=f"{operator_name} (mode {m})")
            ax.axvline(self.L1 * self.a, color="k", ls="--")
            ax.axvline((self.L1 + self.L2) * self.a, color="k", ls="--")
            ax.legend()
            ax.set_ylabel(f"⟨{operator_name}⟩")

        axes[-1].set_xlabel("x")
        fig.suptitle(f"Spatially resolved ⟨{operator_name}⟩ at " 
                     f"E={self.energy}, width = {self.width}, binning={self.binning}")
        plt.tight_layout()
        if path is not None:
            plt.savefig(path, dpi=600)
        plt.show()

    def plot_operator_summed(self, bin_centers, O_total, operator_name="O", path = None):
        n_modes = O_total.shape[0]
        O_sum = O_total.sum(axis=0)
        plt.figure(figsize=(6,4))
        plt.plot(bin_centers, O_sum, label=f"Sum over modes")
        plt.axvline(self.L1 * self.a, color="k", ls="--")
        plt.axvline((self.L1 + self.L2) * self.a, color="k", ls="--")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel(f"Sum of ⟨{operator_name}⟩")
        plt.title(f"Spatially resolved sum of ⟨{operator_name}⟩ at "
                    f"E={self.energy}, width = {self.width}, binning={self.binning}")
        plt.tight_layout()
        if path is not None:
            plt.savefig(path, dpi=600)
        plt.show()
    

    def _orbital_current_bonds(self, psi, Lmat, direction="x"):
        results = []

        sites = list(self.syst.sites)
        norb = psi.shape[1]

        # get sparse Hamiltonian
        H = self.syst.hamiltonian_submatrix(sparse=True).tocsr()

        # iterate over nonzero off-diagonal blocks
        for i in range(len(sites)):
            for j in H[i*norb:(i+1)*norb].nonzero()[1] // norb:
                if j <= i:
                    continue

                s1 = sites[i]
                s2 = sites[j]

                r1 = np.array(s1.pos)
                r2 = np.array(s2.pos)
                d = r2 - r1

                if direction == "x" and d[0] <= 0:
                    continue
                if direction == "y" and d[1] <= 0:
                    continue

                psi1 = psi[i]
                psi2 = psi[j]

                A = psi1.conj() @ (Lmat) @ psi2
                J = 1j * (A - A.conj())

                x_mid = 0.5 * (r1[0] + r2[0])
                results.append((x_mid, J.real))

        return results

    
    def compute_orbital_current_binned(
            self,
            Lmat,
            energy=0,
            lead=0,
            binning_coeff=1.0,
            direction="x",
            summed_over_modes=False):
        """
        Compute spatially resolved orbital current ⟨J_L⟩(x)
        """
        self.binning = binning_coeff
        self.energy = energy

        # --- site positions (for binning scale) ---
        sites = list(self.syst.sites)
        x_sites = np.array([s.pos[0] for s in sites])
        x_min, x_max = x_sites.min(), x_sites.max()

        default_bins = max(1, int((x_max - x_min + 1)))
        n_bins = max(1, int(default_bins * binning_coeff))

        edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # --- get modes ---
        modes = self._get_modes(energy, lead)
        n_modes = modes.shape[0]

        J_total = np.zeros((n_modes, n_bins))

        # --- loop over modes ---
        for m in range(n_modes):
            bond_currents = self._orbital_current_bonds(
                modes[m], Lmat, direction=direction
            )

            for x_mid, J in bond_currents:
                b = np.digitize(x_mid, edges) - 1
                if 0 <= b < n_bins:
                    J_total[m, b] += J

        # remove zero-current modes
        keep = [i for i in range(n_modes) if np.abs(J_total[i]).sum() > 1e-14]
        J_total = J_total[keep]

        if summed_over_modes:
            J_total = J_total.sum(axis=0).reshape(1, -1)/len(keep)

        print("Orbital current shape:", J_total.shape)
        return bin_centers, J_total

    
    def binned_wavefunction(self, energy=0, lead=0, binning_coeff=1.0):
        modes = self._get_modes(energy, lead)
        return self.compute_operator_binned(
            operator=np.eye(self.norb),  # identity gives local density
            energy=energy,
            lead = lead,
            binning_coeff=binning_coeff,
            summed_over_modes=False,
            normed=False
        )
    
    def plot_operator_and_modes(self, bins, operator, wfs, lead_modes, name="Lz"):
        n_modes = wfs.shape[0]
        lm = lead_modes['wavefunctions']
        lm[np.abs(lm) < 1e-10] = 0.0
        weights = np.sum(np.abs(lm)**2, axis=1)
        weights_norm = weights / weights.sum(axis=1, keepdims=True)
        total_density = np.sum(wfs, axis = 0)/wfs.shape[0]
        total_op = np.sum(operator, axis = 0)/operator.shape[0]

        labels = ["$p_x$", "$p_y$", "$p_z$"]
        x = np.arange(3)
        fig, axes = plt.subplots(n_modes+1, 3, figsize=(8, 2*(n_modes)))
        for m in range(n_modes):
            axes[m][0].plot(bins, operator[m], label=f"{name} (mode {m})", color="navy")
            axes[m][0].axvline(self.L1, color='grey', linestyle='--', linewidth=1)
            axes[m][0].axvline(self.L1+self.L2, color='grey', linestyle='--', linewidth=1)
            axes[m][1].plot(bins, wfs[m], label=f"Mode {m} density", color = "darkred")
            axes[m][1].axvline(self.L1, color='grey', linestyle='--', linewidth=1)
            axes[m][1].axvline(self.L1+self.L2, color='grey', linestyle='--', linewidth=1)
            axes[m][2].bar(x, weights_norm[m])
            axes[m][2].set_xticks(x)
            axes[m][2].set_xticklabels(labels)
            axes[m][2].set_ylim(0, 1)
            axes[m][2].set_ylabel(f"Mode {m}")
            axes[m][0].legend()
            axes[m][0].set_ylabel(f"{name}")
            axes[m][1].legend()
            axes[m][1].set_ylabel("Density")

        axes[-1][0].plot(bins, total_op, label = f"Mean {name}")
        axes[-1][1].plot(bins, total_density, label = "Mean state density")
        axes[-1][0].set_xlabel("x")
        axes[-1][1].set_xlabel("x")
        axes[0][2].set_title("Orbital content of lead modes")
        plt.tight_layout()
        plt.show()

    def max_Lz_right_lead(self, energy=0.0, operator=None,
                           weight_by_transmission=True, threshold=1e-8,
                           signed=True):
        """
        Orbital-moment expectation of the outgoing modes of the right
        lead, reduced to a single scalar for use in a theta-scan.

        Parameters
        ----------
        operator : (norb,norb) array, default self.Lz
        weight_by_transmission : if True, only consider outgoing modes
            that actually carry flux transmitted from the left lead at
            this energy (i.e. mode n such that sum_m |S_{n,m}|^2 > threshold).
            If False, returns the max over *all* propagating right-lead
            modes regardless of whether they're populated -- an upper
            bound independent of the injection.
        signed : if True, return the signed value with largest |Lz|
            (so you can see sign flips vs theta, cf. Fig. 2a/2c of the
            paper). If False, return |Lz|.

        Returns
        -------
        float
        """
        if operator is None:
            operator = self.Lz
        if self.syst is None:
            raise RuntimeError("System not built. Call build() first.")

        smatrix = kwant.smatrix(self.syst, energy)
        modes_right = smatrix.lead_info[1]        # PropagatingModes, right lead,
                                                    # already ordered to match submatrix()
        wf = modes_right.wave_functions             # (norb*n_sites_uc, n_modes)
        n_sites_uc = wf.shape[0] // self.norb
        norb = self.norb
        wf = wf.T.reshape(-1, n_sites_uc, norb)      # (n_modes, n_sites_uc, norb)

        numer = np.einsum("nsi,ij,nsj->n", wf.conj(), operator, wf).real
        norm = np.einsum("nsi,nsi->n", wf.conj(), wf).real
        Lz_modes = numer / norm                      # (n_modes,)

        if len(Lz_modes) == 0:
            return 0.0

        if weight_by_transmission:
            t = smatrix.submatrix(1, 0)              # (n_out_right, n_in_left)
            T_per_mode = np.sum(np.abs(t) ** 2, axis=1)
            keep = T_per_mode > threshold
            if not np.any(keep):
                return 0.0
            Lz_modes = Lz_modes[keep]

        j = np.argmax(np.abs(Lz_modes))
        return Lz_modes[j] if signed else abs(Lz_modes[j])
    
    def transmitted_Lz_amplitude(self, energy=0.0, operator=None, n_cells=500):
        """
        Far-field <Lz>(x) of the state transmitted into the right lead,
        for an unpolarized (equal-flux) mixture of all incoming left-lead
        modes, built from the S-matrix and outgoing-mode coherences.

        Returns (max |<Lz>(x)|, (cell_index, Lz_x)).
        """
        if operator is None:
            operator = self.Lz
        sm = kwant.smatrix(self.syst, energy)
        t = sm.submatrix(1, 0)                    # (n_out_R, n_in_L)
        n_out = t.shape[0]
        if n_out == 0:
            return 0.0, (None, None)

        modes = sm.lead_info[1]
        # PropagatingModes: first half incoming, second half outgoing
        wf = modes.wave_functions[:, n_out:]      # outgoing only
        k = modes.momenta[n_out:]                  # Bloch phase e^{ik} per cell
        nuc = wf.shape[0] // self.norb
        u = wf.T.reshape(n_out, nuc, self.norb)

        rho = t @ t.conj().T                       # transmitted coherence matrix
        M = np.einsum('nsi,ij,msj->nm', u.conj(), operator, u)
        N = np.einsum('nsi,msi->nm', u.conj(), u)

        cells = np.arange(n_cells)
        dk = k[None, :] - k[:, None]               # k_m - k_n
        ph = np.exp(1j * dk[:, :, None] * cells)   # (n, m, x)

        num = np.einsum('mn,nm,nmx->x', rho, M, ph).real
        den = np.einsum('mn,nm,nmx->x', rho, N, ph).real
        Lz_x = num / den
        return np.max(np.abs(Lz_x)), (cells, Lz_x)
    
    def scan_max_Lz_vs_theta(self, thetas, energy=0.0, **kwargs):
        """
        Rebuild the system at each theta and evaluate max_Lz_right_lead.
        Returns (thetas, values) as arrays.
        """
        values = np.empty(len(thetas))
        for k, th in enumerate(thetas):
            self.theta = th
            self.R = np.array([[np.cos(th), -np.sin(th)],
                                [np.sin(th),  np.cos(th)]])
            self.build()
            values[k] = self.transmitted_Lz_amplitude(energy=energy, **kwargs)
        return np.asarray(thetas), values

    def transmitted_Lz_envelope(self, energy=0.0, operator=Lz):
        """
        L_z^max = (2/N) * sum_p |rho_{yp,xp}|  for the unpolarized equal-flux
        mixture of incoming left-lead modes, where rho = t t^dagger is the
        transmitted coherence matrix and the sum runs over pairs of outgoing
        right-lead modes (n, m) coupled by the operator (py <-> px, same
        transverse channel p -- automatically selected by the matrix element).

        Implementation note: rather than classifying modes by orbital/channel
        by hand, we use the operator matrix elements M_nm = <u_n|Lz|u_m>
        between outgoing modes. Pairs not coupled by Lz (different transverse
        channel, or involving pz) have M_nm = 0 by the orthogonality argument
        of the derivation, so the generic sum

            L_env = (2/N) * sum_{n<m} |rho_mn| * |M_nm| / sqrt(N_nn N_mm)

        reduces to the appendix formula without manual bookkeeping, and stays
        correct if the lead is ever modified (e.g. onsite disorder in tests).
        Returns 0.0 if the right lead carries no propagating modes.
        """
        sm = kwant.smatrix(self.syst, energy)
        t = sm.submatrix(1, 0)                        # (n_out_R, n_in_L)
        n_out = t.shape[0]
        if n_out == 0:
            return 0.0

        modes = sm.lead_info[1]
        wf = modes.wave_functions[:, n_out:]          # outgoing modes only
        nuc = wf.shape[0] // self.norb
        u = wf.T.reshape(n_out, nuc, self.norb)       # (n, sites_uc, norb)

        rho = t @ t.conj().T                          # transmitted coherences
        M = np.einsum('nsi,ij,msj->nm', u.conj(), operator, u)
        G = np.einsum('nsi,msi->nm', u.conj(), u)     # mode overlaps (diag ~ norms)

        N = np.real(np.sum(np.diag(rho) * np.diag(G)))
        if N <= 1e-14:
            return 0.0

        # envelope: sum of |coherence * matrix element| over distinct pairs;
        # factor 2 because (n,m) and (m,n) are the conjugate halves of the
        # same sinusoid (cf. 2 Im[...] in the derivation)
        env = 0.0
        for n in range(n_out):
            for m in range(n + 1, n_out):
                env += 2.0 * np.abs(rho[m, n] * M[n, m])
        return env / N
    