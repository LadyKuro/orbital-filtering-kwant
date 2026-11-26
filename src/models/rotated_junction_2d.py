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

    def build(self):
        if self.width < 2:
            raise ValueError("Rotated2D requires width >= 2 (a truly 2D junction).")
        self.syst = self.build_system_2D()
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
    