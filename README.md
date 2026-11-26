# orbital-filtering-kwant

Code for the paper on orbital angular momentum filtering by mirror-symmetry
breaking in multi-orbital junctions *(reference and DOI will be added on publication)*.

Two tight-binding models with p_x, p_y, p_z orbitals, solved with
[Kwant](https://kwant-project.org):

- **1D two-segment chain** (`orbfilter.chain_1d`): orbital-isotropic leads and a
  central region with a mirror-breaking orbital coupling g(x) χ_z, with g = g₁ and
  g₂ in two segments. The orbital pseudospin precesses by
  Δφ = Σᵢ (k₋ − k₊)ᵢ dᵢ, and g₂ = −g₁ is an exact echo (Sec. II).
- **Rotated 2D junction** (`orbfilter.rotated_2d`): square-lattice leads and a
  central region whose lattice is rotated by θ, with Slater–Koster hoppings.
  The transmitted orbital polarization L_z^max is mapped over the (E, θ) plane
  for several widths W (Sec. III).

## Repository layout

```
src/orbfilter/        the models (installable package)
  chain_1d.py         OrbitalChain1D, transmitted_state, orbital operators
  rotated_2d.py       Rotated2D, transmitted_Lz_envelope
scripts/              one script per paper figure
  supplementary/      validation and robustness checks
tests/                pytest suite
data/, figures/       created by the scripts (not tracked by git)
```

## Installation

Kwant and the MUMPS solver are installed from conda-forge (building Kwant with
pip needs a compiler and gives no MUMPS support):

```bash
conda env create -f environment.yml
conda activate orbital
python -c "import kwant.solvers.default as s; print(s.smatrix.__self__.__class__.__module__)"
# should print kwant.solvers.mumps
```

To update an existing environment instead: `conda env update -f environment.yml`.

## Reproducing the figures

Run every script from the repository root, e.g. `python scripts/fig5_spectrum_vs_theta.py`.
Data go to `data/`, figures to `figures/`.

| Figure | Script | Content |
|---|---|---|
| 1 | `fig1_concept.py` | Concept schematic (drawing only) |
| 2 | `fig2_fig3_chain_1d.py` | ⟨L_z⟩_out vs E for several g₂/g₁, inputs \|p_y⟩ and \|+z⟩ |
| 3 | `fig2_fig3_chain_1d.py` | ⟨L_z⟩_out over the (g₂/g₁, E) plane |
| 4 | `fig4_schematic.py`, `fig4_orbital_zoom.py` | 2D model schematic and orbital-basis zoom (drawing only) |
| 5 | `fig5_spectrum_vs_theta.py` | Spectrum of the isolated rotated region vs θ, W = 2 and 4 |
| 6 | `fig6_map_Lz.py` | L_z^max(E, θ) for W = 4 |
| 7 | `fig7_compute_widths.py`, then `fig7_plot_widths.py` | L_z^max(E, θ) for W = 2 … 100 |

**Parameters.**
- 1D chain (Figs. 2–3): t = 1, ε = 0, g₁ = 0.5, d₁ = d₂ = 50.
- 2D junction (Figs. 5–7): V_σ = 1, V_π = −0.3, ε₀ = 0, interface coupling α = 0.5,
  interface cutoff r_c = 1.2a, central length 400 (200 for Fig. 5),
  401 × 401 grid over E ∈ [−1.5, 1.5], θ ∈ [0, π/2].
- W is the number of lattice sites across the strip.

Every `.npz` file stores the full parameter set it was computed with.

**Runtime.** The 2D maps are the expensive part: each pixel is one Kwant
scattering problem. The scripts run pixels in parallel with joblib
(`n_jobs=-1`, all cores). Large widths (W = 50, 100) are much slower than
small ones. `fig7_compute_widths.py` saves each width as soon as it
finishes, so after an interruption you only need to rerun the missing widths
(edit `WIDTHS`).

## Interface model (2D)

At each interface, every lead site is bonded to all rotated-lattice sites
within r_c·a, or to its nearest rotated site if there is none, with hopping
α (a/d)² T(ϕ). Here d and ϕ are the length and direction of the bond, and T is
the Slater–Koster matrix. Because a ≤ r_c·a < √2·a, the junction reduces
exactly to a uniform lead at θ = 0 and π/2. Pass `interface_cutoff=None` to
bond each lead site to its nearest rotated site only.

## Tests

```bash
pytest -q
```

The tests check the Slater–Koster matrices, the orbital pseudospin algebra, the
exact g₂ = −g₁ echo of the 1D chain, the θ = 0 and θ = π/2 limits, the
θ → θ + π/2 equivalence, and full transmission in the perfect-lead limit.

## Data

The data behind the figures will be deposited on Zenodo *(DOI to be added)*.

## License

MIT, see [LICENSE](LICENSE).
