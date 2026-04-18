---
title: Mountain Waves
emoji: 🏔️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Interactive 2-D mountain-wave visualizer (Rust + Dash)
---

# Mountain Waves

An interactive **2-D linear mountain-wave visualizer** with a Rust compute
core and a Python/Dash front end. Live web demo:
<https://huggingface.co/spaces/snesbitt/mountain-waves>.

## Attribution

This project is a port and extension of the interactive MATLAB mountain-wave
model originally written by **Dr. Robert E. (Bob) Hart** (currently at Florida
State University) as a Fall 1995 seminar project for Meteo 574 at Penn State
University under Dr. Peter Bannon. The numerical method, user-interface
design, and example cases all originate from Hart's work.

> Hart, R. E., 1995: *Interactive Model for 2-D Mountain Wave Visualization.*
> Penn State Meteo 574 seminar project.
> Documentation and MATLAB source: <https://moe.met.fsu.edu/~rhart/mtnwave.html>
> Contact: `rhart@fsu.edu`

Bob Hart last updated the MATLAB package in March 2018 for newer MATLAB
versions. This port, by **Steve Nesbitt** (CliMAS, University of Illinois
Urbana–Champaign), preserves the physics and the MATLAB user-experience
conventions while replacing MATLAB-specific glue with a Rust compute core
and a browser-based Dash UI. Any bugs in the port are the port's fault,
not the original model's.

## Theory

### Problem setup

Steady, 2-D, linearized, Boussinesq flow over a smooth obstacle. A mean-state
horizontal wind `U(z)` and potential temperature `θ̄(z)` are prescribed; the
solver returns the stationary perturbation fields `(w', u', p', θ')` forced
by a mountain shape `h(x)`.

### The Taylor–Goldstein equation

Linearizing the Boussinesq momentum, continuity, and thermodynamic equations
about the mean state and taking a single horizontal Fourier mode
`w'(x, z) = Re{ŵ(k, z) · exp(ikx)}` reduces the full PDE system to a single
ODE in `z` per horizontal wavenumber `k`:

```
∂²ŵ/∂z²  +  [ l²(z) − k² ] ŵ  =  0
```

This is the Taylor–Goldstein equation. The coefficient `l²(z)` is the
**Scorer parameter**:

```
l²(z)  =  N²(z) / U²(z)  −  (1/U(z)) · d²U/dz²
```

with `N² = (g/θ̄) · dθ̄/dz` the Brunt–Väisälä frequency squared. `l²` can
go negative — unstable stratification (`N² < 0`) or a locally positive
shear curvature (`d²U/dz² > 0`) can flip its sign, in which case every
wavenumber is evanescent at that height.

### Propagating vs. evanescent, and the Scorer condition

Within a layer of approximately uniform `l²`:

* `l² > k²` — `ŵ` is oscillatory; the wave propagates vertically.
* `l² < k²` — `ŵ` is evanescent; the physical branch decays with height.

A configuration with `l²_lower > l²_upper` admits a band of intermediate
wavenumbers that propagate in the lower layer but are evanescent aloft.
These partially reflect at the transition, are trapped between the surface
and the aloft "lid," and interfere downstream to produce the classic
resonant **lee-wave train** (Scorer 1949).

### Two-layer analytic solver

Each layer has uniform `L`, so `ŵ` is a linear combination of upward and
downward modes per layer. Three conditions close the system:

1. **Surface:** `ŵ(0, k) = ikU(0) ĥ(k)` — the linearized kinematic
   boundary condition `w' = U · ∂h/∂x` for a stationary obstacle of shape
   `h(x)` with Fourier transform `ĥ(k)`.
2. **Interface (`z = H`):** continuity of `ŵ` and `∂ŵ/∂z`.
3. **Top:** Sommerfeld radiation — above the interface, only the
   upward-energy-propagating branch is retained.

This gives closed-form reflection and transmission coefficients that the
Rust core evaluates analytically for every `k` in parallel. It is a direct
port of Hart's `tlwplot.m`.

### Multi-layer propagator-matrix solver

For arbitrary `U(z), θ̄(z)` the atmosphere is discretized into a fine stack
of sub-layers, each with its own locally-uniform `l²_j`. Inside each
sub-layer the exact analytic solution is still available (oscillatory if
`l²_j > k²`, evanescent if `l²_j < k²`). A 2×2 transfer matrix propagates
`(ŵ, ∂ŵ/∂z)` between sub-layer interfaces; sweeping bottom-to-top yields the
full vertical structure per `k`. The top boundary is again a radiation
condition, here expressed in a `(σ, ŵ)` basis that stays well-conditioned
when `l² < k²` aloft (pure decay).

### Synthesis: inverse Fourier transform

After solving per-`k`, the physical fields are reconstructed by trapezoidal
inverse Fourier transform over the wavenumber grid:

```
w'(x, z)  =  ∫ ŵ(k, z) · ĥ(k) · exp(ikx) dk  +  c.c.
```

The horizontal wind perturbation `u'` is recovered from linearized
continuity `ik û + ∂ŵ/∂z = 0` directly in spectral space:

```
u'(x, z)  =  ∫  [ −(i/k) · ∂ŵ/∂z ]  ·  ĥ(k) · exp(ikx) dk  +  c.c.
```

evaluated on the same wavenumber grid as `w'`. Both fields are displayed in
the UI as colored contour maps side-by-side.

### Linearized streamlines

For small perturbations, a streamline originating at upstream height `z₀`
is vertically displaced by

```
δz(x, z₀)  =  (1 / U(z₀)) · ∫₋∞ˣ w'(x', z₀) dx'
```

The solver integrates this along `x` for a set of evenly spaced release
heights to produce the overlay. When `U` varies with height, the
per-streamline advection speed is `U(z₀)`, not a single surface value.

### What linear theory cannot capture

This is a small-amplitude, inviscid, non-rotating, 2-D model. **Downslope
windstorms, hydraulic jumps, wave breaking, and critical-level nonlinear
amplification are fundamentally nonlinear phenomena and are not captured
here.** The "Near-downslope" preset in the two-layer mode approaches that
regime as a forcing-parameter lookup but will systematically underpredict
the surface winds observed in real events.

## What's in the port

1. **Two-layer analytic solver** — direct port of Hart's `tlwplot.m`.
   Uniform Scorer parameter in each layer, analytic reflection /
   transmission at the interface, Fourier summation over horizontal
   wavenumbers.
2. **Multi-layer profile solver** — takes arbitrary `θ̄(z)` and `u(z)` and
   solves the Taylor–Goldstein ODE per wavenumber via the propagator-matrix
   sweep described above.
3. **Interactive profile editor** — drag control points on `θ̄(z)` and
   `u(z)` graphs to prescribe arbitrary profiles; results re-render on the
   next *Analyze flow* click.
4. **u′ heatmap** — rendered alongside `w'` so you can see where the wave
   train speeds up or slows the low-level flow.
5. **Latitude slider** — exposes the Coriolis parameter so the Rossby-number
   readout reflects the user's chosen latitude (the MATLAB original
   hard-coded the pole).
6. **HRRR initialization** (Profile tab) — fetch the nearest HRRR analysis
   column from NOAA's public AWS bucket to seed `θ̄(z)` and `u(z)` from a
   real atmosphere, then drag the gold circles to edit further.

## Layout

```
Mountain Waves/
├── Cargo.toml                       # Rust crate manifest
├── pyproject.toml                   # maturin build + project metadata
├── Dockerfile                       # Hugging Face Spaces build
├── src/lib.rs                       # Rust compute core (PyO3 bindings)
├── python/mountain_waves/
│   ├── __init__.py
│   ├── app.py                       # Dash app entry point
│   ├── reference.py                 # pure-Python reference solver (fallback)
│   ├── profile.py                   # θ(z)/u(z) → Scorer parameter utilities
│   ├── hrrr.py                      # HRRR column fetcher (AWS)
│   └── solver.py                    # picks Rust or Python backend
├── run.py                           # launcher: `uv run python run.py`
├── validate.py                      # compares Rust vs. reference vs. MATLAB
└── tlwmenu.m / tlwplot.m / stream.m # Hart's original MATLAB sources
```

## Quick start (uv)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package + environment
manager. One-shot setup from the project root:

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
#   or: brew install uv     (macOS)
#   or: pipx install uv

# 2. Create a virtual environment pinned to Python 3.11
uv venv --python 3.11
source .venv/bin/activate          # or `.venv\Scripts\activate` on Windows

# 3. Install Python dependencies from pyproject.toml
uv sync

# 4. (Optional, for the Rust backend) install the Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 5. Build the Rust extension into the active venv
uv run maturin develop --release --uv

# 6. Launch the web UI
uv run python run.py               # http://127.0.0.1:8050
```

`uv sync` reads `pyproject.toml` and installs every runtime dependency
(`numpy`, `scipy`, `dash`, `plotly`, plus the HRRR stack `boto3`, `xarray`,
`cfgrib`, `eccodes`, `eccodeslib`). `maturin develop --uv` installs the
compiled extension into the uv-managed environment; `--release` enables
optimizations (typically 20×–30× faster than the pure-Python fallback).

Dev extras (`pytest` etc.):

```bash
uv sync --extra dev
```

### Pure-Python fallback (no Rust toolchain needed)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv run python run.py
```

The launcher detects the missing Rust extension and transparently uses
`python/mountain_waves/reference.py`. Every numerical result matches the
Rust core within floating-point tolerance; it just runs slower.

### pip alternative

```bash
pip install -e .            # installs deps; maturin builds _core if rustc is found
python run.py
```

## Validation

```bash
uv run python validate.py
```

Runs the solver on three canonical cases (uniform atmosphere, trapped lee
waves, and a profile-based approximation of the trapped case) and asserts
reasonable amplitudes. When both the Rust and Python backends are built
it also cross-checks them element-wise.

## Docker / Hugging Face Space

The `Dockerfile` is a two-stage build tuned for Hugging Face Spaces
(SDK `docker`, port 7860). Stage 1 builds a release wheel of the Rust
extension on `rust:1.82-slim`; stage 2 installs that wheel onto
`python:3.11-slim` plus the runtime Python dependencies. The live Space
is at <https://huggingface.co/spaces/snesbitt/mountain-waves>.

## References

* Scorer, R. S., 1949: *Theory of waves in the lee of mountains*.
  Q. J. R. Meteorol. Soc., **75**, 41–56.
* Booker, J. R., and F. P. Bretherton, 1967: *The critical layer for
  internal gravity waves in a shear flow.* J. Fluid Mech., **27**,
  513–539. Motivates the "Critical layer at 2 km" preset: wave
  attenuation across `U = 0` scales as `exp(−2π √(Ri − 1/4))` for
  `Ri > 1/4`.
* Durran, D. R., 1986: *Mountain Waves*, in *Mesoscale Meteorology and
  Forecasting*, American Meteorological Society, pp. 472–492.
* Hart, R. E., 1995: *Interactive Model for 2-D Mountain Wave
  Visualization.* Penn State Meteo 574 seminar project.
* Doyle, J. D., and D. R. Durran, 2002: *The dynamics of mountain-wave-
  induced rotors.* J. Atmos. Sci., **59**, 186–201. Observational and
  numerical context for the wind-reversal preset; the T-REX / Sierra
  Rotors campaign (Grubišić et al. 2008, BAMS **89**, 1513–1533)
  documented the atmospheric structures this preset is meant to
  caricature.

## License

Original MATLAB model and physics © Robert E. Hart. The Rust + Python
port is provided for research and teaching purposes under the MIT license.
If you use this tool in published work, please cite Hart (1995) and the
[FSU documentation page](https://moe.met.fsu.edu/~rhart/mtnwave.html).
