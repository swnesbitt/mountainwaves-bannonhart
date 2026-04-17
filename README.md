# Mountain Waves

An interactive 2-D mountain-wave visualization tool, with a **Rust compute
core** and a **Python/Dash interactive front end**.

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
versions. This port preserves the physics and the MATLAB user-experience
conventions while replacing MATLAB-specific glue with a Rust compute core and
a browser-based Dash UI. Any bugs in the port are the port's fault, not the
original model's.

## What's in the port

1. **Two-layer analytic solver** — a direct port of Hart's `tlwplot.m`.
   Uniform Scorer parameter in each layer, analytic reflection / transmission
   at the interface, Fourier summation over horizontal wavenumbers.
2. **Multi-layer profile solver** — takes arbitrary potential-temperature
   `theta(z)` and zonal-wind `u(z)` profiles, converts them to a
   height-dependent Scorer parameter `L^2(z)`, and solves the resulting
   Taylor–Goldstein ODE per wavenumber via a propagator-matrix sweep.
3. **Interactive profile editor** — drag control points on `theta(z)` and
   `u(z)` graphs to prescribe arbitrary profiles; results re-render on the
   next "Analyze flow" click.
4. **Latitude slider** — exposes the Coriolis parameter so the Rossby-number
   readout reflects the user's chosen latitude (the original MATLAB hard-coded
   the pole).

## Layout

```
Mountain Waves/
├── Cargo.toml                       # Rust crate manifest
├── pyproject.toml                   # maturin build + uv project metadata
├── src/lib.rs                       # Rust compute core (PyO3 bindings)
├── python/mountain_waves/
│   ├── __init__.py
│   ├── app.py                       # Dash app entry point
│   ├── reference.py                 # pure-Python reference solver (fallback)
│   ├── profile.py                   # theta/u -> Scorer parameter utilities
│   └── solver.py                    # picks Rust or Python backend
├── run.py                           # launcher: `uv run python run.py`
├── validate.py                      # compares Rust vs. reference vs. MATLAB
└── (original) stream.m / tlwmenu.m / tlwplot.m    # Hart's MATLAB sources
```

## Quick start (uv)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package + environment
manager. One-shot setup from the project root:

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
#   or: brew install uv     macOS
#   or: pipx install uv

# 2. Create a virtual environment pinned to a Python 3.11
uv venv --python 3.11
source .venv/bin/activate          # or `.venv\Scripts\activate` on Windows

# 3. Install Python dependencies from pyproject.toml
uv sync

# 4. (Optional but recommended) Install the Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 5. Build the Rust extension into the active venv
uv run maturin develop --release --uv

# 6. Launch the web UI
uv run python run.py               # http://127.0.0.1:8050
```

`uv sync` reads `pyproject.toml` and installs every runtime dependency
(`numpy`, `scipy`, `dash`, `plotly`). `maturin develop --uv` asks maturin to
install the compiled extension into the `uv`-managed environment; `--release`
turns on optimizations (typically ~20× faster than the Python fallback).

If you only want to install extras (`maturin`, `pytest`):

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

The app detects the missing Rust extension and transparently uses
`python/mountain_waves/reference.py`. Every numerical result is identical to
the Rust core within floating-point tolerance; it just runs slower.

### pip alternative

If you prefer plain pip, `pip install -e .` works too, followed by
`maturin develop --release` to build the extension.

## Validation

```bash
uv run python validate.py
```

Runs the reference solver on three canonical cases (uniform atmosphere,
trapped lee waves, and a profile-based approximation of the trapped case) and
asserts reasonable amplitudes. When both the Rust and Python backends are
available it also cross-checks them element-wise.

## References

* Scorer, R. S., 1949: *Theory of waves in the lee of mountains*. Q. J. R.
  Meteorol. Soc., **75**, 41–56.
* Durran, D. R., 1986: *Mountain Waves*, in Mesoscale Meteorology and
  Forecasting, American Meteorological Society, pp. 472–492.
* Hart, R. E., 1995: *Interactive Model for 2-D Mountain Wave Visualization.*
  Penn State Meteo 574 seminar project.

## License

Original MATLAB model and physics © Robert E. Hart. The Rust + Python port
is provided for research and teaching purposes under the MIT license. If you
use this tool in published work, please cite Hart (1995) and the
[FSU documentation page](https://moe.met.fsu.edu/~rhart/mtnwave.html).
