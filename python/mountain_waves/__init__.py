"""Interactive 2-D mountain-wave model.

Python/Rust port of Dr. Robert E. (Bob) Hart's 1995 MATLAB mountain-wave
model (``tlwplot.m`` / ``tlwmenu.m`` / ``stream.m``). The numerical scheme,
user-interface layout, and example cases originate from Hart's Penn State
Meteo 574 seminar project. Documentation and MATLAB sources:
https://moe.met.fsu.edu/~rhart/mtnwave.html (contact: ``rhart@fsu.edu``).

Top-level package. Numerical entry points live in :mod:`mountain_waves.solver`
which selects the Rust extension if available and otherwise falls back to
the pure-Python reference implementation.
"""

from .solver import (
    compute_two_layer,
    compute_from_profile,
    streamlines,
    backend_name,
)

__all__ = [
    "compute_two_layer",
    "compute_from_profile",
    "streamlines",
    "backend_name",
]
