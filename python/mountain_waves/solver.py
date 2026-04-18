"""Backend selector: prefer the Rust extension, fall back to pure Python.

Both paths expose identical call signatures so the Dash app doesn't need
to care which one is running. ``backend_name()`` tells you which it is.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:  # pragma: no cover - import-time branching
    from . import _core as _rust  # type: ignore

    _BACKEND = "rust"
except ImportError:
    _rust = None  # type: ignore
    _BACKEND = "python"

from . import reference as _ref


def backend_name() -> str:
    """Return ``"rust"`` if the compiled extension is in use, else ``"python"``."""
    return _BACKEND


def _ensure_4tuple_two_layer(result, args):
    """Coerce older Rust binaries (returning 3-tuples) into the 4-tuple ABI.

    The Rust extension was extended to return u_prime alongside w. If the
    user hasn't rebuilt the .so yet, fall back to the Python reference for
    the full solve so u_prime is still correct rather than silently zero.
    """
    if isinstance(result, tuple) and len(result) == 4:
        return result
    # Stale binary — recompute via Python reference so u' is accurate.
    return _ref.compute_two_layer(*args)


def _ensure_4tuple_profile(result, args):
    if isinstance(result, tuple) and len(result) == 4:
        return result
    return _ref.compute_from_profile(*args)


def compute_two_layer(
    l_upper: float,
    l_lower: float,
    u: float,
    h: float,
    a: float,
    ho: float,
    xdom: float,
    zdom: float,
    mink: float,
    maxk: float,
    npts: int = 100,
):
    args = (l_upper, l_lower, u, h, a, ho, xdom, zdom, mink, maxk, npts)
    if _rust is not None:
        return _ensure_4tuple_two_layer(_rust.compute_two_layer(*args), args)
    return _ref.compute_two_layer(*args)


def compute_from_profile(
    z_profile,
    u_profile,
    theta_profile,
    a: float,
    ho: float,
    xdom: float,
    zdom: float,
    mink: float,
    maxk: float,
    npts: int = 100,
):
    zp = np.ascontiguousarray(np.asarray(z_profile, dtype=np.float64))
    up = np.ascontiguousarray(np.asarray(u_profile, dtype=np.float64))
    tp = np.ascontiguousarray(np.asarray(theta_profile, dtype=np.float64))
    args = (zp, up, tp, a, ho, xdom, zdom, mink, maxk, npts)
    if _rust is not None:
        return _ensure_4tuple_profile(_rust.compute_from_profile(*args), args)
    return _ref.compute_from_profile(*args)


def streamlines(x, z, u, w, num: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Trace ``num`` linearized streamlines through the ``w(x, z)`` field.

    ``u`` can be a scalar (uniform mean flow, as in the two-layer analytic)
    or a 1-D array of length ``nz`` giving the mean wind at each render-grid
    height. When an array is given we route through the Python tracer so the
    per-streamline advection speed is ``U(z₀)``, not ``U_surface``.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
    # Rust streamlines() only accepts scalar u. For array-valued u (profile
    # mode with shear) dispatch to the Python tracer — it's only num*nx
    # floating-point adds, so the perf difference is negligible.
    if u_arr.size > 1:
        return _ref.streamlines(x, z, u_arr, w, num)
    u_scalar = float(u_arr[0])
    if _rust is not None:
        return _rust.streamlines(x, z, u_scalar, w, num)
    return _ref.streamlines(x, z, u_scalar, w, num)
