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
    if _rust is not None:
        return _rust.compute_two_layer(
            l_upper, l_lower, u, h, a, ho, xdom, zdom, mink, maxk, npts
        )
    return _ref.compute_two_layer(
        l_upper, l_lower, u, h, a, ho, xdom, zdom, mink, maxk, npts
    )


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
    if _rust is not None:
        return _rust.compute_from_profile(
            zp, up, tp, a, ho, xdom, zdom, mink, maxk, npts
        )
    return _ref.compute_from_profile(
        zp, up, tp, a, ho, xdom, zdom, mink, maxk, npts
    )


def streamlines(x, z, u: float, w, num: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    z = np.ascontiguousarray(np.asarray(z, dtype=np.float64))
    w = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
    if _rust is not None:
        return _rust.streamlines(x, z, u, w, num)
    return _ref.streamlines(x, z, u, w, num)
