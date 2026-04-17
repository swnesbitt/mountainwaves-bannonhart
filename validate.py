#!/usr/bin/env python3
"""Validate Rust vs. pure-Python solver, and sanity-check physics.

Run with: python validate.py

If the Rust extension hasn't been built, only the pure-Python code is
exercised — useful for catching mistakes in the port of tlwplot.m.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from mountain_waves import compute_two_layer, compute_from_profile, streamlines, backend_name
from mountain_waves import reference as ref
from mountain_waves.profile import (
    default_profile_heights,
    default_theta_profile,
    default_u_profile,
)


def describe(name, w):
    print(f"  {name:<28} min={w.min():+.4f}  max={w.max():+.4f}  rms={np.sqrt((w**2).mean()):.4f}")


def case_uniform():
    # Example 1: uniform atmosphere. No wave trapping.
    return dict(l_upper=4e-4, l_lower=4e-4, u=20.0, h=3500.0, a=2500.0, ho=500.0,
                xdom=40000.0, zdom=10000.0, mink=0.0, maxk=30.0 / 2500.0, npts=100)


def case_trapped():
    # Example 2: trapped lee waves.
    return dict(l_upper=4e-4, l_lower=10e-4, u=20.0, h=3500.0, a=2500.0, ho=500.0,
                xdom=40000.0, zdom=10000.0, mink=0.0, maxk=30.0 / 2500.0, npts=100)


def main() -> int:
    print(f"Active backend: {backend_name()}")

    # ---- Python reference sanity: uniform atmosphere, w ≈ odd in x near crest
    print("\n[1] Two-layer, uniform atmosphere (pure-Python reference):")
    p = case_uniform()
    x, z, w = ref.compute_two_layer(**p)
    assert w.shape == (101, 101)
    describe("w (reference)", w)

    print("\n[2] Two-layer, trapped-wave case:")
    pt = case_trapped()
    x, z, w_ref = ref.compute_two_layer(**pt)
    describe("w (reference)", w_ref)

    # Expect substantial wave amplitude in the lee (x > 0) for the trapped case.
    # Find amplitude in z ~ 1-2 km and x ~ 5-20 km.
    xi = (x >= 5000) & (x <= 20000)
    zi = (z >= 500) & (z <= 2000)
    lee_rms = np.sqrt((w_ref[np.ix_(zi, xi)] ** 2).mean())
    print(f"  lee-wave rms(1-2 km, 5-20 km) = {lee_rms:.3f} m/s")
    assert lee_rms > 0.05, "Trapped-wave case showed suspiciously weak lee waves."

    # ---- Rust vs Python, if Rust is available
    try:
        from mountain_waves import _core  # type: ignore
        has_rust = True
    except ImportError:
        has_rust = False

    if has_rust:
        print("\n[3] Rust vs. Python two-layer (trapped case):")
        _, _, w_rust = _core.compute_two_layer(**pt)
        err = np.max(np.abs(w_rust - w_ref))
        print(f"  max|w_rust - w_python| = {err:.2e}")
        assert err < 1e-6, f"Rust/Python disagreement too large: {err}"
    else:
        print("\n[3] (skipped — Rust extension not built)")

    # ---- Profile solver: should approximate two-layer as profile resolution → ∞
    print("\n[4] Profile solver — trapped-wave profile:")
    zs = default_profile_heights(10.0, 17)
    us = np.full_like(zs, 20.0)  # constant wind
    # Construct theta so that N^2/u^2 approximates L_lower/L_upper below/above 3.5 km.
    # L_lower^2 = N^2/u^2 - (1/u) d2u/dz2, constant u ⇒ second term 0.
    # Want L^2_lower = 10e-4^2 -> N^2 = L^2 * u^2.
    N2_lower = (10e-4) ** 2 * 20.0 ** 2
    N2_upper = (4e-4) ** 2 * 20.0 ** 2
    thetas = np.empty_like(zs)
    thetas[0] = 290.0
    for i in range(1, zs.size):
        dz = zs[i] - zs[i - 1]
        n2 = N2_lower if zs[i] <= 3500.0 else N2_upper
        thetas[i] = thetas[i - 1] + thetas[i - 1] * n2 / 9.80665 * dz

    x, z, w_prof = ref.compute_from_profile(
        zs, us, thetas,
        a=2500.0, ho=500.0, xdom=40000.0, zdom=10000.0,
        mink=0.0, maxk=30.0 / 2500.0, npts=100,
    )
    describe("w (profile solver)", w_prof)
    # Compare RMS to two-layer case — should be broadly similar.
    rms_prof = np.sqrt(np.mean(w_prof ** 2))
    rms_ref = np.sqrt(np.mean(w_ref ** 2))
    ratio = rms_prof / rms_ref
    print(f"  rms(profile) / rms(2-layer) = {ratio:.3f}")
    assert 0.3 < ratio < 3.0, "Profile solver RMS differs unreasonably from two-layer reference."

    # ---- Streamline tracer: the first line should sweep over the mountain crest.
    print("\n[5] Streamline tracer:")
    lines = streamlines(x, z, 20.0, w_ref, num=10)
    assert len(lines) == 10
    xs0, ys0 = lines[0]
    assert xs0[0] < 0 < xs0[-1]
    print(f"  first streamline: x ∈ [{xs0[0]:.0f}, {xs0[-1]:.0f}] m, y ∈ [{ys0.min():.1f}, {ys0.max():.1f}] m")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
