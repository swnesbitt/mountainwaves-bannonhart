"""Pure-Python reference implementation of the mountain-wave solver.

This module is a direct transcription of the MATLAB routines ``tlwplot.m``
and ``stream.m`` written by Dr. Robert E. (Bob) Hart in 1995 as a Penn
State Meteo 574 seminar project (see
https://moe.met.fsu.edu/~rhart/mtnwave.html). The two-layer routine here
mirrors Hart's MATLAB code line-for-line (reformulated in NumPy); the
multi-layer routine is a natural generalization using the same Fourier +
transfer-matrix scheme.

It exists for two reasons:

1. It's the fallback when the Rust extension isn't built.
2. It's the reference used by ``validate.py`` to confirm the Rust core
   produces bit-similar results.

The multi-layer solver implements the same transfer-matrix scheme as the
Rust version so they can be compared.
"""

from __future__ import annotations

import numpy as np

G = 9.80665


# ---------------------------------------------------------------------------
# Two-layer analytic solver (port of tlwplot.m)
# ---------------------------------------------------------------------------


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
    """Return ``(x, z, w)`` for flow over a witch-of-Agnesi mountain.

    Arrays match the MATLAB conventions: ``x`` has shape ``(npts + 1,)``,
    ``z`` has shape ``(npts + 1,)``, and ``w`` has shape
    ``(z.size, x.size)`` indexed as ``w[z_index, x_index]``.
    """
    dk = 0.367 / a
    nk = max(1, int((maxk - mink) // dk))

    minx = -0.25 * xdom
    maxx = 0.75 * xdom

    dx = (maxx - minx) / npts
    dz = zdom / npts

    x = minx + dx * np.arange(npts + 1)
    z = dz * np.arange(npts + 1)

    X, Z = np.meshgrid(x, z)  # shape (npts+1, npts+1), Z[i, j] varies along i

    matrix1 = np.zeros_like(X, dtype=complex)
    matrix3 = np.zeros_like(X, dtype=complex)
    ht = 0.0

    for kloop in range(nk + 1):
        kk = mink + dk * kloop
        m = np.sqrt(complex(l_lower ** 2 - kk ** 2))
        n = np.sqrt(complex(kk ** 2 - l_upper ** 2))
        denom = m + 1j * n
        if abs(denom) < 1e-300:
            r = complex(9e99, 0.0)
        else:
            r = (m - 1j * n) / denom
        R = r * np.exp(2j * m * h)
        A = (1 + r) * np.exp(h * n + 1j * h * m) / (1 + R)
        C = 1.0 / (1 + R)
        D = R * C

        ksign = abs(kk)
        hs = np.pi * a * ho * np.exp(-a * ksign)
        ht += np.pi * dk * a * np.exp(-a * ksign) if kloop > 0 else 0.0

        above = A * np.exp(-Z * n) * (Z > h)
        below = (C * np.exp(1j * Z * m) + D * np.exp(-1j * Z * m)) * (Z <= h)
        matrix2 = (-1j * kk * hs * u * (above + below)) * np.exp(-1j * X * kk)

        if kloop > 0:
            matrix3 += 0.5 * (matrix1 + matrix2) * dk
        matrix1 = matrix2

    if ht == 0.0:
        ht = 1.0
    w = np.real(matrix3 / ht)
    return x, z, w


# ---------------------------------------------------------------------------
# Multi-layer profile solver
# ---------------------------------------------------------------------------


def scorer_from_profile(z_profile, u_profile, theta_profile):
    """Return Scorer parameter L^2(z) computed from profile data."""
    z = np.asarray(z_profile, dtype=float)
    u = np.asarray(u_profile, dtype=float)
    theta = np.asarray(theta_profile, dtype=float)
    n = z.size
    l2 = np.zeros(n)
    for i in range(n):
        if i == 0:
            dthdz = (theta[1] - theta[0]) / (z[1] - z[0])
            if n >= 3:
                h1 = z[1] - z[0]
                h2 = z[2] - z[1]
                d2u = 2.0 * (u[2] * h1 - u[1] * (h1 + h2) + u[0] * h2) / (h1 * h2 * (h1 + h2))
            else:
                d2u = 0.0
        elif i == n - 1:
            dthdz = (theta[-1] - theta[-2]) / (z[-1] - z[-2])
            if n >= 3:
                h1 = z[-2] - z[-3]
                h2 = z[-1] - z[-2]
                d2u = 2.0 * (u[-1] * h1 - u[-2] * (h1 + h2) + u[-3] * h2) / (h1 * h2 * (h1 + h2))
            else:
                d2u = 0.0
        else:
            h1 = z[i] - z[i - 1]
            h2 = z[i + 1] - z[i]
            dthdz = (
                theta[i + 1] * h1 ** 2
                - theta[i - 1] * h2 ** 2
                + theta[i] * (h2 ** 2 - h1 ** 2)
            ) / (h1 * h2 * (h1 + h2))
            d2u = 2.0 * (u[i + 1] * h1 - u[i] * (h1 + h2) + u[i - 1] * h2) / (h1 * h2 * (h1 + h2))
        n2 = (G / theta[i]) * dthdz
        uu = u[i]
        if abs(uu) > 1e-6:
            l2[i] = n2 / uu ** 2 - d2u / uu
        else:
            l2[i] = 0.0
    return l2


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
    """Arbitrary u(z)/theta(z) mountain-wave solver using transfer matrices.

    The atmosphere is split into piecewise-constant L^2 layers centered on
    the profile points. Inside each layer the wave-transform equation
    reduces to an exponential ansatz; continuity of ŵ and ŵ' at interfaces
    plus a radiation / decay condition aloft closes the system.
    """
    zp = np.asarray(z_profile, dtype=float)
    up = np.asarray(u_profile, dtype=float)
    tp = np.asarray(theta_profile, dtype=float)
    l2 = scorer_from_profile(zp, up, tp)
    u_surface = float(up[0])

    nlayers = zp.size
    layer_bot = np.empty(nlayers)
    layer_top = np.empty(nlayers)
    for j in range(nlayers):
        layer_bot[j] = 0.0 if j == 0 else 0.5 * (zp[j - 1] + zp[j])
        layer_top[j] = np.inf if j == nlayers - 1 else 0.5 * (zp[j] + zp[j + 1])

    dk = 0.367 / a
    nk = max(1, int((maxk - mink) // dk))
    nslab = nk + 1

    minx = -0.25 * xdom
    maxx = 0.75 * xdom
    dx = (maxx - minx) / npts
    dz = zdom / npts
    x = minx + dx * np.arange(npts + 1)
    z = dz * np.arange(npts + 1)

    # Layer index for each vertical grid point
    layer_of = np.zeros(z.size, dtype=int)
    for j, zj in enumerate(z):
        idx = nlayers - 1
        for lj in range(nlayers):
            if zj < layer_top[lj]:
                idx = lj
                break
        layer_of[j] = idx

    matrix1 = np.zeros((z.size, x.size), dtype=complex)
    matrix3 = np.zeros((z.size, x.size), dtype=complex)
    ht = 0.0

    for kloop in range(nslab):
        kk = mink + dk * kloop
        ksign = abs(kk)
        hs = np.pi * a * ho * np.exp(-a * ksign)
        if kloop > 0:
            ht += np.pi * dk * a * np.exp(-a * ksign)

        if kk == 0.0:
            matrix2 = np.zeros_like(matrix1)
        else:
            # Principal-branch sigma: in each layer, the "a" coefficient
            # multiplies exp(-sigma*dz), which is always the outgoing /
            # decaying branch when Im(sigma) >= 0.
            sigma = np.empty(nlayers, dtype=complex)
            for j in range(nlayers):
                s = np.sqrt(complex(kk ** 2 - l2[j]))
                if s.imag < 0:
                    s = -s
                sigma[j] = s

            aj = np.zeros(nlayers, dtype=complex)
            bj = np.zeros(nlayers, dtype=complex)
            aj[-1] = 1.0
            bj[-1] = 0.0
            for j in range(nlayers - 2, -1, -1):
                dz_j = layer_top[j] - layer_bot[j]
                e_minus = np.exp(-sigma[j] * dz_j)
                e_plus = np.exp(sigma[j] * dz_j)
                alpha = aj[j + 1] + bj[j + 1]
                beta = -aj[j + 1] + bj[j + 1]
                ratio = sigma[j + 1] / sigma[j]
                aj[j] = 0.5 * (alpha - ratio * beta) * e_plus
                bj[j] = 0.5 * (alpha + ratio * beta) * e_minus

            w_surface = aj[0] + bj[0]
            if abs(w_surface) < 1e-300:
                amp = 0.0 + 0.0j
            else:
                amp = -1j * kk * u_surface * hs / w_surface
            aj *= amp
            bj *= amp

            # Build ŵ on the vertical grid.
            zfac = np.zeros(z.size, dtype=complex)
            for j in range(z.size):
                lj = layer_of[j]
                dz_l = z[j] - layer_bot[lj]
                zfac[j] = aj[lj] * np.exp(-sigma[lj] * dz_l) + bj[lj] * np.exp(sigma[lj] * dz_l)

            xfac = np.exp(-1j * x * kk)
            matrix2 = np.outer(zfac, xfac)

        if kloop > 0:
            matrix3 += 0.5 * (matrix1 + matrix2) * dk
        matrix1 = matrix2

    if ht == 0.0:
        ht = 1.0
    w = np.real(matrix3 / ht)
    return x, z, w


# ---------------------------------------------------------------------------
# Streamline tracer (port of stream.m)
# ---------------------------------------------------------------------------


def streamlines(x, z, u: float, w, num: int = 10):
    """Return ``num`` streamlines as ``[(xs, ys), ...]`` polylines."""
    x = np.asarray(x)
    z = np.asarray(z)
    w = np.asarray(w)
    nx = x.size
    nz = z.size
    if nx < 2 or nz < 2 or num == 0:
        return []
    minx = float(x[0])
    dx = float(x[1] - x[0])
    tstep = dx / u
    dh = nz / num

    lines = []
    for j in range(num):
        ycell = 1.0 + dh * j
        if ycell < 1.0:
            ycell = 1.0
        if ycell > nz:
            ycell = nz
        yci = int(round(ycell) - 1)
        yci = max(0, min(nz - 1, yci))
        xs = np.empty(nx)
        ys = np.empty(nx)
        xs[0] = minx
        ys[0] = z[yci]
        for i in range(1, nx):
            xs[i] = x[i]
            ys[i] = ys[i - 1] + tstep * w[yci, i]
        lines.append((xs, ys))
    return lines
