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

import math

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
    """Return ``(x, z, w, u_prime)`` for flow over a witch-of-Agnesi mountain.

    Arrays match the MATLAB conventions: ``x`` has shape ``(npts + 1,)``,
    ``z`` has shape ``(npts + 1,)``, and both ``w`` and ``u_prime`` have
    shape ``(z.size, x.size)`` indexed as ``[z_index, x_index]``.

    ``u_prime`` is the wave-induced horizontal wind perturbation, obtained
    from linearized continuity ``∂u'/∂x + ∂w/∂z = 0``. Per wavenumber,
    ``u'_k = −(i/k) · ∂ŵ_k/∂z``; we analytically differentiate the two-layer
    eigenfunctions (``A e^{−n z}`` above the interface, ``C e^{i m z} +
    D e^{−i m z}`` below) and accumulate the same trapezoidal k-integration
    used for ``w``.
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
    matrix1_u = np.zeros_like(X, dtype=complex)
    matrix3_u = np.zeros_like(X, dtype=complex)
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

        # Analytic z-derivative of the eigenfunctions (the two branches are
        # continuous at z=h by construction, so the jump in the step factor
        # contributes nothing to the derivative inside each region).
        dabove = (-n) * A * np.exp(-Z * n) * (Z > h)
        dbelow = (1j * m) * (C * np.exp(1j * Z * m) - D * np.exp(-1j * Z * m)) * (Z <= h)
        # u'_k(x, z) = (-i/k) · ∂ŵ_k/∂z. Combining the −ik factor baked into
        # matrix2's ŵ formula with the −i/k in front yields −hs·U·∂(above+
        # below)/∂z. This removes the apparent 1/k singularity at k=0 — the
        # result is analytic there — and avoids division edge cases.
        matrix2_u = (-hs * u) * (dabove + dbelow) * np.exp(-1j * X * kk)

        if kloop > 0:
            matrix3 += 0.5 * (matrix1 + matrix2) * dk
            matrix3_u += 0.5 * (matrix1_u + matrix2_u) * dk
        matrix1 = matrix2
        matrix1_u = matrix2_u

    if ht == 0.0:
        ht = 1.0
    w = np.real(matrix3 / ht)
    u_prime = np.real(matrix3_u / ht)
    return x, z, w, u_prime


# ---------------------------------------------------------------------------
# Multi-layer profile solver
# ---------------------------------------------------------------------------


# Minimum |U| used in the Scorer-parameter denominator. In pure linear
# theory, U(z) = 0 is a critical level where l² = N²/U² − (U″/U) is
# singular; linear Scorer/Taylor-Goldstein cannot honestly solve across
# such a level. In a teaching tool we *want* students to be able to set
# up a wind-reversal profile and see what happens away from the critical
# level rather than have the whole solve NaN out. We clamp |U| to this
# floor (preserving sign) when evaluating the Scorer coefficients. Away
# from U≈0 this is a no-op; within ±0.5 m/s it caps l² at a large but
# finite value and the UI emits a "critical level detected" warning so
# nobody is misled into treating the capped zone as physical.
U_FLOOR_SCORER = 0.5  # m/s


def _u_clamped_for_scorer(uu: float) -> float:
    """Return ``uu`` with ``|uu|`` lifted to ``U_FLOOR_SCORER``; sign preserved."""
    if uu >= 0.0:
        return max(uu, U_FLOOR_SCORER)
    return min(uu, -U_FLOOR_SCORER)


def scorer_from_profile(z_profile, u_profile, theta_profile):
    """Return Scorer parameter L^2(z) computed from profile data.

    Handles wind reversals (sign changes in ``u_profile``) by clamping the
    magnitude of ``U`` at ``U_FLOOR_SCORER`` when it evaluates the
    ``N²/U² − U″/U`` combination. This keeps the solver numerically
    well-behaved across a critical level (``U = 0``) at the cost of a
    physically sharp feature there — see the ``critical_levels`` helper
    below for the companion diagnostic surfaced in the UI.
    """
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
        uu = _u_clamped_for_scorer(u[i])
        l2[i] = n2 / uu ** 2 - d2u / uu
    return l2


def critical_levels(z_profile, u_profile):
    """Return heights (m) where ``u_profile`` crosses zero, linearly interpolated.

    A "critical level" for steady, 2-D, horizontally uniform linear mountain
    waves is a height where the mean flow vanishes (``U = 0``). Linear
    Scorer/Taylor-Goldstein theory is singular there — wave energy is
    absorbed rather than propagated (Booker & Bretherton 1967) — so any
    result the solver returns *near* a critical level should be read as
    "this is where the linear model breaks down," not as a prediction.

    Caller (the Dash UI) surfaces the returned heights in a diagnostics
    badge so students can see where their profile is violating the
    assumptions of the model.
    """
    z = np.asarray(z_profile, dtype=float)
    u = np.asarray(u_profile, dtype=float)
    heights = []
    for i in range(1, z.size):
        u_prev, u_curr = u[i - 1], u[i]
        # Treat exact zeros as crossings at that sample.
        if u_curr == 0.0:
            heights.append(float(z[i]))
            continue
        if u_prev == 0.0:
            # Already recorded by the previous iteration's "u_curr == 0" branch.
            continue
        if (u_prev > 0.0 and u_curr < 0.0) or (u_prev < 0.0 and u_curr > 0.0):
            # Linear interp to the zero crossing.
            t = u_prev / (u_prev - u_curr)
            heights.append(float(z[i - 1] + t * (z[i] - z[i - 1])))
    return heights


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

    Returns ``(x, z, w, u_prime)``. The atmosphere is split into
    piecewise-constant L² layers centered on the profile points. Inside
    each layer the wave-transform equation reduces to an exponential
    ansatz; continuity of ŵ and ŵ' at interfaces plus a radiation / decay
    condition aloft closes the system.

    The wave-induced horizontal wind perturbation ``u_prime`` is obtained
    in the same Fourier loop: for each wavenumber ``k ≠ 0`` we take the
    analytic z-derivative of the per-layer ŵ basis (``σ_j · (−a_j
    e^{−σ_j Δz} + b_j e^{+σ_j Δz})``) and multiply by ``−i/k`` from the
    linearized continuity relation ``u'_k = −(i/k) · ∂ŵ_k/∂z``.
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
    matrix1_u = np.zeros((z.size, x.size), dtype=complex)
    matrix3_u = np.zeros((z.size, x.size), dtype=complex)
    ht = 0.0

    for kloop in range(nslab):
        kk = mink + dk * kloop
        ksign = abs(kk)
        hs = np.pi * a * ho * np.exp(-a * ksign)
        if kloop > 0:
            ht += np.pi * dk * a * np.exp(-a * ksign)

        if kk == 0.0:
            # DC mode has no wave contribution; u' also vanishes here.
            matrix2 = np.zeros_like(matrix1)
            matrix2_u = np.zeros_like(matrix1)
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

            # Build ŵ and ∂ŵ/∂z on the vertical grid. The per-layer basis
            #   ŵ_j(z) = a_j e^{−σ_j Δz} + b_j e^{+σ_j Δz}
            # differentiates cleanly to
            #   ∂ŵ_j/∂z = σ_j · (−a_j e^{−σ_j Δz} + b_j e^{+σ_j Δz})
            # and the continuity relation gives u'_k = −(i/k) · ∂ŵ/∂z.
            zfac = np.zeros(z.size, dtype=complex)
            zfac_u = np.zeros(z.size, dtype=complex)
            inv_ik = -1j / kk
            for j in range(z.size):
                lj = layer_of[j]
                dz_l = z[j] - layer_bot[lj]
                e_minus = np.exp(-sigma[lj] * dz_l)
                e_plus = np.exp(sigma[lj] * dz_l)
                zfac[j] = aj[lj] * e_minus + bj[lj] * e_plus
                dwdz = sigma[lj] * (-aj[lj] * e_minus + bj[lj] * e_plus)
                zfac_u[j] = inv_ik * dwdz

            xfac = np.exp(-1j * x * kk)
            matrix2 = np.outer(zfac, xfac)
            matrix2_u = np.outer(zfac_u, xfac)

        if kloop > 0:
            matrix3 += 0.5 * (matrix1 + matrix2) * dk
            matrix3_u += 0.5 * (matrix1_u + matrix2_u) * dk
        matrix1 = matrix2
        matrix1_u = matrix2_u

    if ht == 0.0:
        ht = 1.0
    w = np.real(matrix3 / ht)
    u_prime = np.real(matrix3_u / ht)
    return x, z, w, u_prime


# ---------------------------------------------------------------------------
# Streamline tracer (port of stream.m)
# ---------------------------------------------------------------------------


def streamlines(x, z, u, w, num: int = 10):
    """Return ``num`` streamlines as ``[(xs, ys), ...]`` polylines.

    ``u`` may be a scalar (uniform mean flow, used for the two-layer solver)
    or a 1-D array of length ``nz`` giving the mean wind at each render-grid
    height. In linear wave theory the parcel displacement at height ``z₀`` is
    ``η(x, z₀) = (1/U(z₀)) · ∫ w(x', z₀) dx'``, so the time step used to
    integrate along each streamline depends on the wind at that streamline's
    height — not on the surface wind. Using a single scalar ``U_surface`` for
    every streamline (as Hart's MATLAB ``stream.m`` did because the two-layer
    case assumed uniform ``U``) over-amplifies upper streamlines whenever the
    real profile has shear.

    We guard against near-zero ``U(z₀)`` (which would blow up the tracer) with
    a 0.1 m/s floor — a parcel literally at rest cannot trace a linear
    streamline in this framework, so we just freeze it there.
    """
    x = np.asarray(x)
    z = np.asarray(z)
    w = np.asarray(w)
    nx = x.size
    nz = z.size
    if nx < 2 or nz < 2 or num == 0:
        return []
    minx = float(x[0])
    dx = float(x[1] - x[0])

    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    if u_arr.size == 1:
        u_by_row = np.full(nz, float(u_arr[0]))
    elif u_arr.size == nz:
        u_by_row = u_arr
    else:
        # Caller gave an array of the wrong length — fall back to the mean so
        # the plot still renders rather than raising mid-draw.
        u_by_row = np.full(nz, float(np.mean(u_arr)))

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
        u_local = float(u_by_row[yci])
        # 0.1 m/s floor prevents 1/u blowups at stagnant layers.
        u_local = u_local if abs(u_local) > 0.1 else math.copysign(0.1, u_local) if u_local != 0 else 0.1
        tstep = dx / u_local
        xs = np.empty(nx)
        ys = np.empty(nx)
        xs[0] = minx
        ys[0] = z[yci]
        for i in range(1, nx):
            xs[i] = x[i]
            ys[i] = ys[i - 1] + tstep * w[yci, i]
        lines.append((xs, ys))
    return lines
