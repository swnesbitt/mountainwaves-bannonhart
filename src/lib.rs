//! # mountain_waves core
//!
//! Rust port of Dr. Robert E. (Bob) Hart's 1995 MATLAB mountain-wave model
//! (`tlwplot.m`, `tlwmenu.m`, `stream.m`). Hart developed the original tool
//! as a Penn State Meteo 574 seminar project under Dr. Peter Bannon; the
//! numerical scheme, user-interface layout, and example cases all come from
//! his work. Documentation and MATLAB sources:
//! <https://moe.met.fsu.edu/~rhart/mtnwave.html>. Contact: `rhart@fsu.edu`.
//!
//! Rust compute core for the 2-D mountain-wave visualization tool. This crate
//! is compiled as a CPython extension module (`mountain_waves._core`) via
//! PyO3 and exposes three callables to Python:
//!
//! * [`compute_two_layer`] — exact analytic two-layer Scorer-parameter
//!   solution, a direct port of `tlwplot.m`.
//! * [`compute_from_profile`] — multi-layer Taylor–Goldstein solver that
//!   accepts arbitrary `u(z)` and `theta(z)` profiles.
//! * [`streamlines`] — streamline-tracing helper that integrates a constant
//!   `U` + perturbation `w(x, z)` field, returning a list of polylines.
//!
//! Arrays cross the Python boundary as NumPy arrays (via the `numpy` crate)
//! to avoid per-point allocation. Wavenumber integration is parallelized
//! over wavenumber samples with Rayon.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const I: Complex64 = Complex64::new(0.0, 1.0);

fn c(x: f64) -> Complex64 {
    Complex64::new(x, 0.0)
}

// ---------------------------------------------------------------------------
// Two-layer analytic solver (port of tlwplot.m)
// ---------------------------------------------------------------------------

/// Compute the vertical-velocity field `w(x, z)` for flow over a
/// witch-of-Agnesi mountain in a two-layer atmosphere.
///
/// Arrays returned:
///
/// * `x` — shape `(npts + 1,)`, meters
/// * `z` — shape `(npts + 1,)`, meters
/// * `w` — shape `(nz, nx)` with rows along z, columns along x (m s⁻¹)
#[pyfunction]
#[pyo3(signature = (l_upper, l_lower, u, h, a, ho, xdom, zdom, mink, maxk, npts=100))]
#[allow(clippy::too_many_arguments)]
fn compute_two_layer<'py>(
    py: Python<'py>,
    l_upper: f64,
    l_lower: f64,
    u: f64,
    h: f64,
    a: f64,
    ho: f64,
    xdom: f64,
    zdom: f64,
    mink: f64,
    maxk: f64,
    npts: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (x, z, w) = py.allow_threads(|| {
        two_layer(l_upper, l_lower, u, h, a, ho, xdom, zdom, mink, maxk, npts)
    });
    Ok((
        x.into_pyarray_bound(py),
        z.into_pyarray_bound(py),
        w.into_pyarray_bound(py),
    ))
}

fn two_layer(
    l_upper: f64,
    l_lower: f64,
    u: f64,
    h: f64,
    a: f64,
    ho: f64,
    xdom: f64,
    zdom: f64,
    mink: f64,
    maxk: f64,
    npts: usize,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let dk = 0.367 / a;
    let nk = (((maxk - mink) / dk).floor() as usize).max(1);
    let nslab = nk + 1;

    let minx = -0.25 * xdom;
    let maxx = 0.75 * xdom;
    let nx = npts + 1;
    let nz = npts + 1;
    let dx = (maxx - minx) / npts as f64;
    let dz = zdom / npts as f64;

    let x = Array1::from_iter((0..nx).map(|i| minx + dx * i as f64));
    let z = Array1::from_iter((0..nz).map(|j| dz * j as f64));

    // Parallel across wavenumber samples. Each worker returns a contribution
    // to the running trapezoidal sum stored as a flattened (nz * nx) complex
    // vector, plus a partial ht weight. We accumulate the trapezoid pairwise
    // afterward because trapezoidal summation needs adjacent samples.
    let slabs: Vec<Vec<Complex64>> = (0..nslab)
        .into_par_iter()
        .map(|idx| {
            let kk = mink + dk * idx as f64;
            let ksign = kk.abs();

            let m = c(l_lower * l_lower - kk * kk).sqrt();
            let n = c(kk * kk - l_upper * l_upper).sqrt();

            let denom = m + I * n;
            let r = if denom.norm() < 1e-300 {
                c(9e99)
            } else {
                (m - I * n) / denom
            };
            let big_r = r * (c(2.0) * I * m * h).exp();

            let a_coef = (c(1.0) + r) * (c(h) * n + I * c(h) * m).exp() / (c(1.0) + big_r);
            let c_coef = c(1.0) / (c(1.0) + big_r);
            let d_coef = big_r * c_coef;

            let hs = std::f64::consts::PI * a * ho * (-a * ksign).exp();

            let mut slab = vec![Complex64::new(0.0, 0.0); nz * nx];
            let prefactor = -I * c(kk) * c(hs * u);

            for (j, &zj) in z.iter().enumerate() {
                let zfac = if zj <= h {
                    c_coef * (I * c(zj) * m).exp() + d_coef * (-I * c(zj) * m).exp()
                } else {
                    a_coef * (-c(zj) * n).exp()
                };
                let row_scale = prefactor * zfac;
                let row_start = j * nx;
                for (i, &xi) in x.iter().enumerate() {
                    slab[row_start + i] = row_scale * (-I * c(xi * kk)).exp();
                }
            }
            slab
        })
        .collect();

    let ht_samples: Vec<f64> = (0..nslab)
        .map(|idx| {
            let kk = mink + dk * idx as f64;
            std::f64::consts::PI * dk * a * (-a * kk.abs()).exp()
        })
        .collect();

    // Trapezoidal integration along the wavenumber axis.
    let mut wc = vec![Complex64::new(0.0, 0.0); nz * nx];
    for kloop in 1..nslab {
        for i in 0..(nz * nx) {
            wc[i] += c(0.5 * dk) * (slabs[kloop - 1][i] + slabs[kloop][i]);
        }
    }
    let ht: f64 = ht_samples[1..].iter().sum();
    let inv_ht = if ht.abs() > 0.0 { 1.0 / ht } else { 1.0 };

    let mut w = Array2::<f64>::zeros((nz, nx));
    for j in 0..nz {
        for i in 0..nx {
            w[[j, i]] = wc[j * nx + i].re * inv_ht;
        }
    }

    (x, z, w)
}

// ---------------------------------------------------------------------------
// Multi-layer profile solver
// ---------------------------------------------------------------------------

/// Multi-layer Taylor–Goldstein solver accepting arbitrary `u(z)` / `theta(z)`.
///
/// The solver uses the basis
///
///     ŵ_j(z) = a_j exp(-σ_j (z - z_bot_j)) + b_j exp(+σ_j (z - z_bot_j))
///
/// inside layer `j`, with `σ_j = sqrt(k² - L_j²)` taken on the principal
/// branch (non-negative imaginary part). In this basis the `a` coefficient
/// is always the outgoing branch — it decays upward when the layer is
/// evanescent and has downward phase velocity (= upward group velocity) when
/// the layer is propagating. The radiation / decay condition at the top of
/// the domain is therefore simply `b_top = 0`.
#[pyfunction]
#[pyo3(signature = (z_profile, u_profile, theta_profile, a, ho, xdom, zdom, mink, maxk, npts=100))]
#[allow(clippy::too_many_arguments)]
fn compute_from_profile<'py>(
    py: Python<'py>,
    z_profile: PyReadonlyArray1<'py, f64>,
    u_profile: PyReadonlyArray1<'py, f64>,
    theta_profile: PyReadonlyArray1<'py, f64>,
    a: f64,
    ho: f64,
    xdom: f64,
    zdom: f64,
    mink: f64,
    maxk: f64,
    npts: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let zp = z_profile.as_array().to_owned();
    let up = u_profile.as_array().to_owned();
    let tp = theta_profile.as_array().to_owned();
    let (x, z, w) = py.allow_threads(|| {
        multi_layer(&zp, &up, &tp, a, ho, xdom, zdom, mink, maxk, npts)
    });
    Ok((
        x.into_pyarray_bound(py),
        z.into_pyarray_bound(py),
        w.into_pyarray_bound(py),
    ))
}

fn scorer_from_profile(z: &Array1<f64>, u: &Array1<f64>, theta: &Array1<f64>) -> Array1<f64> {
    const G: f64 = 9.80665;
    let n = z.len();
    let mut l2 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (dthdz, d2udz2) = if i == 0 {
            let dth = (theta[1] - theta[0]) / (z[1] - z[0]);
            let d2u = if n >= 3 {
                let h1 = z[1] - z[0];
                let h2 = z[2] - z[1];
                2.0 * (u[2] * h1 - u[1] * (h1 + h2) + u[0] * h2) / (h1 * h2 * (h1 + h2))
            } else {
                0.0
            };
            (dth, d2u)
        } else if i == n - 1 {
            let dth = (theta[n - 1] - theta[n - 2]) / (z[n - 1] - z[n - 2]);
            let d2u = if n >= 3 {
                let h1 = z[n - 2] - z[n - 3];
                let h2 = z[n - 1] - z[n - 2];
                2.0 * (u[n - 1] * h1 - u[n - 2] * (h1 + h2) + u[n - 3] * h2) / (h1 * h2 * (h1 + h2))
            } else {
                0.0
            };
            (dth, d2u)
        } else {
            let h1 = z[i] - z[i - 1];
            let h2 = z[i + 1] - z[i];
            let dth = (theta[i + 1] * h1 * h1 - theta[i - 1] * h2 * h2
                + theta[i] * (h2 * h2 - h1 * h1))
                / (h1 * h2 * (h1 + h2));
            let d2u = 2.0 * (u[i + 1] * h1 - u[i] * (h1 + h2) + u[i - 1] * h2) / (h1 * h2 * (h1 + h2));
            (dth, d2u)
        };
        let n2 = (G / theta[i]) * dthdz;
        let uu = u[i];
        l2[i] = if uu.abs() > 1e-6 {
            n2 / (uu * uu) - d2udz2 / uu
        } else {
            0.0
        };
    }
    l2
}

/// Principal branch of sqrt chosen so that Im(result) ≥ 0 (or Re ≥ 0 when
/// argument is a non-negative real). This corresponds to the outgoing /
/// decaying branch of exp(-σz) for mountain-wave radiation conditions.
fn principal_sigma(k2_minus_l2: f64) -> Complex64 {
    let s = c(k2_minus_l2).sqrt();
    if s.im < 0.0 {
        -s
    } else {
        s
    }
}

fn multi_layer(
    z_profile: &Array1<f64>,
    u_profile: &Array1<f64>,
    theta_profile: &Array1<f64>,
    a: f64,
    ho: f64,
    xdom: f64,
    zdom: f64,
    mink: f64,
    maxk: f64,
    npts: usize,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let l2_profile = scorer_from_profile(z_profile, u_profile, theta_profile);
    let u_surface = u_profile[0];

    let nlayers = z_profile.len();
    let mut layer_bot = Array1::<f64>::zeros(nlayers);
    let mut layer_top = Array1::<f64>::zeros(nlayers);
    for j in 0..nlayers {
        layer_bot[j] = if j == 0 {
            0.0
        } else {
            0.5 * (z_profile[j - 1] + z_profile[j])
        };
        layer_top[j] = if j == nlayers - 1 {
            f64::INFINITY
        } else {
            0.5 * (z_profile[j] + z_profile[j + 1])
        };
    }

    let dk = 0.367 / a;
    let nk = (((maxk - mink) / dk).floor() as usize).max(1);
    let nslab = nk + 1;

    let minx = -0.25 * xdom;
    let maxx = 0.75 * xdom;
    let nx = npts + 1;
    let nz = npts + 1;
    let dx = (maxx - minx) / npts as f64;
    let dz_grid = zdom / npts as f64;

    let x = Array1::from_iter((0..nx).map(|i| minx + dx * i as f64));
    let z = Array1::from_iter((0..nz).map(|j| dz_grid * j as f64));

    let layer_of: Vec<usize> = z
        .iter()
        .map(|&zj| {
            let mut idx = nlayers - 1;
            for l in 0..nlayers {
                if zj < layer_top[l] {
                    idx = l;
                    break;
                }
            }
            idx
        })
        .collect();

    let slabs: Vec<Vec<Complex64>> = (0..nslab)
        .into_par_iter()
        .map(|idx| {
            let kk = mink + dk * idx as f64;
            let mut slab = vec![Complex64::new(0.0, 0.0); nz * nx];
            if kk == 0.0 {
                return slab;
            }
            let ksign = kk.abs();

            let sigma: Vec<Complex64> = (0..nlayers)
                .map(|j| principal_sigma(kk * kk - l2_profile[j]))
                .collect();

            // Top-down sweep: a_top = 1, b_top = 0.
            let mut aj = vec![Complex64::new(0.0, 0.0); nlayers];
            let mut bj = vec![Complex64::new(0.0, 0.0); nlayers];
            aj[nlayers - 1] = c(1.0);
            bj[nlayers - 1] = c(0.0);

            for j in (0..nlayers - 1).rev() {
                let dz_j = layer_top[j] - layer_bot[j];
                let e_minus = (-sigma[j] * dz_j).exp();
                let e_plus = (sigma[j] * dz_j).exp();
                let alpha = aj[j + 1] + bj[j + 1];
                let beta = -aj[j + 1] + bj[j + 1];
                let ratio = sigma[j + 1] / sigma[j];
                aj[j] = 0.5 * (alpha - ratio * beta) * e_plus;
                bj[j] = 0.5 * (alpha + ratio * beta) * e_minus;
            }

            let hs = std::f64::consts::PI * a * ho * (-a * ksign).exp();
            let w_surface_unnorm = aj[0] + bj[0];
            let amp = if w_surface_unnorm.norm() < 1e-300 {
                Complex64::new(0.0, 0.0)
            } else {
                -I * c(kk * u_surface * hs) / w_surface_unnorm
            };
            for j in 0..nlayers {
                aj[j] *= amp;
                bj[j] *= amp;
            }

            // Assemble slab.
            let xfac: Vec<Complex64> =
                x.iter().map(|&xi| (-I * c(xi * kk)).exp()).collect();

            for j in 0..nz {
                let l = layer_of[j];
                let dz_l = z[j] - layer_bot[l];
                let zval = aj[l] * (-sigma[l] * dz_l).exp()
                    + bj[l] * (sigma[l] * dz_l).exp();
                let row_start = j * nx;
                for i in 0..nx {
                    slab[row_start + i] = zval * xfac[i];
                }
            }
            slab
        })
        .collect();

    let ht_samples: Vec<f64> = (0..nslab)
        .map(|idx| {
            let kk = mink + dk * idx as f64;
            std::f64::consts::PI * dk * a * (-a * kk.abs()).exp()
        })
        .collect();
    let ht: f64 = ht_samples[1..].iter().sum();
    let inv_ht = if ht.abs() > 0.0 { 1.0 / ht } else { 1.0 };

    let mut wc = vec![Complex64::new(0.0, 0.0); nz * nx];
    for kloop in 1..nslab {
        for i in 0..(nz * nx) {
            wc[i] += c(0.5 * dk) * (slabs[kloop - 1][i] + slabs[kloop][i]);
        }
    }

    let mut w = Array2::<f64>::zeros((nz, nx));
    for j in 0..nz {
        for i in 0..nx {
            w[[j, i]] = wc[j * nx + i].re * inv_ht;
        }
    }

    (x, z, w)
}

// ---------------------------------------------------------------------------
// Streamline tracer
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (x, z, u, w, num=10))]
fn streamlines<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    z: PyReadonlyArray1<'py, f64>,
    u: f64,
    w: PyReadonlyArray2<'py, f64>,
    num: usize,
) -> PyResult<Vec<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>> {
    let xa = x.as_array().to_owned();
    let za = z.as_array().to_owned();
    let wa = w.as_array().to_owned();
    let lines = py.allow_threads(|| trace_streamlines(&xa, &za, u, &wa, num));
    Ok(lines
        .into_iter()
        .map(|(xs, ys)| (xs.into_pyarray_bound(py), ys.into_pyarray_bound(py)))
        .collect())
}

fn trace_streamlines(
    x: &Array1<f64>,
    z: &Array1<f64>,
    u: f64,
    w: &Array2<f64>,
    num: usize,
) -> Vec<(Array1<f64>, Array1<f64>)> {
    let nx = x.len();
    let nz = z.len();
    if nx < 2 || nz < 2 || num == 0 {
        return Vec::new();
    }
    let minx = x[0];
    let dx = x[1] - x[0];
    let tstep = dx / u;
    let dh = nz as f64 / num as f64;

    let mut lines = Vec::with_capacity(num);
    for j in 0..num {
        let mut ycell = 1.0 + dh * j as f64;
        if ycell < 1.0 {
            ycell = 1.0;
        }
        if ycell > nz as f64 {
            ycell = nz as f64;
        }
        let yci = (ycell.round() as isize - 1).clamp(0, nz as isize - 1) as usize;
        let mut xs = Array1::<f64>::zeros(nx);
        let mut ys = Array1::<f64>::zeros(nx);
        xs[0] = minx;
        ys[0] = z[yci];
        for i in 1..nx {
            xs[i] = x[i];
            ys[i] = ys[i - 1] + tstep * w[[yci, i]];
        }
        lines.push((xs, ys));
    }
    lines
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_two_layer, m)?)?;
    m.add_function(wrap_pyfunction!(compute_from_profile, m)?)?;
    m.add_function(wrap_pyfunction!(streamlines, m)?)?;
    m.add("__doc__", "Mountain Waves Rust compute core.")?;
    Ok(())
}
