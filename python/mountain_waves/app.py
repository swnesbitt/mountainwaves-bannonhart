"""Interactive Dash front end for the Mountain Waves model.

This UI is the browser-based counterpart to the original MATLAB menu in
Dr. Robert E. (Bob) Hart's 1995 mountain-wave tool (``tlwmenu.m``). Hart's
slider layout, example cases, and readouts (Scorer condition, Rossby
number, dimensionless mountain height) guided the design here. Original
model and documentation: https://moe.met.fsu.edu/~rhart/mtnwave.html.

Two solver modes are exposed:

* **Two-layer** — sliders for the original MATLAB parameters
  (surface wind, upper/lower Scorer, interface height, mountain geometry,
  domain, spectrum).
* **Profile** — drag control points on an ``u(z)`` graph and a
  ``theta(z)`` graph to prescribe arbitrary profiles. The app converts
  these to a Scorer parameter profile and runs the multi-layer solver.

Both modes render streamline analysis and a vertical-velocity contour
plot side-by-side, along with diagnostic readouts (Scorer condition,
Rossby number, solver backend).
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

from . import solver
from .profile import (
    brunt_vaisala,
    default_profile_heights,
    default_theta_profile,
    default_u_profile,
    scorer_from_profile,
)


OMEGA = 7.292e-5
LATIT_RAD_DEFAULT = math.radians(45.0)


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

PRESETS = {
    "uniform": {
        "label": "Uniform atmosphere (Example 1)",
        "U": 20.0,
        "L_upper": 4.0,
        "L_lower": 4.0,
        "H": 3.5,
        "mtn_h": 0.5,
        "mtn_a": 2.5,
        "xdom": 40.0,
        "zdom": 10.0,
        "mink_k": 0,
        "maxk_k": 30,
    },
    "trapped": {
        "label": "Trapped lee waves (Example 2)",
        "U": 20.0,
        "L_upper": 4.0,
        "L_lower": 10.0,
        "H": 3.5,
        "mtn_h": 0.5,
        "mtn_a": 2.5,
        "xdom": 40.0,
        "zdom": 10.0,
        "mink_k": 0,
        "maxk_k": 30,
    },
    "downslope": {
        # Not from Hart's MATLAB (he only documented Examples 1 & 2). This is
        # a "strong lee wave / near-downslope" case kept inside linear theory.
        # Nh/U = L_lower * ho = 10e-4 * 800 = 0.80, just under the
        # wave-breaking threshold (~0.85). Scorer condition ≈ 3.0, so a
        # robust trapped-wave packet forms downstream — the closest the
        # linear Fourier/Scorer model can get to a real downslope windstorm
        # without blowing up, which Hart himself flagged in the "Problems"
        # section of his documentation.
        "label": "Strong lee waves / near-downslope (linear-theory edge)",
        "U": 25.0,
        "L_upper": 3.0,
        "L_lower": 10.0,
        "H": 2.5,
        "mtn_h": 0.8,
        "mtn_a": 2.5,
        "xdom": 40.0,
        "zdom": 10.0,
        "mink_k": 0,
        "maxk_k": 30,
    },
}


def _rossby(u: float, a: float, latit_deg: float) -> float:
    """Return U / (f a) with f evaluated at the given latitude (degrees N)."""
    lat = math.radians(max(0.0, min(90.0, latit_deg)))
    f = 2.0 * OMEGA * math.sin(lat)
    if f == 0.0 or a == 0.0:
        return float("inf")
    return u / (f * a)


def _two_layer_plots(params, latit_deg: float = 45.0) -> Tuple[go.Figure, go.Figure, go.Figure, float, float]:
    """Run the two-layer solver and build the streamline, w, and u' figures."""
    U = params["U"]
    L_upper = params["L_upper"] * 1e-4
    L_lower = params["L_lower"] * 1e-4
    H = params["H"] * 1000.0
    a = params["mtn_a"] * 1000.0
    ho = params["mtn_h"] * 1000.0
    xdom = params["xdom"] * 1000.0
    zdom = params["zdom"] * 1000.0
    mink = params["mink_k"] / a
    maxk = params["maxk_k"] / a

    x, z, w, u_prime = solver.compute_two_layer(
        L_upper, L_lower, U, H, a, ho, xdom, zdom, mink, maxk, npts=100
    )
    lines = solver.streamlines(x, z, U, w, num=10)

    scorer_cond = 4.0 * H * H * (L_lower ** 2 - L_upper ** 2) / (math.pi ** 2)
    rossby = _rossby(U, a, latit_deg)

    return (*_make_plots(x, z, w, u_prime, lines, H), scorer_cond, rossby)


def _profile_plots(params, z_prof, u_prof, theta_prof, latit_deg: float = 45.0) -> Tuple[go.Figure, go.Figure, go.Figure, float, float]:
    """Run the multi-layer solver from prescribed profiles."""
    U_surface = float(u_prof[0])
    a = params["mtn_a"] * 1000.0
    ho = params["mtn_h"] * 1000.0
    xdom = params["xdom"] * 1000.0
    zdom = params["zdom"] * 1000.0
    mink = params["mink_k"] / a
    maxk = params["maxk_k"] / a

    x, z, w, u_prime = solver.compute_from_profile(
        z_prof, u_prof, theta_prof, a, ho, xdom, zdom, mink, maxk, npts=100
    )
    # Pass the full u(z) profile (interpolated onto the render grid) so the
    # streamline tracer uses the local mean wind at each streamline's height.
    # Using a single U_surface over-amplifies upper streamlines whenever the
    # profile has shear, because linear theory gives η(x, z₀) = (1/U(z₀)) ·
    # ∫ w dx'.
    u_at_z = np.interp(np.asarray(z, dtype=float), np.asarray(z_prof, dtype=float),
                       np.asarray(u_prof, dtype=float))
    lines = solver.streamlines(x, z, u_at_z, w, num=10)

    l2 = scorer_from_profile(z_prof, u_prof, theta_prof)
    # Effective 2-layer diagnostic: split profile at midpoint of z_prof range.
    mid = z_prof[len(z_prof) // 2]
    below = l2[z_prof <= mid]
    above = l2[z_prof > mid]
    l_lower2 = float(np.mean(np.maximum(below, 0.0))) if below.size else 0.0
    l_upper2 = float(np.mean(np.maximum(above, 0.0))) if above.size else 0.0
    scorer_cond = 4.0 * mid ** 2 * (l_lower2 - l_upper2) / (math.pi ** 2)

    rossby = _rossby(U_surface, a, latit_deg)

    # Decorate the streamline plot with a dashed line showing the profile's
    # dominant interface: the height of maximum |dL^2/dz|.
    interface_z = None
    if l2.size > 2:
        dl = np.abs(np.diff(l2))
        interface_z = float(z_prof[1:][np.argmax(dl)])

    return (*_make_plots(x, z, w, u_prime, lines, interface_z), scorer_cond, rossby)


def _make_plots(x, z, w, u_prime, lines, interface_z):
    """Build streamline, vertical-velocity, and u' figures from solver output."""
    x_km = np.asarray(x) / 1000.0
    z_km = np.asarray(z) / 1000.0

    # Streamline figure — the first streamline traces the mountain surface;
    # fill it to draw the mountain.
    stream_fig = go.Figure()
    mountain_color = "rgb(50, 196, 50)"
    if lines:
        xs, ys = lines[0]
        xs_km = np.asarray(xs) / 1000.0
        ys_km = np.asarray(ys) / 1000.0
        stream_fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs_km, [x_km[-1], x_km[0], x_km[0]]]),
                y=np.concatenate([ys_km, [0.0, 0.0, ys_km[0]]]),
                fill="toself",
                mode="lines",
                line=dict(width=1, color=mountain_color),
                fillcolor=mountain_color,
                name="Mountain",
                hoverinfo="skip",
            )
        )
        for xs, ys in lines[1:]:
            stream_fig.add_trace(
                go.Scatter(
                    x=np.asarray(xs) / 1000.0,
                    y=np.asarray(ys) / 1000.0,
                    mode="lines",
                    line=dict(width=1.2, color="white"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if interface_z is not None:
        stream_fig.add_hline(
            y=interface_z / 1000.0,
            line=dict(color="magenta", dash="dash", width=1.5),
            annotation_text="interface",
            annotation_position="top right",
            annotation_font_color="magenta",
        )

    stream_fig.update_layout(
        title="Streamline Analysis",
        xaxis_title="x (km)",
        yaxis_title="height (km)",
        template="plotly_dark",
        yaxis=dict(range=[0, z_km[-1]]),
        xaxis=dict(range=[x_km[0], x_km[-1]]),
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )

    # Vertical-velocity contour. Clip at ±10 m/s like the MATLAB caxis.
    w_clip = np.clip(w, -10.0, 10.0)
    w_fig = go.Figure(
        data=go.Heatmap(
            x=x_km,
            y=z_km,
            z=w_clip,
            zmin=-10,
            zmax=10,
            colorscale="RdBu_r",
            reversescale=False,
            colorbar=dict(title="w (m/s)"),
            zsmooth="best",
        )
    )
    if interface_z is not None:
        w_fig.add_hline(
            y=interface_z / 1000.0,
            line=dict(color="magenta", dash="dash", width=1.5),
        )
    w_fig.update_layout(
        title="Vertical Velocity w (m/s)",
        xaxis_title="x (km)",
        yaxis_title="height (km)",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )

    # Horizontal-wind perturbation u'. Scale symmetrically around zero using
    # the 98th-percentile magnitude so outlier spikes near the mountain don't
    # wash out the lee-wave pattern. Use a diverging colormap (PuOr) distinct
    # from w's RdBu_r so the two fields stay visually separable at a glance.
    up = np.asarray(u_prime)
    up_absmax = float(np.nanpercentile(np.abs(up), 98.0))
    if not math.isfinite(up_absmax) or up_absmax <= 0.0:
        up_absmax = 1.0
    up_clip = np.clip(up, -up_absmax, up_absmax)
    uprime_fig = go.Figure(
        data=go.Heatmap(
            x=x_km,
            y=z_km,
            z=up_clip,
            zmin=-up_absmax,
            zmax=up_absmax,
            colorscale="PuOr",
            reversescale=False,
            colorbar=dict(title="u' (m/s)"),
            zsmooth="best",
        )
    )
    if interface_z is not None:
        uprime_fig.add_hline(
            y=interface_z / 1000.0,
            line=dict(color="magenta", dash="dash", width=1.5),
        )
    uprime_fig.update_layout(
        title="Horizontal-wind perturbation u' (m/s)",
        xaxis_title="x (km)",
        yaxis_title="height (km)",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )

    return stream_fig, w_fig, uprime_fig


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _slider(id_, label, mn, mx, step, value, unit=""):
    return html.Div(
        className="slider-row",
        children=[
            html.Div(
                [html.Span(label, className="slider-label"), html.Span(f"{value}{unit}", id=f"{id_}-val", className="slider-value")],
                className="slider-header",
            ),
            dcc.Slider(
                id=id_,
                min=mn,
                max=mx,
                step=step,
                value=value,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
    )


PROFILE_ZDOM_KM = 10.0
PROFILE_N_POINTS = 32
PROFILE_FIG_HEIGHT_PX = 680   # taller graph so 32 layers don't crowd
# Ranges used both for axis limits and drag clamping. Kept as module-level
# constants so the figure builder, the drag handler, and the redraw all agree.
# The U axis allows negative values so HRRR profiles whose mean flow points
# against the user-chosen "flow from" direction read as negative rather than
# getting silently clamped to 0. Theta floor is dropped to 250 K so cold
# continental winter columns stay visible.
U_RANGE = (-40.0, 80.0)
THETA_RANGE = (250.0, 400.0)
UNDO_HISTORY_MAX = 50


def _init_profile_figures():
    # Defaults tuned to reproduce MATLAB "Example 2" (trapped lee waves):
    # sharp low-level stability, weak stability aloft, light surface shear.
    # 32 evenly-spaced levels give finer resolution in both the boundary
    # layer and the free troposphere while keeping the column at 10 km.
    # At n=32 the transfer-matrix solver converges to within ~1% of the
    # two-layer analytic result for the trapped-wave case.
    zs = default_profile_heights(PROFILE_ZDOM_KM, PROFILE_N_POINTS)
    us = default_u_profile(zs, u_surface=20.0, shear=0.5)
    thetas = default_theta_profile(zs, interface_km=3.5, lapse_lower=11.8, lapse_upper=1.9)
    return zs, us, thetas


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Dash:
    # suppress_callback_exceptions is required because `controls-block` is
    # swapped between two-layer and profile panels at runtime — the IDs
    # referenced in the Analyze callback don't all exist in the initial layout.
    # Ship static assets (CliMAS logo, favicon) from the package itself
    # so the app works regardless of where it's launched from.
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _assets_dir = os.path.join(_pkg_dir, "assets")

    app = Dash(
        __name__,
        title="Mountain Waves",
        update_title=None,
        suppress_callback_exceptions=True,
        assets_folder=_assets_dir,
    )

    zs0, us0, thetas0 = _init_profile_figures()

    app.layout = html.Div(
        className="mw-root",
        children=[
            html.Header(
                className="mw-header",
                children=[
                    html.Img(
                        src=app.get_asset_url("climas_icon_64.png"),
                        alt="Climate, Meteorology & Atmospheric Sciences",
                        className="mw-logo",
                    ),
                    html.Div(
                        className="mw-header-text",
                        children=[
                            html.H1("Interactive 2-D Mountain-Wave Visualizer"),
                            html.Div(
                                className="mw-subtitle",
                                children=[
                                    html.Span("Rust + Python port of "),
                                    html.A("Bob Hart's original MATLAB model", href="https://moe.met.fsu.edu/~rhart/mtnwave.html", target="_blank"),
                                    html.Span(f" · compute backend: {solver.backend_name()}"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="mode-tabs",
                value="two-layer",
                children=[
                    dcc.Tab(label="Two-layer (analytic)", value="two-layer"),
                    dcc.Tab(label="Profile (multi-layer)", value="profile"),
                ],
            ),
            html.Div(
                className="mw-global",
                children=[
                    html.Div("Latitude (°N)", className="mw-global-label"),
                    dcc.Slider(
                        id="latit",
                        min=0,
                        max=90,
                        step=1,
                        value=45,
                        marks={0: "0°", 23: "23°", 45: "45°", 66: "66°", 90: "90°"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Span(id="latit-val", className="mw-global-value"),
                    html.Span(
                        "Coriolis f = 2Ω sin(φ) — affects the Rossby-number readout only.",
                        className="mw-global-note",
                    ),
                ],
            ),
            # Both control panels live in the DOM simultaneously so every
            # callback Input/State ID always resolves; the inactive one is
            # hidden via CSS (see the mode-tabs callback below).
            html.Div(
                id="controls-block",
                children=[
                    html.Div(id="controls-two-layer", children=_two_layer_controls()),
                    html.Div(
                        id="controls-profile",
                        children=_profile_controls(zs0, us0, thetas0),
                        style={"display": "none"},
                    ),
                ],
            ),
            html.Div(
                className="mw-diagnostics",
                children=[
                    html.Div(id="scorer-readout", className="diag-card"),
                    html.Div(id="rossby-readout", className="diag-card"),
                    html.Div(id="nonlin-readout", className="diag-card"),
                ],
            ),
            # Top row: w and u' heatmaps side-by-side. Streamline view lives
            # below at full width so the trajectory pattern is legible at
            # wide aspect ratios (it's the "money plot" of the tool).
            html.Div(
                className="mw-heatmaps",
                children=[
                    dcc.Graph(id="w-plot", config={"displayModeBar": False}),
                    dcc.Graph(id="uprime-plot", config={"displayModeBar": False}),
                ],
            ),
            html.Div(
                className="mw-streamline-full",
                children=[
                    dcc.Graph(id="streamline-plot", config={"displayModeBar": False}),
                ],
            ),
            # Shared state for editable profiles.
            dcc.Store(
                id="profile-store",
                data={
                    "z": zs0.tolist(),
                    "u": us0.tolist(),
                    "theta": thetas0.tolist(),
                },
            ),
            # Undo history — a stack of prior profile-store states. Each edit
            # pushes the pre-edit state; undo pops the most recent entry.
            dcc.Store(id="profile-history", data=[]),
            # Raw HRRR column cached after a successful fetch: the east/north
            # wind components (u, v), potential temperature, target heights,
            # and meta info. Lets the "flow from" slider re-project the wind
            # onto a new axis without another AWS round-trip.
            dcc.Store(id="hrrr-raw-store", data={}),
            html.Footer(
                className="mw-footer",
                children=[
                    html.Span("On the Profile tab, drag the gold circles left/right to edit u(z) and θ(z). "),
                    html.Span("Click 'Analyze flow' to update the fields."),
                ],
            ),
        ],
    )

    app.index_string = """
<!doctype html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
body { background: #0b0e14; color: #dfe3ea; font-family: -apple-system, system-ui, sans-serif; }
.mw-root { max-width: 1400px; margin: 0 auto; padding: 20px; }
.mw-header { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
.mw-header-text { display: flex; flex-direction: column; }
.mw-logo { width: 56px; height: 56px; border-radius: 6px; flex-shrink: 0; }
.mw-header h1 { margin: 0 0 4px 0; font-size: 24px; }
.mw-subtitle { color: #9aa3ad; font-size: 13px; }
.mw-subtitle a { color: #57b3ff; }
.mw-controls { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px 28px; background: #11161f; padding: 18px; border-radius: 8px; margin: 14px 0; }
.mw-controls.profile { grid-template-columns: 1fr 1fr 1fr; }
.slider-row { display: flex; flex-direction: column; gap: 4px; }
.slider-header { display: flex; justify-content: space-between; font-size: 13px; }
.slider-label { color: #c7ced6; }
.slider-value { color: #6ecbff; font-variant-numeric: tabular-nums; }
.mw-presets { display: flex; gap: 8px; margin: 10px 0 0 0; }
.mw-presets button, .mw-analyze { background: #1e2835; border: 1px solid #2d3a4b; color: #dfe3ea; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.mw-presets button:hover, .mw-analyze:hover { background: #2a3a4e; }
.mw-analyze { background: #2469c6; border-color: #3b86e6; font-weight: 600; padding: 8px 18px; }
.mw-analyze:hover { background: #3079d8; }
.mw-diagnostics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 10px 0; }
.diag-card { background: #11161f; padding: 10px 16px; border-radius: 6px; font-size: 14px; }
.diag-card .v { color: #ffd685; font-weight: 700; margin-left: 8px; font-variant-numeric: tabular-nums; }
.diag-card .note { color: #9aa3ad; font-size: 12px; margin-left: 10px; }
.mw-plots { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.mw-heatmaps { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.mw-streamline-full { margin-top: 12px; }
.mw-streamline-full > .dash-graph { width: 100%; }
.mw-profile-editors { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 6px; }
.mw-profile-toolbar { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.mw-undo { background: #1e2835; border: 1px solid #2d3a4b; color: #dfe3ea; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.mw-undo:hover:not(:disabled) { background: #2a3a4e; }
.mw-undo:disabled { opacity: 0.4; cursor: default; }
.mw-undo-status { color: #8d97a2; font-size: 12px; }
/* HRRR card takes the first 2 of 3 columns of .mw-controls.profile (so
   its one-line input row has room), and the Actions card sits in the
   remaining column to its right. */
.mw-hrrr-section { grid-column: span 2; }
.mw-actions-section { grid-column: span 1; display: flex; flex-direction: column; gap: 10px; }
.mw-action-btn { width: 100%; padding: 10px 14px; font-size: 14px; text-align: center; background: #1e2835; border: 1px solid #2d3a4b; color: #dfe3ea; border-radius: 6px; cursor: pointer; }
.mw-action-btn:hover { background: #2a3a4e; }
.mw-action-btn.mw-analyze { background: #2469c6; border-color: #3b86e6; font-weight: 600; }
.mw-action-btn.mw-analyze:hover { background: #3079d8; }
.mw-hrrr-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
.mw-hrrr-row-compact { gap: 8px; row-gap: 4px; flex-wrap: nowrap; width: 100%; }
.mw-hrrr-lbl { font-size: 11px; color: #9aa3ad; min-width: 0; white-space: nowrap; flex-shrink: 0; }
.mw-hrrr-input { background: #0f1520; border: 1px solid #2d3a4b; color: #dfe3ea; padding: 2px 6px; border-radius: 4px; font-size: 11px; width: 70px; font-family: inherit; flex-shrink: 0; }
.mw-hrrr-input.wide { flex: 1 1 auto; min-width: 110px; width: auto; }
.mw-hrrr-btn { font-size: 11px; padding: 3px 12px; margin-left: auto; flex-shrink: 0; }
.mw-hrrr-status { color: #9aa3ad; font-size: 11px; margin-top: 4px; min-height: 14px; }
.mw-hrrr-status.error { color: #ff8a8a; }
.mw-hrrr-status.ok { color: #89d185; }
/* Flow-from slider row — label on the left, wide slider stretching to the
   edge, current value pinned on the right. */
.mw-hrrr-dir-row { display: flex; align-items: center; gap: 10px; margin-top: 10px; width: 100%; }
.mw-hrrr-dir-row > label { min-width: 60px; flex-shrink: 0; }
.mw-hrrr-dir-slider { flex: 1 1 auto; min-width: 0; padding: 0 8px; }
.mw-hrrr-dir-val { color: #6ecbff; font-variant-numeric: tabular-nums; font-weight: 600; min-width: 56px; text-align: right; flex-shrink: 0; font-size: 12px; }
.mw-profile-diagnostics { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
.mw-footer { margin-top: 20px; color: #8f98a3; font-size: 12px; }
.mw-global { display: grid; grid-template-columns: 130px 1fr 80px 1fr; align-items: center; gap: 14px; background: #131a24; padding: 10px 16px; border-radius: 6px; margin-top: 10px; }
.mw-global-label { font-size: 13px; color: #c7ced6; }
.mw-global-value { color: #6ecbff; font-variant-numeric: tabular-nums; font-weight: 600; }
.mw-global-note { font-size: 12px; color: #8f98a3; }
.section-title { font-size: 11px; font-weight: 600; color: #8d97a2; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.section { background: #161d29; padding: 12px; border-radius: 6px; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

    _register_callbacks(app)
    return app


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _two_layer_controls():
    return html.Div(
        className="mw-controls",
        children=[
            html.Div(
                className="section",
                children=[
                    html.Div("Atmospheric profile", className="section-title"),
                    _slider("U", "Surface wind (m/s)", 0, 100, 1, 20),
                    _slider("L_upper", "L upper (×10⁻⁴)", 0, 50, 0.1, 4.0),
                    _slider("L_lower", "L lower (×10⁻⁴)", 0, 50, 0.1, 10.0),
                    _slider("H", "Interface ht. (km)", 0, 20, 0.1, 3.5),
                ],
            ),
            html.Div(
                className="section",
                children=[
                    html.Div("Terrain profile", className="section-title"),
                    _slider("mtn_h", "Max height (km)", 0, 3, 0.05, 0.5),
                    _slider("mtn_a", "Half-width (km)", 0.25, 25, 0.25, 2.5),
                    html.Div("Domain profile", className="section-title", style={"marginTop": "12px"}),
                    _slider("xdom", "Horizontal (km)", 5, 100, 1, 40),
                    _slider("zdom", "Vertical (km)", 1, 20, 0.5, 10),
                    html.Div("Spectral profile", className="section-title", style={"marginTop": "12px"}),
                    _slider("mink_k", "Min wave# (half-widths)", 0, 50, 0.5, 0),
                    _slider("maxk_k", "Max wave# (half-widths)", 0, 50, 0.5, 30),
                    html.Div(
                        className="mw-presets",
                        children=[
                            html.Button("Uniform", id="preset-uniform"),
                            html.Button("Trapped waves", id="preset-trapped"),
                            html.Button(
                                "Near-downslope",
                                id="preset-downslope",
                                title=(
                                    "Strong lee waves approaching the downslope-windstorm "
                                    "regime. Not from Hart's examples — linear Fourier/Scorer "
                                    "theory cannot fully simulate real downslope windstorms "
                                    "because they are fundamentally nonlinear."
                                ),
                            ),
                            html.Button("Analyze flow", id="analyze-two", className="mw-analyze"),
                        ],
                    ),
                ],
            ),
        ],
    )


def _profile_controls(zs, us, thetas):
    return html.Div(
        children=[
            html.Div(
                className="mw-controls profile",
                children=[
                    html.Div(
                        className="section",
                        children=[
                            html.Div("Terrain", className="section-title"),
                            _slider("p_mtn_h", "Max height (km)", 0, 3, 0.05, 0.5),
                            _slider("p_mtn_a", "Half-width (km)", 0.25, 25, 0.25, 2.5),
                        ],
                    ),
                    html.Div(
                        className="section",
                        children=[
                            html.Div("Domain", className="section-title"),
                            _slider("p_xdom", "Horizontal (km)", 5, 100, 1, 40),
                            _slider("p_zdom", "Vertical (km)", 1, 20, 0.5, 10),
                        ],
                    ),
                    html.Div(
                        className="section",
                        children=[
                            html.Div("Spectrum", className="section-title"),
                            _slider("p_mink_k", "Min wave# (half-widths)", 0, 50, 0.5, 0),
                            _slider("p_maxk_k", "Max wave# (half-widths)", 0, 50, 0.5, 30),
                        ],
                    ),
                    html.Div(
                        className="section mw-hrrr-section",
                        children=[
                            html.Div("Initialize from HRRR (AWS)", className="section-title"),
                            html.Div(
                                className="mw-hrrr-row mw-hrrr-row-compact",
                                children=[
                                    html.Label("Lat °N", className="mw-hrrr-lbl"),
                                    dcc.Input(
                                        id="hrrr-lat",
                                        type="number",
                                        value=40.0,
                                        step=0.01,
                                        className="mw-hrrr-input",
                                    ),
                                    html.Label("Lon °E", className="mw-hrrr-lbl"),
                                    dcc.Input(
                                        id="hrrr-lon",
                                        type="number",
                                        value=-105.5,
                                        step=0.01,
                                        className="mw-hrrr-input",
                                    ),
                                    html.Label(
                                        "Cycle",
                                        className="mw-hrrr-lbl",
                                        title="UTC init time as YYYYMMDDHH, e.g. 2024060112.",
                                    ),
                                    dcc.Input(
                                        id="hrrr-datetime",
                                        type="text",
                                        value="",
                                        placeholder="YYYYMMDDHH",
                                        className="mw-hrrr-input wide",
                                    ),
                                    html.Button(
                                        "Fetch",
                                        id="hrrr-fetch",
                                        className="mw-analyze mw-hrrr-btn",
                                    ),
                                ],
                            ),
                            # Flow-from slider lives on its own row so it can
                            # stretch across the full card. Moving it updates
                            # the along-flow projection *without* triggering
                            # another HRRR download — the raw east/north wind
                            # components are cached in hrrr-raw-store below.
                            html.Div(
                                className="mw-hrrr-dir-row",
                                children=[
                                    html.Label(
                                        "Flow from",
                                        className="mw-hrrr-lbl",
                                        title=(
                                            "Meteorological convention: direction "
                                            "the wind is blowing FROM. 0=N, 90=E, "
                                            "180=S, 270=W. Moving this slider "
                                            "re-projects the cached HRRR wind "
                                            "onto the new axis (no re-download)."
                                        ),
                                    ),
                                    html.Div(
                                        className="mw-hrrr-dir-slider",
                                        children=dcc.Slider(
                                            id="hrrr-dir",
                                            min=0,
                                            max=360,
                                            step=1,
                                            value=270,
                                            marks={
                                                0: "0° N",
                                                90: "90° E",
                                                180: "180° S",
                                                270: "270° W",
                                                360: "360° N",
                                            },
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ),
                                    html.Span(id="hrrr-dir-val", className="mw-hrrr-dir-val"),
                                ],
                            ),
                            dcc.Loading(
                                id="hrrr-loading",
                                type="circle",
                                children=html.Div(
                                    id="hrrr-status",
                                    className="mw-hrrr-status",
                                ),
                            ),
                        ],
                    ),
                    # Actions panel — sits in the 3rd grid column beside the
                    # 2/3-width HRRR card so the Analyze button is always
                    # visible without scrolling past the fetch controls.
                    html.Div(
                        className="section mw-actions-section",
                        children=[
                            html.Div("Actions", className="section-title"),
                            html.Button(
                                "Reset profile",
                                id="reset-profile",
                                className="mw-action-btn",
                            ),
                            html.Button(
                                "Analyze flow",
                                id="analyze-profile",
                                className="mw-analyze mw-action-btn",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="mw-profile-toolbar",
                children=[
                    html.Button(
                        "↶ Undo",
                        id="undo-profile",
                        className="mw-undo",
                        disabled=True,
                        title="Revert the last profile edit (up to 50 steps).",
                    ),
                    html.Span(id="undo-status", className="mw-undo-status"),
                ],
            ),
            html.Div(
                className="mw-profile-editors",
                children=[
                    dcc.Graph(
                        id="u-profile-graph",
                        figure=_profile_editor_figure("Zonal wind u(z)", us, zs, xunit="m s⁻¹", xrange=U_RANGE),
                        config={"edits": {"shapePosition": True}, "displayModeBar": False, "scrollZoom": False},
                    ),
                    dcc.Graph(
                        id="theta-profile-graph",
                        figure=_profile_editor_figure(
                            "Potential temperature θ(z)", thetas, zs, xunit="K", xrange=THETA_RANGE
                        ),
                        config={"edits": {"shapePosition": True}, "displayModeBar": False, "scrollZoom": False},
                    ),
                ],
            ),
            html.Div(
                "Drag the gold circles left/right on each profile to change the u or θ "
                "value at that height. Heights are fixed at 18 evenly-spaced levels; the "
                "blue line shows the current profile and follows the points as you drag. "
                "The four diagnostics below update live — Scorer parameter L² (trapped "
                "waves need L² to decrease aloft), Richardson number Ri (< 0.25 flags "
                "shear instability), Brunt–Väisälä N² (stratification strength), and "
                "the potential-temperature lapse rate dθ/dz. Click 'Analyze flow' to "
                "update the wave solution.",
                style={"color": "#8f98a3", "fontSize": "12px", "marginTop": "6px"},
            ),
            html.Div("Live profile diagnostics", className="section-title",
                     style={"marginTop": "14px"}),
            html.Div(
                className="mw-profile-diagnostics",
                children=[
                    dcc.Graph(id="diag-scorer", config={"displayModeBar": False}),
                    dcc.Graph(id="diag-ri", config={"displayModeBar": False}),
                    dcc.Graph(id="diag-n2", config={"displayModeBar": False}),
                    dcc.Graph(id="diag-dthdz", config={"displayModeBar": False}),
                ],
            ),
        ],
    )


def _profile_editor_figure(title, values, zs, xunit, xrange):
    """Build an editable profile figure.

    Each point is represented as a draggable circle *shape* — Plotly supports
    shape dragging via the ``edits.shapePosition`` config flag — and a line
    trace connects the current values for visualization. When the user drags
    a circle, ``relayoutData`` fires with ``shapes[i].x0``/``x1`` keys, which
    the ``_profile_update`` callback parses back into the shared store. The
    store-driven ``_redraw_profiles`` callback then regenerates both figures
    so the connecting line follows the dragged points.
    """
    # Sanitize any NaN/inf so a corrupted store value can't knock a circle
    # off screen. We do NOT clip to the nominal xrange here — if HRRR
    # delivers a value outside the default axis, extend the axis rather
    # than silently mangling the data. The drag callback applies its own
    # clamp so user-dragged points still stay on screen.
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.5 * (xrange[0] + xrange[1]))
    zs_km = np.asarray(zs, dtype=float) / 1000.0

    # Data-driven axis: take the wider of the default xrange and the actual
    # data span, with a little padding so edge points aren't on the border.
    if values.size:
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        span = max(v_max - v_min, 1.0)
        pad = 0.05 * span
        axis_lo = min(xrange[0], v_min - pad)
        axis_hi = max(xrange[1], v_max + pad)
    else:
        axis_lo, axis_hi = xrange
    effective_xrange = (axis_lo, axis_hi)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=values,
            y=zs_km,
            mode="lines+markers",
            marker=dict(size=8, color="#6ecbff", line=dict(color="white", width=1)),
            line=dict(color="#6ecbff", width=2),
            name=title,
            hoverinfo="skip",
        )
    )

    # One draggable circle per profile point. Pixel sizing makes every
    # shape render as a visually round circle regardless of how stretched
    # the axes are; dragging updates shapes[i].xanchor in relayoutData.
    radius_px = 9
    shapes = []
    for v, zk in zip(values, zs_km):
        shapes.append(
            dict(
                type="circle",
                xref="x",
                yref="y",
                xsizemode="pixel",
                ysizemode="pixel",
                xanchor=float(v),
                yanchor=float(zk),
                x0=-radius_px,
                x1=radius_px,
                y0=-radius_px,
                y1=radius_px,
                fillcolor="#ffd685",
                line=dict(color="white", width=1.5),
                editable=True,
                layer="above",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"value ({xunit})",
        yaxis_title="height (km)",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=50),
        height=PROFILE_FIG_HEIGHT_PX,
        dragmode=False,
        xaxis=dict(range=list(effective_xrange), fixedrange=True),
        yaxis=dict(range=[0, zs_km[-1]], fixedrange=True),
        shapes=shapes,
    )
    return fig


def _profile_diagnostics(zs, us, thetas):
    """Return (L², Ri, N², dθ/dz) arrays evaluated on the profile grid."""
    zs = np.asarray(zs, dtype=float)
    us = np.asarray(us, dtype=float)
    thetas = np.asarray(thetas, dtype=float)
    l2 = scorer_from_profile(zs, us, thetas)
    n2 = brunt_vaisala(zs, thetas)

    # dU/dz and dθ/dz via the same 3-point finite difference used in
    # reference.py / scorer_from_profile so diagnostics stay consistent.
    n = zs.size
    dudz = np.zeros(n)
    dthdz = np.zeros(n)
    for i in range(n):
        if i == 0:
            dudz[i] = (us[1] - us[0]) / (zs[1] - zs[0])
            dthdz[i] = (thetas[1] - thetas[0]) / (zs[1] - zs[0])
        elif i == n - 1:
            dudz[i] = (us[-1] - us[-2]) / (zs[-1] - zs[-2])
            dthdz[i] = (thetas[-1] - thetas[-2]) / (zs[-1] - zs[-2])
        else:
            h1 = zs[i] - zs[i - 1]
            h2 = zs[i + 1] - zs[i]
            dudz[i] = (
                us[i + 1] * h1 ** 2
                - us[i - 1] * h2 ** 2
                + us[i] * (h2 ** 2 - h1 ** 2)
            ) / (h1 * h2 * (h1 + h2))
            dthdz[i] = (
                thetas[i + 1] * h1 ** 2
                - thetas[i - 1] * h2 ** 2
                + thetas[i] * (h2 ** 2 - h1 ** 2)
            ) / (h1 * h2 * (h1 + h2))

    # Ri = N² / (dU/dz)². Guard against vanishing shear.
    with np.errstate(divide="ignore", invalid="ignore"):
        ri = np.where(np.abs(dudz) > 1e-6, n2 / (dudz ** 2), np.inf)
    return l2, ri, n2, dthdz


def _diagnostic_figure(title, values, zs, xunit, color, xrange=None, log_x=False, ref_line=None):
    """Small read-only profile plot of ``values(z)``."""
    values = np.asarray(values, dtype=float)
    zs_km = np.asarray(zs, dtype=float) / 1000.0

    plot_x = values
    if log_x:
        # Use symmetric log to tolerate negative values without dropping them.
        plot_x = values

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_x,
            y=zs_km,
            mode="lines+markers",
            marker=dict(size=6, color=color, line=dict(color="white", width=0.5)),
            line=dict(color=color, width=2),
            hovertemplate=f"%{{x:.3g}} {xunit}<br>%{{y:.2f}} km<extra></extra>",
            showlegend=False,
        )
    )
    if ref_line is not None:
        fig.add_vline(x=ref_line, line=dict(color="#ffd685", dash="dot", width=1))

    xaxis_kwargs = dict(title=f"{xunit}")
    if log_x:
        xaxis_kwargs["type"] = "log"
    if xrange is not None:
        xaxis_kwargs["range"] = list(xrange)

    fig.update_layout(
        title=title,
        yaxis_title="height (km)",
        template="plotly_dark",
        margin=dict(l=50, r=15, t=40, b=40),
        height=340,
        xaxis=xaxis_kwargs,
        yaxis=dict(range=[0, zs_km[-1]]),
    )
    return fig


def _diagnostic_figures(store):
    """Build all four diagnostic figures from the current profile store."""
    zs = np.asarray(store["z"], dtype=float)
    us = np.asarray(store["u"], dtype=float)
    thetas = np.asarray(store["theta"], dtype=float)
    l2, ri, n2, dthdz = _profile_diagnostics(zs, us, thetas)

    # Scorer parameter. Display in units of 10⁻⁸ m⁻² so the magnitudes line
    # up with the two-layer L sliders: L_slider = L × 10⁴ m⁻¹ means
    # L_slider² = L² × 10⁸, so L²(display) directly equals L_slider².
    # E.g. the two-layer defaults of L_lower=10, L_upper=4 correspond to
    # L²_lower=100, L²_upper=16 on this plot.
    l2_scaled = l2 * 1e8
    l_fig = _diagnostic_figure(
        "Scorer parameter L²(z)",
        l2_scaled,
        zs,
        xunit="L² (×10⁻⁸ m⁻²)  — same units as (two-layer L slider)²",
        color="#6ecbff",
        ref_line=0.0,
    )

    # Richardson number. Real-world Ri values span many orders of magnitude
    # — a still-stable free-troposphere profile is easily 100s while the
    # instability threshold is 0.25. Use a log x-axis so that variation is
    # visible; clip negatives (from N² < 0 regions) to a small positive
    # floor and mark them with a trailing annotation instead.
    neg_mask = ri <= 0
    ri_pos = np.where(neg_mask, np.nan, ri)
    ri_pos = np.where(np.isfinite(ri_pos), ri_pos, 1e4)
    ri_pos = np.clip(ri_pos, 1e-2, 1e4)
    ri_fig = _diagnostic_figure(
        "Richardson number Ri(z)",
        ri_pos,
        zs,
        xunit="Ri (log scale, clipped 10⁻² – 10⁴)",
        color="#ffb86c",
        log_x=True,
        ref_line=0.25,
    )

    # N² — units of s⁻², typical tropospheric values ~10⁻⁴; can be negative
    # if the user drags θ(z) into a superadiabatic configuration.
    n2_scaled = n2 * 1e4
    n2_fig = _diagnostic_figure(
        "Brunt–Väisälä N²(z)",
        n2_scaled,
        zs,
        xunit="N² (×10⁻⁴ s⁻²)",
        color="#a0e878",
        ref_line=0.0,
    )

    # dθ/dz in K/km — the raw stratification before dividing by θ.
    dthdz_fig = _diagnostic_figure(
        "Lapse rate dθ/dz(z)",
        dthdz * 1000.0,
        zs,
        xunit="dθ/dz (K km⁻¹)",
        color="#f57aa5",
        ref_line=0.0,
    )
    return l_fig, ri_fig, n2_fig, dthdz_fig


def _register_callbacks(app: Dash):
    # --- latitude readout ------------------------------------------------
    app.clientside_callback(
        "function(v) { return Math.round(v) + '\u00b0 N'; }",
        Output("latit-val", "children"),
        Input("latit", "value"),
    )

    # --- HRRR flow-from slider readout -----------------------------------
    # Shows the current azimuth next to the slider with a compass letter
    # (N/NE/E/.../NW). Purely a UI hint — the server callback still uses
    # the raw degrees when computing the along-flow projection.
    app.clientside_callback(
        """
        function(v) {
            if (v === null || v === undefined) return '';
            var deg = Math.round(v);
            var dirs = ['N','NE','E','SE','S','SW','W','NW','N'];
            var idx = Math.round(deg / 45) % 8;
            return deg + '\u00b0 ' + dirs[idx];
        }
        """,
        Output("hrrr-dir-val", "children"),
        Input("hrrr-dir", "value"),
    )

    # --- live slider readouts --------------------------------------------
    for sid, unit in [
        ("U", " m/s"),
        ("L_upper", ""),
        ("L_lower", ""),
        ("H", " km"),
        ("mtn_h", " km"),
        ("mtn_a", " km"),
        ("xdom", " km"),
        ("zdom", " km"),
        ("mink_k", ""),
        ("maxk_k", ""),
        ("p_mtn_h", " km"),
        ("p_mtn_a", " km"),
        ("p_xdom", " km"),
        ("p_zdom", " km"),
        ("p_mink_k", ""),
        ("p_maxk_k", ""),
    ]:
        app.clientside_callback(
            f"function(v) {{ return (Math.round(v*100)/100) + '{unit}'; }}",
            Output(f"{sid}-val", "children"),
            Input(sid, "value"),
        )

    # --- swap control panel on tab change --------------------------------
    # Both control panels are always in the DOM; we just toggle visibility.
    @app.callback(
        [Output("controls-two-layer", "style"), Output("controls-profile", "style")],
        Input("mode-tabs", "value"),
    )
    def _swap(mode):
        if mode == "two-layer":
            return {}, {"display": "none"}
        return {"display": "none"}, {}

    # --- presets apply to sliders ----------------------------------------
    @app.callback(
        [
            Output("U", "value"),
            Output("L_upper", "value"),
            Output("L_lower", "value"),
            Output("H", "value"),
            Output("mtn_h", "value"),
            Output("mtn_a", "value"),
            Output("xdom", "value"),
            Output("zdom", "value"),
            Output("mink_k", "value"),
            Output("maxk_k", "value"),
        ],
        [
            Input("preset-uniform", "n_clicks"),
            Input("preset-trapped", "n_clicks"),
            Input("preset-downslope", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def _apply_preset(nu, nt, nd):
        from dash import ctx

        trig = ctx.triggered_id
        key = {"preset-uniform": "uniform", "preset-trapped": "trapped", "preset-downslope": "downslope"}.get(trig)
        if key is None:
            return [no_update] * 10
        p = PRESETS[key]
        return [p["U"], p["L_upper"], p["L_lower"], p["H"], p["mtn_h"], p["mtn_a"], p["xdom"], p["zdom"], p["mink_k"], p["maxk_k"]]

    # --- two-layer analyze ------------------------------------------------
    @app.callback(
        [
            Output("streamline-plot", "figure"),
            Output("w-plot", "figure"),
            Output("uprime-plot", "figure"),
            Output("scorer-readout", "children"),
            Output("rossby-readout", "children"),
            Output("nonlin-readout", "children"),
        ],
        [
            Input("analyze-two", "n_clicks"),
            Input("analyze-profile", "n_clicks"),
        ],
        [
            State("mode-tabs", "value"),
            State("U", "value"),
            State("L_upper", "value"),
            State("L_lower", "value"),
            State("H", "value"),
            State("mtn_h", "value"),
            State("mtn_a", "value"),
            State("xdom", "value"),
            State("zdom", "value"),
            State("mink_k", "value"),
            State("maxk_k", "value"),
            State("p_mtn_h", "value"),
            State("p_mtn_a", "value"),
            State("p_xdom", "value"),
            State("p_zdom", "value"),
            State("p_mink_k", "value"),
            State("p_maxk_k", "value"),
            State("profile-store", "data"),
            State("latit", "value"),
        ],
        prevent_initial_call=False,
    )
    def _run(
        n_two,
        n_prof,
        mode,
        U,
        L_upper,
        L_lower,
        H,
        mtn_h,
        mtn_a,
        xdom,
        zdom,
        mink_k,
        maxk_k,
        p_mtn_h,
        p_mtn_a,
        p_xdom,
        p_zdom,
        p_mink_k,
        p_maxk_k,
        store,
        latit_deg,
    ):
        from dash import ctx
        trig = ctx.triggered_id
        print(
            f"[mountain-waves] _run fired: trigger={trig!r} mode={mode!r} "
            f"n_two={n_two} n_prof={n_prof} lat={latit_deg}",
            flush=True,
        )
        lat = latit_deg if latit_deg is not None else 45.0
        try:
            if mode == "profile":
                params = {
                    "mtn_h": p_mtn_h if p_mtn_h is not None else 0.5,
                    "mtn_a": p_mtn_a if p_mtn_a is not None else 2.5,
                    "xdom": p_xdom if p_xdom is not None else 40,
                    "zdom": p_zdom if p_zdom is not None else 10,
                    "mink_k": p_mink_k if p_mink_k is not None else 0,
                    "maxk_k": p_maxk_k if p_maxk_k is not None else 30,
                }
                zs = np.asarray(store["z"])
                us = np.asarray(store["u"])
                thetas = np.asarray(store["theta"])
                sfig, wfig, upfig, scorer, rossby = _profile_plots(params, zs, us, thetas, latit_deg=lat)
            else:
                params = {
                    "U": U if U is not None else 20.0,
                    "L_upper": L_upper if L_upper is not None else 4.0,
                    "L_lower": L_lower if L_lower is not None else 10.0,
                    "H": H if H is not None else 3.5,
                    "mtn_h": mtn_h if mtn_h is not None else 0.5,
                    "mtn_a": mtn_a if mtn_a is not None else 2.5,
                    "xdom": xdom if xdom is not None else 40,
                    "zdom": zdom if zdom is not None else 10,
                    "mink_k": mink_k if mink_k is not None else 0,
                    "maxk_k": maxk_k if maxk_k is not None else 30,
                }
                sfig, wfig, upfig, scorer, rossby = _two_layer_plots(params, latit_deg=lat)
            print(
                f"[mountain-waves]   ok: scorer={scorer:.3f} rossby={rossby:.3f}",
                flush=True,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[mountain-waves] _run FAILED: {exc!r}", flush=True)
            raise

        # Nonlinearity indicator Nh/U = L * h (in the layer where the wave
        # is forced). Linear Fourier/Scorer theory breaks down around
        # Nh/U ≳ 0.85; beyond that, real flow would overturn or transition
        # to hydraulic/downslope-windstorm behavior that this model cannot
        # represent. See Smith (1985) and Hart's "Problems" note.
        if mode == "profile":
            zs = np.asarray(store["z"])
            thetas = np.asarray(store["theta"])
            us = np.asarray(store["u"])
            # crude surface N from finite-difference dθ/dz
            dth = (thetas[1] - thetas[0]) / (zs[1] - zs[0])
            n_sfc = math.sqrt(max(9.80665 / thetas[0] * dth, 0.0))
            u_sfc = abs(float(us[0])) or 1e-6
            h_m = (p_mtn_h if p_mtn_h is not None else 0.5) * 1000.0
            nhu = n_sfc * h_m / u_sfc
        else:
            # L = N/U for uniform U → N = L·U, and Nh/U = L·h.
            L_lower_m = params["L_lower"] * 1e-4
            nhu = L_lower_m * params["mtn_h"] * 1000.0

        if nhu < 0.5:
            flag = "linear regime"
        elif nhu < 0.85:
            flag = "approaching nonlinear"
        elif nhu < 1.2:
            flag = "near breaking — results questionable"
        else:
            flag = "past breaking — model unreliable"

        scorer_badge = html.Span(
            [
                html.B("Scorer condition"),
                html.Span(f"{scorer:.3f}", className="v"),
                html.Span("(trapped > 1)", className="note"),
            ]
        )
        rossby_badge = html.Span(
            [
                html.B(f"Rossby number ({lat:.0f}°N)"),
                html.Span(f"{rossby:.2f}", className="v"),
                html.Span("(Coriolis significant if ≲ 1)", className="note"),
            ]
        )
        nonlin_badge = html.Span(
            [
                html.B("Nh/U (nonlinearity)"),
                html.Span(f"{nhu:.2f}", className="v"),
                html.Span(f"({flag})", className="note"),
            ]
        )
        return sfig, wfig, upfig, scorer_badge, rossby_badge, nonlin_badge

    # --- profile editing ---------------------------------------------------
    # Drag-on-graph → update store. Each profile point is a Plotly "shape",
    # and Plotly emits relayoutData entries like {"shapes[3].xanchor": ...}
    # when a shape is dragged. We read that, clamp it into the axis range
    # (so a stray drag off-screen cannot turn the profile into nonsense),
    # and push the previous store onto the undo history before committing.
    @app.callback(
        [
            Output("profile-store", "data", allow_duplicate=True),
            Output("profile-history", "data", allow_duplicate=True),
        ],
        [
            Input("u-profile-graph", "relayoutData"),
            Input("theta-profile-graph", "relayoutData"),
            Input("reset-profile", "n_clicks"),
            Input("undo-profile", "n_clicks"),
        ],
        [
            State("profile-store", "data"),
            State("profile-history", "data"),
        ],
        prevent_initial_call=True,
    )
    def _profile_update(u_relay, th_relay, reset_clicks, undo_clicks, store, history):
        from dash import ctx

        trig = ctx.triggered_id
        history = list(history or [])

        def _push(snapshot):
            # Deduplicate and cap length.
            if history and history[-1] == snapshot:
                return
            history.append(snapshot)
            if len(history) > UNDO_HISTORY_MAX:
                del history[: len(history) - UNDO_HISTORY_MAX]

        def _snapshot(s):
            # Deep-ish copy — plain lists/floats survive JSON round-tripping fine.
            return {
                "z": list(s["z"]),
                "u": list(s["u"]),
                "theta": list(s["theta"]),
            }

        # ---- undo -------------------------------------------------------
        if trig == "undo-profile":
            if not history:
                return no_update, no_update
            prev = history.pop()
            return prev, history

        # ---- reset ------------------------------------------------------
        if trig == "reset-profile":
            _push(_snapshot(store))
            zs, us, thetas = _init_profile_figures()
            return (
                {"z": zs.tolist(), "u": us.tolist(), "theta": thetas.tolist()},
                history,
            )

        # ---- drag -------------------------------------------------------
        lo, hi = (U_RANGE if trig == "u-profile-graph" else THETA_RANGE)

        def _has_shape_keys(relay):
            return bool(relay) and any(
                isinstance(k, str) and k.startswith("shapes[") for k in relay.keys()
            )

        def _apply(relay, current):
            """Pull new values out of a relayoutData payload and clamp them.

            We ONLY honor ``shapes[i].xanchor`` here. Pixel-sized circles
            also emit ``shapes[i].x0`` / ``x1`` when the user accidentally
            grabs a resize handle — but those are *pixel* offsets, not data
            coordinates, so averaging them yields garbage (which was the
            bug behind "value shoots off-screen" reports). Rejecting
            resize emissions means an edge-grab just does nothing instead
            of corrupting the profile.
            """
            if not relay:
                return current, False
            out = list(current)
            changed = False
            for i in range(len(out)):
                kxa = f"shapes[{i}].xanchor"
                if kxa not in relay:
                    continue
                try:
                    v = float(relay[kxa])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                v = max(lo, min(hi, v))
                if v != out[i]:
                    out[i] = v
                    changed = True
            return out, changed

        if trig == "u-profile-graph":
            relay, field = u_relay, "u"
        elif trig == "theta-profile-graph":
            relay, field = th_relay, "theta"
        else:
            return no_update, no_update

        new_values, changed = _apply(relay, store[field])
        shape_edit = _has_shape_keys(relay)
        if changed:
            _push(_snapshot(store))
            store[field] = new_values
        if not changed and not shape_edit:
            # Nothing shape-related happened (pan, zoom, autosize) —
            # leave the store alone so _redraw_profiles doesn't fire.
            return no_update, no_update
        # Bump a revision stamp on every shape event so the redraw
        # callback fires even if the edit was rejected (e.g. a resize-
        # handle grab). This snaps any stretched oval back to its
        # canonical round shape.
        store["_rev"] = int(store.get("_rev", 0)) + 1
        return store, history

    # Enable/disable the undo button and show a depth hint.
    @app.callback(
        [
            Output("undo-profile", "disabled"),
            Output("undo-status", "children"),
        ],
        Input("profile-history", "data"),
    )
    def _undo_status(history):
        n = len(history or [])
        if n == 0:
            return True, ""
        return False, f"{n} edit{'s' if n != 1 else ''} in history"

    # --- HRRR profile fetch ----------------------------------------------
    # Pulls U, V, T, HGT from the HRRR 0-h analysis on AWS (byte-range
    # subset via the .idx sidecar), computes θ, and caches the raw east/north
    # wind components in ``hrrr-raw-store``. The initial along-flow projection
    # uses the current ``hrrr-dir`` slider value; further slider moves reuse
    # the cached column (see ``_hrrr_redirect``) so no re-download is needed.
    @app.callback(
        [
            Output("profile-store", "data", allow_duplicate=True),
            Output("profile-history", "data", allow_duplicate=True),
            Output("hrrr-raw-store", "data"),
            Output("hrrr-status", "children"),
            Output("hrrr-status", "className"),
        ],
        Input("hrrr-fetch", "n_clicks"),
        [
            State("hrrr-lat", "value"),
            State("hrrr-lon", "value"),
            State("hrrr-dir", "value"),
            State("hrrr-datetime", "value"),
            State("profile-store", "data"),
            State("profile-history", "data"),
        ],
        prevent_initial_call=True,
    )
    def _hrrr_fetch(n_clicks, lat, lon, flow_deg, yyyymmddhh, store, history):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update

        # Basic input validation — bail out early with a clear message
        # rather than making the user wait on a network round-trip.
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            flow_f = float(flow_deg)
        except (TypeError, ValueError):
            return (
                no_update,
                no_update,
                no_update,
                "Lat / Lon / Flow dir must all be numbers.",
                "mw-hrrr-status error",
            )
        if not (-90.0 <= lat_f <= 90.0):
            return (
                no_update,
                no_update,
                no_update,
                f"Lat {lat_f} outside [-90, 90].",
                "mw-hrrr-status error",
            )

        try:
            from .hrrr import along_flow_positive, fetch_profile
        except ImportError as exc:
            return (
                no_update,
                no_update,
                no_update,
                f"HRRR module failed to import: {exc}",
                "mw-hrrr-status error",
            )

        zs_target = np.asarray(store["z"], dtype=float)
        try:
            z, u_raw, v_raw, theta, meta = fetch_profile(
                lat_f, lon_f, yyyymmddhh, z_target_m=zs_target
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return (
                no_update,
                no_update,
                no_update,
                f"Fetch failed: {exc}",
                "mw-hrrr-status error",
            )

        # Don't clip HRRR data — the widened U_RANGE / THETA_RANGE should
        # accommodate most real columns, and silently clamping was masking
        # real wind and temperature structure (negative along-flow legs,
        # cold-air-mass surface θ, etc.). If a sample still falls outside
        # the axis, the editor auto-extends at render time.
        u_arr = np.asarray(u_raw, dtype=float)
        v_arr = np.asarray(v_raw, dtype=float)
        th_arr = np.asarray(theta, dtype=float)
        u_along = along_flow_positive(u_arr, v_arr, flow_f)

        # Push current state onto history so the fetch is undoable.
        history = list(history or [])
        snap = {"z": list(store["z"]), "u": list(store["u"]), "theta": list(store["theta"])}
        if not history or history[-1] != snap:
            history.append(snap)
            if len(history) > UNDO_HISTORY_MAX:
                del history[: len(history) - UNDO_HISTORY_MAX]

        new_store = {
            "z": zs_target.tolist(),
            "u": u_along.tolist(),
            "theta": th_arr.tolist(),
            "_rev": int(store.get("_rev", 0)) + 1,
        }
        # Cache raw east/north wind (and θ / z) so the slider can re-project
        # without re-downloading. Meta is kept so the status message can be
        # refreshed after a reprojection.
        raw_store = {
            "z": zs_target.tolist(),
            "u": u_arr.tolist(),
            "v": v_arr.tolist(),
            "theta": th_arr.tolist(),
            "meta": {
                "s3_key": meta.get("s3_key"),
                "grid_lat": float(meta.get("grid_lat", float("nan"))),
                "grid_lon": float(meta.get("grid_lon", float("nan"))),
                "bytes": int(meta.get("bytes", 0)),
                "n_levels": int(meta.get("n_levels", 0)),
                "cycle": yyyymmddhh,
            },
        }
        mb = meta["bytes"] / 1e6
        msg = (
            f"HRRR {yyyymmddhh} at grid point "
            f"({meta['grid_lat']:.3f}°N, {meta['grid_lon']:.3f}°E) — "
            f"{meta['n_levels']} levels, {mb:.1f} MB fetched. "
            f"Flow from {flow_f:.0f}°."
        )
        return new_store, history, raw_store, msg, "mw-hrrr-status ok"

    # --- HRRR reprojection on slider change ------------------------------
    # When the flow-from slider moves, reuse the cached raw column (if any)
    # and rewrite just the profile-store's u field. No network activity.
    @app.callback(
        [
            Output("profile-store", "data", allow_duplicate=True),
            Output("hrrr-status", "children", allow_duplicate=True),
            Output("hrrr-status", "className", allow_duplicate=True),
        ],
        Input("hrrr-dir", "value"),
        [
            State("hrrr-raw-store", "data"),
            State("profile-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def _hrrr_redirect(flow_deg, raw, store):
        # No cached column yet — slider moves before any fetch should be
        # silent no-ops. (The next Fetch will pick up the current value.)
        if not raw or "u" not in raw or "v" not in raw:
            return no_update, no_update, no_update
        try:
            flow_f = float(flow_deg)
        except (TypeError, ValueError):
            return no_update, no_update, no_update

        try:
            from .hrrr import along_flow_positive
        except ImportError:
            return no_update, no_update, no_update

        u_raw = np.asarray(raw["u"], dtype=float)
        v_raw = np.asarray(raw["v"], dtype=float)
        u_along = along_flow_positive(u_raw, v_raw, flow_f)

        new_store = {
            "z": list(raw["z"]),
            "u": u_along.tolist(),
            "theta": list(raw["theta"]),
            "_rev": int((store or {}).get("_rev", 0)) + 1,
        }
        meta = raw.get("meta", {}) or {}
        cycle = meta.get("cycle", "")
        mb = float(meta.get("bytes", 0)) / 1e6
        n_levels = int(meta.get("n_levels", 0))
        glat = meta.get("grid_lat")
        glon = meta.get("grid_lon")
        if glat is not None and glon is not None:
            loc = f"({glat:.3f}°N, {glon:.3f}°E)"
        else:
            loc = ""
        msg = (
            f"HRRR {cycle} at grid point {loc} — {n_levels} levels, "
            f"{mb:.1f} MB cached. Reprojected: flow from {flow_f:.0f}°."
        )
        return new_store, msg, "mw-hrrr-status ok"

    # Store → figures. Whenever the store changes we regenerate both
    # editable profile figures (so the line follows the dragged shapes)
    # and all four diagnostic profiles (L², Ri, N², dθ/dz).
    @app.callback(
        [
            Output("u-profile-graph", "figure"),
            Output("theta-profile-graph", "figure"),
            Output("diag-scorer", "figure"),
            Output("diag-ri", "figure"),
            Output("diag-n2", "figure"),
            Output("diag-dthdz", "figure"),
        ],
        Input("profile-store", "data"),
    )
    def _redraw_profiles(store):
        zs = np.asarray(store["z"])
        us = np.asarray(store["u"])
        thetas = np.asarray(store["theta"])
        u_fig = _profile_editor_figure("Zonal wind u(z)", us, zs, xunit="m s⁻¹", xrange=U_RANGE)
        th_fig = _profile_editor_figure(
            "Potential temperature θ(z)", thetas, zs, xunit="K", xrange=THETA_RANGE
        )
        l_fig, ri_fig, n2_fig, dthdz_fig = _diagnostic_figures(store)
        return u_fig, th_fig, l_fig, ri_fig, n2_fig, dthdz_fig


def main():
    app = create_app()
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
