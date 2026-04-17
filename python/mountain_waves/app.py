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
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

from . import solver
from .profile import (
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
        "label": "Downslope windstorm (Example 3)",
        "U": 30.0,
        "L_upper": 3.0,
        "L_lower": 12.0,
        "H": 2.0,
        "mtn_h": 1.5,
        "mtn_a": 2.0,
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


def _two_layer_plots(params, latit_deg: float = 45.0) -> Tuple[go.Figure, go.Figure, float, float]:
    """Run the two-layer solver and build the streamline + w contour figures."""
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

    x, z, w = solver.compute_two_layer(L_upper, L_lower, U, H, a, ho, xdom, zdom, mink, maxk, npts=100)
    lines = solver.streamlines(x, z, U, w, num=10)

    scorer_cond = 4.0 * H * H * (L_lower ** 2 - L_upper ** 2) / (math.pi ** 2)
    rossby = _rossby(U, a, latit_deg)

    return (*_make_plots(x, z, w, lines, H), scorer_cond, rossby)


def _profile_plots(params, z_prof, u_prof, theta_prof, latit_deg: float = 45.0) -> Tuple[go.Figure, go.Figure, float, float]:
    """Run the multi-layer solver from prescribed profiles."""
    U_surface = float(u_prof[0])
    a = params["mtn_a"] * 1000.0
    ho = params["mtn_h"] * 1000.0
    xdom = params["xdom"] * 1000.0
    zdom = params["zdom"] * 1000.0
    mink = params["mink_k"] / a
    maxk = params["maxk_k"] / a

    x, z, w = solver.compute_from_profile(
        z_prof, u_prof, theta_prof, a, ho, xdom, zdom, mink, maxk, npts=100
    )
    lines = solver.streamlines(x, z, U_surface, w, num=10)

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

    return (*_make_plots(x, z, w, lines, interface_z), scorer_cond, rossby)


def _make_plots(x, z, w, lines, interface_z):
    """Build streamline and vertical-velocity figures from solver output."""
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
        title="Vertical Velocity (m/s)",
        xaxis_title="x (km)",
        yaxis_title="height (km)",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )

    return stream_fig, w_fig


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


def _init_profile_figures():
    # Defaults tuned to reproduce MATLAB "Example 2" (trapped lee waves):
    # sharp low-level stability, weak stability aloft, light surface shear.
    zs = default_profile_heights(10.0, 9)
    us = default_u_profile(zs, u_surface=20.0, shear=0.5)
    thetas = default_theta_profile(zs, interface_km=3.5, lapse_lower=11.8, lapse_upper=1.9)
    return zs, us, thetas


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Dash:
    app = Dash(__name__, title="Mountain Waves", update_title=None)

    zs0, us0, thetas0 = _init_profile_figures()

    app.layout = html.Div(
        className="mw-root",
        children=[
            html.Header(
                className="mw-header",
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
            html.Div(id="controls-block"),
            html.Div(
                className="mw-diagnostics",
                children=[
                    html.Div(id="scorer-readout", className="diag-card"),
                    html.Div(id="rossby-readout", className="diag-card"),
                ],
            ),
            html.Div(
                className="mw-plots",
                children=[
                    dcc.Graph(id="streamline-plot", config={"displayModeBar": False}),
                    dcc.Graph(id="w-plot", config={"displayModeBar": False}),
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
            html.Footer(
                className="mw-footer",
                children=[
                    html.Span("Drag points on the u(z) and θ(z) graphs to edit the profile. "),
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
.mw-header h1 { margin: 0 0 4px 0; font-size: 24px; }
.mw-subtitle { color: #9aa3ad; font-size: 13px; margin-bottom: 16px; }
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
.mw-profile-editors { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px; }
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
                            html.Button("Downslope windstorm", id="preset-downslope"),
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
                            html.Div(
                                className="mw-presets",
                                children=[
                                    html.Button("Reset profile", id="reset-profile"),
                                    html.Button("Analyze flow", id="analyze-profile", className="mw-analyze"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="mw-profile-editors",
                children=[
                    dcc.Graph(
                        id="u-profile-graph",
                        figure=_profile_editor_figure("Zonal wind u(z)", us, zs, xunit="m s⁻¹", xrange=(0, 60)),
                        config={"editable": True, "edits": {"shapePosition": True}, "displayModeBar": False},
                    ),
                    dcc.Graph(
                        id="theta-profile-graph",
                        figure=_profile_editor_figure(
                            "Potential temperature θ(z)", thetas, zs, xunit="K", xrange=(280, 380)
                        ),
                        config={"editable": True, "edits": {"shapePosition": True}, "displayModeBar": False},
                    ),
                ],
            ),
            html.Div(
                "Drag individual points up/down on the profile graphs to edit. Heights are fixed at 9 "
                "evenly-spaced levels; values are interpolated linearly between them.",
                style={"color": "#8f98a3", "fontSize": "12px", "marginTop": "6px"},
            ),
        ],
    )


def _profile_editor_figure(title, values, zs, xunit, xrange):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=values,
            y=zs / 1000.0,
            mode="lines+markers",
            marker=dict(size=12, color="#6ecbff", line=dict(color="white", width=1)),
            line=dict(color="#6ecbff", width=2),
            name=title,
            hovertemplate=f"%{{x:.2f}} {xunit}<br>%{{y:.2f}} km<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"value ({xunit})",
        yaxis_title="height (km)",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=50),
        height=340,
        dragmode="pan",
        xaxis=dict(range=list(xrange)),
        yaxis=dict(range=[0, zs[-1] / 1000.0]),
    )
    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=12),
    )
    return fig


def _register_callbacks(app: Dash):
    # --- latitude readout ------------------------------------------------
    app.clientside_callback(
        "function(v) { return Math.round(v) + '\u00b0 N'; }",
        Output("latit-val", "children"),
        Input("latit", "value"),
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
    @app.callback(
        Output("controls-block", "children"),
        Input("mode-tabs", "value"),
        State("profile-store", "data"),
    )
    def _swap(mode, store):
        if mode == "two-layer":
            return _two_layer_controls()
        zs = np.asarray(store["z"])
        us = np.asarray(store["u"])
        thetas = np.asarray(store["theta"])
        return _profile_controls(zs, us, thetas)

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
            Output("scorer-readout", "children"),
            Output("rossby-readout", "children"),
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
        lat = latit_deg if latit_deg is not None else 45.0
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
            sfig, wfig, scorer, rossby = _profile_plots(params, zs, us, thetas, latit_deg=lat)
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
            sfig, wfig, scorer, rossby = _two_layer_plots(params, latit_deg=lat)

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
        return sfig, wfig, scorer_badge, rossby_badge

    # --- profile editing ---------------------------------------------------
    @app.callback(
        Output("profile-store", "data"),
        [
            Input("u-profile-graph", "relayoutData"),
            Input("theta-profile-graph", "relayoutData"),
            Input("u-profile-graph", "restyleData"),
            Input("theta-profile-graph", "restyleData"),
            Input("reset-profile", "n_clicks"),
        ],
        [State("u-profile-graph", "figure"), State("theta-profile-graph", "figure"), State("profile-store", "data")],
        prevent_initial_call=True,
    )
    def _profile_update(u_relay, th_relay, u_re, th_re, reset_clicks, u_fig, th_fig, store):
        from dash import ctx

        trig = ctx.triggered_id
        if trig == "reset-profile":
            zs, us, thetas = _init_profile_figures()
            return {"z": zs.tolist(), "u": us.tolist(), "theta": thetas.tolist()}

        # Plotly's editable traces expose new coordinates in the figure State.
        try:
            if u_fig and u_fig.get("data"):
                tr = u_fig["data"][0]
                new_u = [float(v) for v in tr.get("x", [])]
                if len(new_u) == len(store["u"]):
                    store["u"] = new_u
            if th_fig and th_fig.get("data"):
                tr = th_fig["data"][0]
                new_th = [float(v) for v in tr.get("x", [])]
                if len(new_th) == len(store["theta"]):
                    store["theta"] = new_th
        except Exception:
            pass
        return store


def main():
    app = create_app()
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
