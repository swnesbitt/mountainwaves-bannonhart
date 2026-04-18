"""HRRR-on-AWS profile fetcher.

Pulls a vertical profile of wind and temperature out of the HRRR analysis
(0-h forecast) on the NODD public bucket ``noaa-hrrr-bdp-pds`` and returns
it as (z, u_along, theta) arrays ready to drop into the profile editor.

Design notes
------------
The full ``wrfprsf00.grib2`` is ~140 MB. We only need a handful of fields
(HGT, TMP, UGRD, VGRD on pressure levels) at a single grid cell, so we:

1. Fetch the ``.idx`` sidecar (a few kB) to find byte offsets.
2. Issue ranged GETs for just the GRIB messages we need (few MB total).
3. Splice them together into a local scratch file.
4. Open with cfgrib, pick the nearest grid column, and compute θ and the
   user-specified flow-direction wind component.

Dependencies: ``boto3``, ``xarray``, ``cfgrib``. cfgrib needs the eccodes
C library; we pull it in from PyPI only (via ``eccodes`` + ``eccodeslib``
on macOS/Linux, or ``eccodes`` + ``ecmwflibs`` on Windows) rather than a
system package, so ``uv sync`` is enough and no ``brew install eccodes``
step is required. See ``pyproject.toml`` for the platform markers.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np

BUCKET = "noaa-hrrr-bdp-pds"
R_OVER_CP = 0.2854  # R_d / c_p for dry air

# Variables and level type we care about.
_WANTED_VARS = ("HGT", "TMP", "UGRD", "VGRD")
_LEVEL_SUFFIX = "mb"


@dataclass
class _IdxRecord:
    num: int
    start: int
    var: str
    level_mb: float


def _parse_yyyymmddhh(s: str) -> datetime:
    s = (s or "").strip()
    if len(s) != 10 or not s.isdigit():
        raise ValueError(
            f"Expected YYYYMMDDHH (10 digits); got {s!r}. "
            f"Example: 2024060112"
        )
    return datetime.strptime(s, "%Y%m%d%H")


def _s3_key(dt: datetime) -> str:
    return f"hrrr.{dt:%Y%m%d}/conus/hrrr.t{dt:%H}z.wrfprsf00.grib2"


def _unsigned_s3_client():
    """boto3 client configured for anonymous (unsigned) access."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _parse_idx(idx_text: str) -> Tuple[List[_IdxRecord], List[int]]:
    """Parse the HRRR .idx file into (wanted_records, all_start_bytes).

    Format of each line (colon-separated):
        record_num:start_byte:d=YYYYMMDDHH:VAR:LEVEL:FCST:anl
    """
    wanted: List[_IdxRecord] = []
    all_starts: List[int] = []
    for line in idx_text.splitlines():
        parts = line.split(":")
        if len(parts) < 6 or not parts[1].isdigit():
            continue
        start = int(parts[1])
        all_starts.append(start)
        var = parts[3]
        level = parts[4].strip()
        if var not in _WANTED_VARS or not level.endswith(_LEVEL_SUFFIX):
            continue
        try:
            level_mb = float(level.split()[0])
        except ValueError:
            continue
        wanted.append(_IdxRecord(num=int(parts[0]), start=start, var=var, level_mb=level_mb))
    return wanted, sorted(all_starts)


def _byte_ranges(records: List[_IdxRecord], all_starts: List[int]) -> List[Tuple[int, int]]:
    """Convert each wanted record's start byte to a (start, end) range.

    End byte is the next record's start minus one, or open-ended for the
    final record. We collapse adjacent ranges into contiguous chunks to
    cut down on the number of HTTP calls.
    """
    ranges: List[Tuple[int, int]] = []
    for r in records:
        i = all_starts.index(r.start)
        if i + 1 < len(all_starts):
            end = all_starts[i + 1] - 1
        else:
            end = -1  # open-ended
        ranges.append((r.start, end))
    # Merge contiguous ranges to reduce request count.
    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in ranges:
        if merged and end != -1 and merged[-1][1] != -1 and start == merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _download_subset(client, key: str, ranges: List[Tuple[int, int]]) -> bytes:
    buf = io.BytesIO()
    for start, end in ranges:
        header = f"bytes={start}-" + ("" if end == -1 else str(end))
        obj = client.get_object(Bucket=BUCKET, Key=key, Range=header)
        buf.write(obj["Body"].read())
    return buf.getvalue()


def _nearest_ij(lats: np.ndarray, lons: np.ndarray, lat0: float, lon0: float) -> Tuple[int, int]:
    """Nearest-neighbor grid cell to (lat0, lon0) on a 2-D HRRR grid."""
    lon0 = ((lon0 + 180) % 360) - 180
    # Convert HRRR lons to -180..180 as well.
    lons = ((lons + 180) % 360) - 180
    d2 = (lats - lat0) ** 2 + (lons - lon0) ** 2
    j, i = np.unravel_index(np.argmin(d2), d2.shape)
    return int(j), int(i)


def along_flow_signed(
    u: np.ndarray, v: np.ndarray, flow_from_deg: float
) -> np.ndarray:
    """Signed along-flow wind component for the mountain-wave solver.

    ``flow_from_deg`` follows the standard meteorological convention — the
    azimuth the wind is blowing *from* (270° = westerly, 160° = from the SSE).
    The returned scalar is the signed component of the wind parallel to
    that "from" direction: **positive** when the wind is blowing *from* a
    direction within 90° of ``flow_from_deg``, and **negative** when the
    wind reverses relative to that reference direction. Callers that
    depended on the old zero-clipped behavior should take
    ``np.maximum(along_flow_signed(...), 0.0)`` explicitly; the solver
    itself now tolerates negative U via the Scorer-parameter critical-level
    clamp, so wind reversals aloft should pass through unmodified and be
    surfaced to the user as actual reversals.

    Derivation: a wind with components ``(u, v)`` (east- and north-positive)
    blowing *from* azimuth ``φ_act`` has magnitude ``s`` and
    ``(u, v) = -s · (sin φ_act, cos φ_act)``. Projecting onto the unit
    vector pointing in the direction the flow is going when it comes from
    ``φ_spec`` (i.e. ``φ_spec + 180``) gives ``s · cos(φ_act − φ_spec)``.
    That evaluates to ``-(u sin φ_spec + v cos φ_spec)``, positive when
    actual and specified "from" directions are parallel and negative when
    antiparallel. No clamping is applied.
    """
    rad = np.deg2rad(flow_from_deg)
    return -(np.asarray(u) * np.sin(rad) + np.asarray(v) * np.cos(rad))


# Backward-compatibility alias — the old name is retained so external
# imports keep working, but it now returns the *signed* along-flow
# component (no zero-clip). Callers that genuinely need the clipped
# variant must apply ``np.maximum(_, 0.0)`` themselves.
along_flow_positive = along_flow_signed


def _theta(T_K: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """Potential temperature (K) referenced to 1000 hPa."""
    return T_K * (1000.0 / p_hpa) ** R_OVER_CP


def fetch_profile(
    lat: float,
    lon: float,
    yyyymmddhh: str,
    z_target_m: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return ``(z, u, v, theta, meta)`` for the HRRR column at (lat, lon).

    Raw east/north wind components are returned (not the along-flow
    projection) so the caller can re-project onto any user-chosen flow
    direction cheaply without re-downloading. Use :func:`along_flow_signed`
    to turn ``(u, v, flow_from_deg)`` into the mountain-wave input; wind
    reversals produce negative values, which the solver handles via the
    Scorer critical-level clamp rather than silently clipping to zero.

    Parameters
    ----------
    lat, lon : float
        Point of interest in decimal degrees. ``lon`` may be ±180 or 0..360.
    yyyymmddhh : str
        UTC cycle time, e.g. ``"2024060112"``.
    z_target_m : np.ndarray, optional
        If given, the profile is linearly interpolated onto these heights
        (meters above ground level). If ``None``, the native HRRR
        pressure-level heights (AGL) are returned as ``z``.

    Returns
    -------
    z : np.ndarray
        Heights in m AGL.
    u, v : np.ndarray
        East- and north-positive wind components in m/s.
    theta : np.ndarray
        Potential temperature in K.
    meta : dict
        Diagnostic info (nearest grid lat/lon, S3 key, bytes transferred).
    """
    try:
        import xarray as xr  # noqa: F401  (used via cfgrib backend)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "xarray is required to read HRRR GRIB2. "
            "Install with: pip install xarray cfgrib"
        ) from exc

    dt = _parse_yyyymmddhh(yyyymmddhh)
    key = _s3_key(dt)
    client = _unsigned_s3_client()

    # 1. Fetch the .idx sidecar.
    try:
        idx_obj = client.get_object(Bucket=BUCKET, Key=key + ".idx")
    except Exception as exc:
        raise RuntimeError(
            f"HRRR .idx not found at s3://{BUCKET}/{key}.idx "
            f"(cycle may not exist yet). Error: {exc}"
        ) from exc
    idx_text = idx_obj["Body"].read().decode("utf-8", errors="replace")
    records, all_starts = _parse_idx(idx_text)
    if not records:
        raise RuntimeError(f"No HGT/TMP/UGRD/VGRD pressure-level records in idx for {key}")

    # 2. Byte-range fetch just the records we need.
    ranges = _byte_ranges(records, all_starts)
    blob = _download_subset(client, key, ranges)
    bytes_downloaded = len(blob)

    # 3. Splice into a scratch file and open with cfgrib.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as f:
        f.write(blob)
        grib_path = f.name

    try:
        import xarray as xr

        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",  # don't leave .idx files around
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            },
        )

        # cfgrib exposes variables as {'gh' or 'HGT', 't', 'u', 'v'} depending on
        # shortName/cfName. Find them robustly.
        def _pick(ds, candidates):
            for name in candidates:
                if name in ds.variables:
                    return ds[name]
            raise KeyError(f"None of {candidates} found in dataset: {list(ds.variables)}")

        # cfgrib reads data lazily — it keeps the grib file open and re-reads
        # when .values is touched. We delete the scratch file below, so we
        # must materialize every array we care about *before* the unlink.
        hgt_a = _pick(ds, ["gh", "HGT", "h"]).values
        tmp_a = _pick(ds, ["t", "TMP"]).values
        ugrd_a = _pick(ds, ["u", "UGRD"]).values
        vgrd_a = _pick(ds, ["v", "VGRD"]).values
        p_dim = "isobaricInhPa"
        pressures_full = ds[p_dim].values.astype(float)  # hPa
        lats_full = ds["latitude"].values
        lons_full = ds["longitude"].values
        ds.close()
    finally:
        try:
            os.unlink(grib_path)
        except OSError:
            pass

    # 4. Pick the nearest column.
    j, i = _nearest_ij(lats_full, lons_full, lat, lon)
    grid_lat = float(lats_full[j, i])
    grid_lon = float(((lons_full[j, i] + 180) % 360) - 180)
    pressures = pressures_full

    h_col = hgt_a[:, j, i]  # geopotential height, m (MSL)
    t_col = tmp_a[:, j, i]  # K
    u_col = ugrd_a[:, j, i]
    v_col = vgrd_a[:, j, i]

    # Sort by pressure descending (so surface first, top last).
    order = np.argsort(-pressures)
    pressures = pressures[order]
    h_col = h_col[order]
    t_col = t_col[order]
    u_col = u_col[order]
    v_col = v_col[order]

    # Keep only levels at or above ground (HRRR pressure levels below the
    # surface are filled with extrapolated values — drop those by requiring
    # monotonic height increase from the surface up).
    sfc_h = float(np.min(h_col))
    valid = h_col >= sfc_h - 1.0
    h_col = h_col[valid]
    t_col = t_col[valid]
    u_col = u_col[valid]
    v_col = v_col[valid]
    pressures = pressures[valid]

    # Make strictly monotonic increasing in height (in case of ties).
    order = np.argsort(h_col)
    h_col = h_col[order]
    t_col = t_col[order]
    u_col = u_col[order]
    v_col = v_col[order]
    pressures = pressures[order]

    # Convert MSL heights to AGL by subtracting the lowest valid level.
    z_agl = h_col - h_col[0]

    theta = _theta(t_col, pressures)

    if z_target_m is not None:
        z_target = np.asarray(z_target_m, dtype=float)
        # Clip target range to what HRRR actually covers at this point.
        z_clipped = np.clip(z_target, float(z_agl[0]), float(z_agl[-1]))
        u_out = np.interp(z_clipped, z_agl, u_col)
        v_out = np.interp(z_clipped, z_agl, v_col)
        th_out = np.interp(z_clipped, z_agl, theta)
        z_out = z_target.copy()
    else:
        z_out = z_agl
        u_out = u_col
        v_out = v_col
        th_out = theta

    meta = {
        "s3_key": key,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "bytes": bytes_downloaded,
        "n_levels": int(z_agl.size),
        "sfc_height_msl": float(h_col[0]),
    }
    return (
        z_out.astype(float),
        u_out.astype(float),
        v_out.astype(float),
        th_out.astype(float),
        meta,
    )
