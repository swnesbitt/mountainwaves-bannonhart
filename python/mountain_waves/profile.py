"""Helpers for building and interpreting theta(z) / u(z) profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

G = 9.80665


def default_profile_heights(zdom_km: float = 10.0, n_points: int = 9) -> np.ndarray:
    """Return ``n_points`` evenly spaced heights (meters) from 0 to ``zdom_km`` km."""
    return np.linspace(0.0, zdom_km * 1000.0, n_points)


def default_u_profile(zs: np.ndarray, u_surface: float = 20.0, shear: float = 0.5) -> np.ndarray:
    """A gently sheared wind profile: ``u(z) = u_surface + shear * z_km``."""
    zs_km = zs / 1000.0
    return u_surface + shear * zs_km


def default_theta_profile(
    zs: np.ndarray,
    theta_surface: float = 290.0,
    lapse_lower: float = 3.0,
    lapse_upper: float = 6.0,
    interface_km: float = 3.5,
) -> np.ndarray:
    """A two-regime potential-temperature profile.

    Stability contrast across ``interface_km`` mirrors the trapped-wave case
    in the original MATLAB example: weaker stability below, stronger above.
    """
    zs_km = zs / 1000.0
    theta = np.empty_like(zs_km)
    for i, zkm in enumerate(zs_km):
        if zkm <= interface_km:
            theta[i] = theta_surface + lapse_lower * zkm
        else:
            theta[i] = (
                theta_surface
                + lapse_lower * interface_km
                + lapse_upper * (zkm - interface_km)
            )
    return theta


def brunt_vaisala(z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Finite-difference Brunt–Väisälä frequency squared (s⁻²)."""
    z = np.asarray(z, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n = z.size
    n2 = np.empty(n)
    for i in range(n):
        if i == 0:
            dthdz = (theta[1] - theta[0]) / (z[1] - z[0])
        elif i == n - 1:
            dthdz = (theta[-1] - theta[-2]) / (z[-1] - z[-2])
        else:
            dthdz = (theta[i + 1] - theta[i - 1]) / (z[i + 1] - z[i - 1])
        n2[i] = (G / theta[i]) * dthdz
    return n2


def scorer_from_profile(z, u, theta) -> np.ndarray:
    """Wrapper around the reference implementation for use in the UI."""
    from .reference import scorer_from_profile as _imp
    return _imp(z, u, theta)


@dataclass
class WaveProfile:
    """Container for an edited profile displayed in the Dash app."""

    z: np.ndarray = field(default_factory=lambda: default_profile_heights())
    u: np.ndarray = field(default_factory=lambda: default_u_profile(default_profile_heights()))
    theta: np.ndarray = field(default_factory=lambda: default_theta_profile(default_profile_heights()))

    def as_lists(self) -> dict:
        return {"z": list(map(float, self.z)), "u": list(map(float, self.u)), "theta": list(map(float, self.theta))}

    @classmethod
    def from_lists(cls, store: dict) -> "WaveProfile":
        return cls(
            z=np.asarray(store["z"], dtype=float),
            u=np.asarray(store["u"], dtype=float),
            theta=np.asarray(store["theta"], dtype=float),
        )
