# syntax=docker/dockerfile:1.6
#
# Hugging Face Space Docker build for Mountain Waves.
#
# Two-stage build: the first stage has the Rust toolchain and produces a
# manylinux-ish wheel of the compiled extension; the second stage is a slim
# Python runtime that just installs the wheel + deps. This keeps the final
# image under ~400 MB without needing cargo at runtime.
#
# HF Spaces routes external traffic to whatever port the container listens
# on via `app_port` in README.md — we use 7860 (the HF default).

# ---------- build stage: compile the Rust extension into a wheel ----------
# rayon 1.12 needs rustc ≥ 1.80; pin a modern stable image so the MSRV of the
# transitive crate graph (rayon, pyo3, numpy) stays satisfied.
FROM rust:1.82-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages "maturin==1.7.*"

WORKDIR /src
COPY Cargo.toml Cargo.lock pyproject.toml README.md ./
COPY src ./src
COPY python ./python

# Build a release wheel. -i python3 pins against the build image's 3.11
# interpreter; the runtime stage uses the same major.minor so the abi3
# wheel loads cleanly.
RUN maturin build --release --out /wheels -i python3


# ---------- runtime stage: Python 3.11 slim + app + wheel ----------
FROM python:3.11-slim

# libgomp1 is the OpenMP runtime rayon links against; eccodes pulls its own
# shared libs in via the eccodeslib wheel, so nothing else is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the compiled Rust wheel + all runtime deps. Deps are listed
# explicitly here instead of using `pip install .` so we don't drag maturin
# and the build system into the runtime image.
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
 && pip install --no-cache-dir \
        "numpy>=1.23" \
        "scipy>=1.10" \
        "dash>=2.16" \
        "plotly>=5.20" \
        "boto3>=1.28" \
        "xarray>=2023.1" \
        "cfgrib>=0.9.10" \
        "eccodes>=2.37" \
        "eccodeslib>=2.46"

# HF Spaces runs the container as uid 1000 with /tmp as the only writable
# directory by default. Cache dirs that libraries write to (plotly, boto3,
# xdg) need to live somewhere the process can actually create files.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mplconfig

# Note: we intentionally do NOT copy python/ into the runtime image. The wheel
# built in the previous stage contains both the pure-Python source and the
# compiled _core*.so, installed into site-packages. If we also copied the
# source tree here, run.py's sys.path.insert(0, "./python") would shadow the
# wheel's copy with a _core-less source tree, and `from . import _core` would
# fail with "partially initialized module" (HF deploy 2026-04-18).
COPY run.py ./run.py

EXPOSE 7860
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860"]
