# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Shared helpers for the Core ML pipelines that run Sova's bundled classifiers on a Mac."""

import math
import os
from pathlib import Path
from typing import Any

from argmaxtools.utils import get_logger


logger = get_logger(__name__)

# The Sova checkout whose `app/Sova/Resources/BundledModels` holds the staged mlmodelc bundles
SOVA_ROOT_ENV = "SOVA_ROOT"
DEFAULT_SOVA_ROOT = Path.home() / "Desktop" / "Projects" / "sova"
COMPUTE_UNITS = {"all": "ALL", "cpu_and_ne": "CPU_AND_NE", "cpu_and_gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY"}


def sova_bundle(relative: str) -> Path:
    """Path of a bundled model inside the Sova checkout named by SOVA_ROOT."""
    root = Path(os.getenv(SOVA_ROOT_ENV, DEFAULT_SOVA_ROOT))
    return root / "app" / "Sova" / "Resources" / "BundledModels" / relative


def load_compiled_model(path: str | Path, compute_units: str) -> Any:
    """Load a compiled Core ML model (`.mlmodelc`) or a package (`.mlpackage`)."""
    import coremltools as ct

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No Core ML model at {path}. Stage Sova's bundles (`make bundle-models` in the Sova repo) "
            f"or set {SOVA_ROOT_ENV} / the pipeline's model path."
        )
    units = getattr(ct.ComputeUnit, COMPUTE_UNITS[compute_units])
    logger.info(f"Loading {path.name} on {compute_units}")
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(path), compute_units=units)
    return ct.models.MLModel(str(path), compute_units=units)


def threshold_shifted_probability(probability: float, threshold: float) -> float:
    """Map a sigmoid head so that its own threshold lands on 0.5.

    Heads with different thresholds (0.60 for asking, 0.10 for off-platform)
    become comparable, so the maximum over heads is a single unsafe score whose
    0.5 crossing matches the per-head decisions.
    """
    p = min(max(float(probability), 1e-7), 1 - 1e-7)
    t = min(max(float(threshold), 1e-7), 1 - 1e-7)
    logit = math.log(p / (1 - p)) - math.log(t / (1 - t))
    return 1 / (1 + math.exp(-logit))
