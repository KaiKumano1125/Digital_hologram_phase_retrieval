"""
Typed configuration loader.

Loads config/config.yaml into a HoloConfig dataclass, providing
named-field access and basic type coercion for all parameters.
"""

from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class HoloConfig:
    """All parameters for simulation and reconstruction."""

    # Optical
    wavelength: float
    z1: float
    z2: float
    dx: float
    dy: float

    # Reference wave
    reference_type: str   # 'spherical' | 'two_point'
    offset: int

    # Propagation
    mode: str             # 'ASM' | 'fresnel'

    # Simulation
    pad_factor: int
    amp_path: str
    phase_path: str

    # Reconstruction
    max_iter: int
    tv_weight: float
    ref_weight: float
    noise_std: float
    learning_rate: float
    log_interval: int

    target_intensity_path: str
    ref_intensity_path: str
    outdir: str
    log_dir_prefix: str


def load_config(path: str = "config/config.yaml") -> HoloConfig:
    """Load and validate a YAML config file into a HoloConfig instance.

    Args:
        path: Path to the YAML config file.

    Returns:
        Populated HoloConfig dataclass.

    Raises:
        KeyError: If a required field is missing from the YAML.
        FileNotFoundError: If the config file does not exist.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return HoloConfig(
        wavelength=float(raw["wavelength"]),
        z1=float(raw["z1"]),
        z2=float(raw["z2"]),
        dx=float(raw["dx"]),
        dy=float(raw["dy"]),
        reference_type=str(raw["reference_type"]),
        offset=int(raw["offset"]),
        mode=str(raw["mode"]),
        pad_factor=int(raw["pad_factor"]),
        amp_path=str(raw["amp_path"]),
        phase_path=str(raw["phase_path"]),
        max_iter=int(raw["max_iter"]),
        tv_weight=float(raw["tv_weight"]),
        ref_weight=float(raw["ref_weight"]),
        noise_std=float(raw["noise_std"]),
        learning_rate=float(raw["learning_rate"]),
        log_interval=int(raw["log_interval"]),
        target_intensity_path=str(raw["target_intensity_path"]),
        ref_intensity_path=str(raw["ref_intensity_path"]),
        outdir=str(raw["outdir"]),
        log_dir_prefix=str(raw["log_dir_prefix"]),
    )
