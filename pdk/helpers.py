"""Helper utilities for working with mock PDK YAML descriptions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import yaml

# --- PDK Constants and Loaders ---
PDK_DIR = Path(__file__).parent

def get_available_pdks() -> List[str]:
    """Returns a list of available PDK names from the pdk directory."""
    return [p.stem for p in PDK_DIR.glob("*.yaml")]

def get_pdk_spec(name: str) -> Dict[str, Any]:
    """Loads a specific PDK spec and adds its path to the dictionary."""
    if not name.endswith(".yaml"):
        name += ".yaml"
    
    pdk_path = PDK_DIR / name
    if not pdk_path.exists():
        raise FileNotFoundError(f"PDK file not found: {pdk_path}")
        
    spec = load_pdk_yaml(pdk_path)
    spec['path'] = str(pdk_path) # Add path for easy access
    return spec

def load_pdk_yaml(path: Path | str) -> Dict[str, Any]:
    """Load a PDK definition stored as YAML."""

    pdk_path = Path(path)
    with pdk_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"PDK file {pdk_path} did not decode into a dictionary")
    return data


def get_px_to_um(pdk: Dict[str, Any]) -> float:
    """Return the pixel-to-micrometre scale factor defined by the PDK."""

    units = pdk.get("units", {})
    if "um_per_pixel" in units:
        return float(units["um_per_pixel"])
    if "unit_um_per_px" in units:
        return float(units["unit_um_per_px"])
    raise KeyError("PDK.units must define either 'um_per_pixel' or 'unit_um_per_px'")


def limit_with_pdk(
    pdk: Dict[str, Any],
    width_px: float,
    height_px: float,
) -> Tuple[float, float]:
    """Clamp the width/height (in pixels) to the PDK's min/max rules."""

    scale = get_px_to_um(pdk)
    drc = pdk.get("drc", {})

    def clamp(value_px: float, min_key: str, max_key: str) -> float:
        value_um = value_px * scale
        min_um = drc.get(min_key)
        max_um = drc.get(max_key)
        if min_um is not None and value_um < float(min_um):
            value_um = float(min_um)
        if max_um is not None and value_um > float(max_um):
            value_um = float(max_um)
        return value_um / scale

    clamped_width_px = clamp(width_px, "min_width_um", "max_width_um")
    clamped_height_px = clamp(height_px, "min_height_um", "max_height_um")
    return clamped_width_px, clamped_height_px
