
"""Simple DRC helpers for the mock, YAML-described PDKs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


from dataclasses import dataclass


@dataclass
class DRCViolation:
    """Container for a DRC violation."""
    rule: str
    message: str
    severity: str = 'error'


@dataclass
class DRCReport:
    """Container for DRC check results."""
    passed: bool
    violations: List[DRCViolation]


def load_pdk(yaml_path: Path | str) -> Dict[str, Any]:
    """Load a YAML PDK description and return it as a dictionary."""

    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"PDK file {yaml_path} did not decode to a dictionary")
    return data


def px_to_um(value_px: float, pdk: Dict[str, Any]) -> float:
    """Convert a value expressed in pixels to micrometres using the PDK scale."""

    try:
        um_per_pixel = float(pdk["units"]["um_per_pixel"])
    except KeyError as exc:
        raise KeyError("PDK is missing units.um_per_pixel") from exc
    return float(value_px) * um_per_pixel


def check_rect_drc(
    width_um: float, height_um: float, pdk: Dict[str, Any],
    *,
    has_vias: bool = False,
    via_pitch_um: float | None = None,
    via_enclosure_um: float | None = None,
) -> List[DRCViolation]:
    """Validate a rectangular feature against PDK rules."""

    drc = pdk.get("drc", {})
    min_width = drc.get("min_width_um")
    max_width = drc.get("max_width_um", float('inf'))
    violations: List[DRCViolation] = []
    
    # Check width limits
    if min_width is not None:
        if width_um < min_width:
            violations.append(DRCViolation(
                'MIN_WIDTH',
                f"width {width_um:.3f}µm below minimum {min_width:.3f}µm"
            ))
        if height_um < min_width:
            violations.append(DRCViolation(
                'MIN_HEIGHT',
                f"height {height_um:.3f}µm below minimum {min_width:.3f}µm"
            ))
    if width_um > max_width:
        violations.append(DRCViolation(
            'MAX_WIDTH',
            f"width {width_um:.3f}µm exceeds maximum {max_width:.3f}µm"
        ))
    if height_um > max_width:
        violations.append(DRCViolation(
            'MAX_HEIGHT',
            f"height {height_um:.3f}µm exceeds maximum {max_width:.3f}µm"
        ))
    
    # Check via rules if enabled
    if has_vias and via_pitch_um is not None and via_enclosure_um is not None:
        min_via_area = 2 * via_enclosure_um * 2 * via_enclosure_um
        plate_area = width_um * height_um
        if plate_area < min_via_area:
            violations.append(DRCViolation(
                'MIN_VIA_AREA',
                f"plate area {plate_area:.3f}µm² too small for via with "
                f"{via_enclosure_um:.3f}µm enclosure (min {min_via_area:.3f}µm²)"
            ))
        if via_pitch_um < 2 * via_enclosure_um:
            violations.append(DRCViolation(
                'MIN_VIA_PITCH',
                f"via pitch {via_pitch_um:.3f}µm smaller than minimum "
                f"{2 * via_enclosure_um:.3f}µm (2x enclosure)"
            ))
        min_width_for_via = 2 * via_enclosure_um + via_pitch_um
        if width_um < min_width_for_via or height_um < min_width_for_via:
            violations.append(DRCViolation(
                'VIA_ENCLOSURE',
                f"plate dimensions ({width_um:.3f}µm x {height_um:.3f}µm) "
                f"too small for via array with {via_enclosure_um:.3f}µm enclosure "
                f"and {via_pitch_um:.3f}µm pitch"
            ))
    
    return violations


def check_min_max(
    width_um: float, height_um: float, pdk: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check simple min/max constraints defined in the dummy PDK YAML."""

    drc = pdk.get("drc", {})
    min_width = drc.get("min_width_um")
    max_width = drc.get("max_width_um")
    min_height = drc.get("min_height_um")
    max_height = drc.get("max_height_um")
    messages = []
    if min_width is not None and width_um < min_width:
        messages.append(f"width {width_um:.3f}um < min {min_width:.3f}um")
    if max_width is not None and width_um > max_width:
        messages.append(f"width {width_um:.3f}um > max {max_width:.3f}um")
    if min_height is not None and height_um < min_height:
        messages.append(f"height {height_um:.3f}um < min {min_height:.3f}um")
    if max_height is not None and height_um > max_height:
        messages.append(f"height {height_um:.3f}um > max {max_height:.3f}um")
    if messages:
        return False, ", ".join(messages)
    return True, "pass"


def snap_to_grid(x_um: float, y_um: float, grid: float) -> Tuple[float, float]:
    """Snap the provided coordinates to the nearest manufacturing grid."""

    if grid <= 0:
        raise ValueError("Grid step must be positive")

    def snap(value: float) -> float:
        return round(value / grid) * grid

    return snap(float(x_um)), snap(float(y_um))


def aggregate_report(rect: Dict[str, float], pdk: Dict[str, Any]) -> DRCReport:
    """Run the available rectangular DRC checks and collect reasons."""

    width_um = float(rect.get("width_um", 0.0))
    height_um = float(rect.get("height_um", 0.0))
    
    violations = check_rect_drc(width_um, height_um, pdk)
    
    passed, message = check_min_max(width_um, height_um, pdk)
    if not passed:
        violations.append(DRCViolation(rule="MIN_MAX", message=message))

    return DRCReport(passed=not violations, violations=violations)


if __name__ == "__main__":
    pdk_path = Path("pdk/mock_tsmc28rf.yaml")
    pdk = load_pdk(pdk_path)
    width_um = 0.2
    height_um = 0.6
    report = aggregate_report({"width_um": width_um, "height_um": height_um}, pdk)
    print(f"PDK: {pdk_path.name}")
    print(f"Check rectangle {width_um}um x {height_um}um -> {'PASS' if report.passed else 'FAIL'}")
    for v in report.violations:
        print(f" - {v.rule}: {v.message}")
