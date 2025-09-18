"""Simple GDS writer utilities for mock capacitors."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, Iterable, Tuple

import sys

import gdstk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drc.checker import load_pdk

LayerInfo = Tuple[int, int]


def _resolve_layer(
    layer_map: Dict[str, Dict[str, Any]], keys: Iterable[str]
) -> LayerInfo:
    for key in keys:
        entry = layer_map.get(key)
        if entry is not None:
            return int(entry.get("layer", 0)), int(entry.get("datatype", 0))
    preferred = ", ".join(keys)
    available = ", ".join(sorted(layer_map))
    raise KeyError(
        f"Could not resolve layer. Tried [{preferred}] in layer_map keys [{available}]"
    )




def make_capacitor_cell(
    pdk: Dict[str, Any],
    width_um: float,
    height_um: float,
    layer_name: str = "cap_top_plate",
    *,
    add_pads: bool = True,
) -> gdstk.Cell:
    """Create a toy capacitor layout composed of simple rectangles.

    Parameters
    ----------
    pdk: dict
        The mocked PDK dictionary; we only look at the GDS layer map entries.
    width_um, height_um: float
        Physical plate dimensions in micrometres.
    layer_name: str
        Preferred PDK key for the top plate layer.
    add_pads: bool
        When true, draw simple probing pads on the left/right edges.
    """

    layer_map = pdk.get("export", {}).get("gds", {}).get("layer_map", {})
    if not layer_map:
        raise ValueError("PDK does not define export.gds.layer_map")

    top_layer, top_datatype = _resolve_layer(
        layer_map, (layer_name, "cap_top_plate", "metal1")
    )
    bottom_layer, bottom_datatype = _resolve_layer(
        layer_map,
        (f"{layer_name}_bottom", "cap_bottom_plate", "metal2"),
    )
    via_layer, via_datatype = _resolve_layer(
        layer_map, (f"{layer_name}_via", "via12", "via")
    )
    try:
        pad_layer, pad_datatype = _resolve_layer(
            layer_map, ("pad", "metal3", "metal2")
        )
    except KeyError:
        pad_layer, pad_datatype = top_layer, top_datatype

    cell_name = f"CAP_{width_um:g}x{height_um:g}"
    cell = gdstk.Cell(cell_name)

    half_w = width_um / 2.0
    half_h = height_um / 2.0

    # Top and bottom plates share the same footprint.
    top_plate = gdstk.rectangle(
        (-half_w, -half_h), (half_w, half_h), layer=top_layer, datatype=top_datatype
    )
    bottom_plate = gdstk.rectangle(
        (-half_w, -half_h),
        (half_w, half_h),
        layer=bottom_layer,
        datatype=bottom_datatype,
    )
    cell.add(top_plate)
    cell.add(bottom_plate)

    if add_pads:
        pad_size = 20.0  # micrometres
        pad_offset = half_w + pad_size
        left_pad = gdstk.rectangle(
            (-pad_offset - pad_size, -pad_size / 2.0),
            (-pad_offset, pad_size / 2.0),
            layer=pad_layer,
            datatype=pad_datatype,
        )
        right_pad = gdstk.rectangle(
            (pad_offset, -pad_size / 2.0),
            (pad_offset + pad_size, pad_size / 2.0),
            layer=pad_layer,
            datatype=pad_datatype,
        )
        cell.add(left_pad)
        cell.add(right_pad)

    if width_um > 10.0 and height_um > 10.0:
        via_size = 1.0
        pitch = via_size * 3
        x = -half_w + pitch
        while x < half_w - pitch:
            y = -half_h + pitch
            while y < half_h - pitch:
                via_rect = gdstk.rectangle(
                    (x, y),
                    (x + via_size, y + via_size),
                    layer=via_layer,
                    datatype=via_datatype,
                )
                cell.add(via_rect)
                y += pitch
            x += pitch

    return cell

def export_gds(cell: gdstk.Cell, out_path: Path | str, dbu: float) -> str:
    """Write the provided cell to GDSII using the specified database unit.

    Returns the absolute path to the written file so callers can hand the
    result straight to UI components (for example, a Gradio download widget).
    """
    out_abs = os.path.abspath(str(out_path))
    precision_m = float(dbu) * 1e-6  # convert micrometers to meters
    lib = gdstk.Library(unit=1e-6, precision=precision_m)
    lib.add(cell)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    try:
        lib.write_gds(out_abs)
        print(f"GDS saved at {out_abs}")
    except OSError as exc:
        print(f"Failed to save GDS at {out_abs}: {exc}")
        raise
    return out_abs


if __name__ == "__main__":
    pdk_path = Path("pdk/mock_tsmc28rf.yaml")
    pdk = load_pdk(pdk_path)
    width_um = 10.0
    height_um = 20.0
    cell = make_capacitor_cell(pdk, width_um, height_um, layer_name="cap_top_plate")
    dbu = pdk.get("units", {}).get("dbu", 0.001)
    export_path = Path("out.gds")
    export_gds(cell, export_path, dbu)
    print(f"Wrote capacitor cell {cell.name} to {export_path} with dbu={dbu} um")
