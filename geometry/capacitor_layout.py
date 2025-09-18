"""Simple GDS writer utilities for mock capacitors."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, Iterable, Tuple

import sys

import gdstk
import numpy as np
import torch
from PIL import Image # For rendering, if gdstk's built-in rendering is not sufficient or we need more control

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drc.checker import load_pdk

LayerInfo = Tuple[int, int, str]


def _resolve_layer(
    layer_map: Dict[str, Dict[str, Any]], keys: Iterable[str]
) -> LayerInfo:
    """Return the first matching (layer, datatype, key) tuple for the supplied keys."""

    for key in keys:
        entry = layer_map.get(key)
        if entry is not None:
            return int(entry.get("layer", 0)), int(entry.get("datatype", 0)), key
    preferred = ", ".join(keys)
    available = ", ".join(sorted(layer_map))
    raise KeyError(
        f"Could not resolve layer. Tried [{preferred}] in layer_map keys [{available}]"
    )


def make_capacitor_cell(
    pdk: Dict[str, Any],
    cap_type: str = "rect",
    # Params for rect
    width_um: float = 10.0,
    height_um: float = 10.0,
    # Params for IDC
    num_fingers: int = 4,
    finger_length_um: float = 20.0,
    finger_width_um: float = 2.0,
    finger_spacing_um: float = 2.0,
    port_width_um: float = 5.0,
    # Common params
    layer_name: str = "cap_top_plate",
    *,
    add_pads: bool = True,
    guard_ring: bool = False,
    ring_width_um: float = 2.0,
    ring_spacing_um: float = 1.0,
    via_pitch_um: float | None = None,
    via_enclosure_um: float | None = None,
    pad_size_um: float | tuple[float, float] | None = None,
    metal_top: int | None = None,
    metal_bot: int | None = None,
) -> gdstk.Cell:
    """Create a capacitor layout. Supports 'rect' and 'idc' types."""

    layer_map = pdk.get("export", {}).get("gds", {}).get("layer_map", {})
    if not layer_map:
        raise ValueError("PDK does not define export.gds.layer_map")

    top_layer, top_datatype, top_key = _resolve_layer(
        layer_map, (layer_name, "cap_top_plate", "metal1")
    )
    bottom_layer, bottom_datatype, bottom_key = _resolve_layer(
        layer_map,
        (f"{layer_name}_bottom", "cap_bottom_plate", "metal2"),
    )
    via_layer, via_datatype, via_key = _resolve_layer(
        layer_map, (f"{layer_name}_via", "via12", "via")
    )
    try:
        pad_layer, pad_datatype, _ = _resolve_layer(
            layer_map, ("pad", "metal3", "metal2")
        )
    except KeyError:
        pad_layer, pad_datatype = top_layer, top_datatype

    if metal_top is not None:
        top_layer = int(metal_top)
    if metal_bot is not None:
        bottom_layer = int(metal_bot)

    if cap_type == "idc":
        # --- Interdigitated Capacitor (IDC) Logic ---
        cell_name = f"IDC_{num_fingers}f_{finger_length_um:.1f}l"
        cell = gdstk.Cell(cell_name)

        # Total height is determined by the number of fingers and their geometry
        total_height = (num_fingers * finger_width_um) + ((num_fingers - 1) * finger_spacing_um)
        
        # Create left and right ports (bus bars)
        left_port_x2 = -finger_length_um / 2
        left_port_x1 = left_port_x2 - port_width_um
        right_port_x1 = finger_length_um / 2
        right_port_x2 = right_port_x1 + port_width_um

        left_port = gdstk.rectangle((left_port_x1, -total_height/2), (left_port_x2, total_height/2), layer=top_layer, datatype=top_datatype)
        right_port = gdstk.rectangle((right_port_x1, -total_height/2), (right_port_x2, total_height/2), layer=bottom_layer, datatype=bottom_datatype)
        cell.add(left_port, right_port)

        # Create fingers
        for i in range(num_fingers):
            y1 = -total_height/2 + i * (finger_width_um + finger_spacing_um)
            y2 = y1 + finger_width_um
            
            if i % 2 == 0:  # Finger from left port (top layer)
                finger = gdstk.rectangle((left_port_x2, y1), (right_port_x1, y2), layer=top_layer, datatype=top_datatype)
            else:  # Finger from right port (bottom layer)
                finger = gdstk.rectangle((left_port_x2, y1), (right_port_x1, y2), layer=bottom_layer, datatype=bottom_datatype)
            cell.add(finger)
        
        # For guard ring and pads, we need a bounding box
        width_um = right_port_x2 - left_port_x1
        height_um = total_height

    elif cap_type == "rect":
        # --- Original Rectangle Logic ---
        cell_name = f"CAP_{width_um:g}x{height_um:g}"
        cell = gdstk.Cell(cell_name)

        half_w = width_um / 2.0
        half_h = height_um / 2.0

        top_plate = gdstk.rectangle(
            (-half_w, -half_h), (half_w, half_h), layer=top_layer, datatype=top_datatype
        )
        bottom_plate = gdstk.rectangle(
            (-half_w, -half_h),
            (half_w, half_h),
            layer=bottom_layer,
            datatype=bottom_datatype,
        )
        cell.add(top_plate, bottom_plate)
    else:
        raise ValueError(f"Unknown cap_type: '{cap_type}'. Must be 'rect' or 'idc'.")


    if add_pads:
        if pad_size_um is None:
            pad_w = pad_h = 20.0
        elif isinstance(pad_size_um, tuple):
            pad_w, pad_h = pad_size_um
        else:
            pad_w = pad_h = float(pad_size_um)
        
        # Note: pad offset logic might need adjustment for IDC
        pad_offset_x = (width_um / 2.0) + pad_w
        left_pad = gdstk.rectangle(
            (-pad_offset_x - pad_w, -pad_h / 2.0),
            (-pad_offset_x, pad_h / 2.0),
            layer=pad_layer,
            datatype=pad_datatype,
        )
        right_pad = gdstk.rectangle(
            (pad_offset_x, -pad_h / 2.0),
            (pad_offset_x + pad_w, pad_h / 2.0),
            layer=pad_layer,
            datatype=pad_datatype,
        )
        cell.add(left_pad, right_pad)

    if guard_ring:
        ring_layer, ring_datatype, _ = _resolve_layer(
            layer_map, ("guard", top_key, "metal1")
        )
        outer_w = width_um + 2.0 * (ring_spacing_um + ring_width_um)
        outer_h = height_um + 2.0 * (ring_spacing_um + ring_width_um)
        inner_w = width_um + 2.0 * ring_spacing_um
        inner_h = height_um + 2.0 * ring_spacing_um
        outer = gdstk.rectangle(
            (-outer_w / 2.0, -outer_h / 2.0),
            (outer_w / 2.0, outer_h / 2.0),
            layer=ring_layer,
            datatype=ring_datatype,
        )
        inner = gdstk.rectangle(
            (-inner_w / 2.0, -inner_h / 2.0),
            (inner_w / 2.0, inner_h / 2.0),
            layer=ring_layer,
            datatype=ring_datatype,
        )
        ring_polys = gdstk.boolean(
            outer, inner, "not", layer=ring_layer, datatype=ring_datatype
        )
        for polygon in ring_polys:
            cell.add(polygon)

    # Note: via logic is not adapted for IDC yet
    if via_pitch_um is not None and cap_type == 'rect' and width_um > 2.0 * via_pitch_um and height_um > 2.0 * via_pitch_um:
        via_meta = pdk.get("layers", {}).get(via_key, {})
        via_size = float(via_meta.get("min_diameter_um", 1.0))
        enclosure = float(via_enclosure_um or 0.0)
        pitch = max(via_pitch_um, via_size + 2 * enclosure)
        x = -width_um / 2.0 + pitch
        while x < width_um / 2.0 - pitch:
            y = -height_um / 2.0 + pitch
            while y < height_um / 2.0 - pitch:
                via_rect = gdstk.rectangle(
                    (x - enclosure, y - enclosure),
                    (x + via_size + enclosure, y + via_size + enclosure),
                    layer=via_layer,
                    datatype=via_datatype,
                )
                cell.add(via_rect)
                y += pitch
            x += pitch

    return cell


def render_cell_to_image(
    cell: gdstk.Cell,
    image_size: int = 64,
    in_channels: int = 1, # For simplicity, start with 1 channel (binary image)
    layer_to_render: int | None = None, # Specific layer to render, if None, render all metal layers
    pdk: Dict[str, Any] | None = None, # Optional: for layer info
) -> torch.Tensor:
    """
    Renders a gdstk.Cell into a 2D image (PyTorch tensor).

    Args:
        cell: The gdstk.Cell object to render.
        image_size: The desired size of the output image (e.g., 64 for 64x64 pixels).
        in_channels: The number of channels in the output image. Currently supports 1 (binary).
        layer_to_render: If specified, only polygons on this GDS layer will be rendered.
        pdk: Optional PDK dictionary to get layer information if needed.

    Returns:
        A torch.Tensor representing the 2D image of the cell layout,
        with shape (in_channels, image_size, image_size).
    """
    if in_channels != 1:
        raise NotImplementedError("Currently only 1 input channel (binary image) is supported for CNNEmulator.")

    # Get bounding box of the cell
    bbox = cell.bounding_box()
    if bbox is None:
        # Handle empty cell case
        return torch.zeros((in_channels, image_size, image_size), dtype=torch.float32)

    min_x, min_y = bbox.min
    max_x, max_y = bbox.max

    # Calculate scaling factor and offset to fit into image_size
    layout_width = max_x - min_x
    layout_height = max_y - min_y

    # Add a small margin to avoid clipping at edges
    margin_factor = 1.1
    effective_width = layout_width * margin_factor
    effective_height = layout_height * margin_factor

    scale_x = image_size / effective_width if effective_width > 0 else 0
    scale_y = image_size / effective_height if effective_height > 0 else 0

    # Use the smaller scale to ensure the entire layout fits
    scale = min(scale_x, scale_y) if min(scale_x, scale_y) > 0 else 1.0

    # Calculate offset to center the layout
    offset_x = -min_x * scale + (image_size - layout_width * scale) / 2.0
    offset_y = -min_y * scale + (image_size - layout_height * scale) / 2.0

    # Create a blank image array
    image_array = np.zeros((image_size, image_size), dtype=np.float32)

    # Iterate through polygons and "draw" them onto the image array
    for polygon in cell.polygons:
        if layer_to_render is not None and polygon.layer != layer_to_render:
            continue

        # Transform polygon points to pixel coordinates
        pixel_points = []
        for point in polygon.points:
            px = int((point[0] * scale + offset_x))
            py = int((point[1] * scale + offset_y))
            pixel_points.append((px, py))
        
        # Fill the polygon in the image array
        # This is a simplified fill. For accurate rendering, a proper rasterization library might be needed.
        # For now, we'll just mark the bounding box of the polygon.
        # A more robust solution would involve using PIL.ImageDraw.polygon or similar.
        
        # Simple bounding box fill for demonstration
        if pixel_points:
            min_px = max(0, min(p[0] for p in pixel_points))
            max_px = min(image_size - 1, max(p[0] for p in pixel_points))
            min_py = max(0, min(p[1] for p in pixel_points))
            max_py = min(image_size - 1, max(p[1] for p in pixel_points))
            image_array[min_py:max_py+1, min_px:max_px+1] = 1.0 # Mark as metal

    # Convert to PyTorch tensor and add channel dimension
    image_tensor = torch.from_numpy(image_array).unsqueeze(0) # Add channel dimension
    return image_tensor


def export_gds(
    width_um: float,
    height_um: float,
    output_path: Path | str,
    cell_name: str = "RF_CAP",
    **kwargs
) -> Path:
    """Export a capacitor design to a GDS-II file.
    
    Args:
        width_um: Width of the capacitor plates in micrometers.
        height_um: Height of the capacitor plates in micrometers.
        output_path: Path where the GDS-II file will be written.
        cell_name: Name for the top-level cell.
        **kwargs: Additional arguments passed to make_capacitor_cell().
        
    Returns:
        Path to the created GDS-II file.python main.py gradio
    """
    # Load default PDK if none provided
    pdk_path = Path("pdk/mock_tsmc28rf.yaml")
    pdk = load_pdk(pdk_path)
    
    # Create the layout
    cell = make_capacitor_cell(pdk, width_um, height_um, layer_name="cap_top_plate", **kwargs)
    
    # Create GDS library with PDK settings
    dbu = pdk.get("units", {}).get("dbu", 0.001)
    out_abs = os.path.abspath(str(output_path))
    precision_m = float(dbu) * 1e-6  # convert micrometres to meters
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
    lib = gdstk.Library(unit=1e-6, precision=dbu * 1e-6)
    lib.add(cell)
    lib.write_gds(export_path)
    print(f"Wrote capacitor cell {cell.name} to {export_path} with dbu={dbu} um")