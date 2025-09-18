# geometry/layout_manager.py
from __future__ import annotations
import numpy as np
import gdstk
from typing import Dict, Tuple, List

def params_to_binary_mask(
    params: Dict[str, float],
    canvas_um: Tuple[float, float] = (200.0, 200.0),
    grid: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """IDC 파라미터를 2D binary mask로 변환."""
    nrow, ncol = grid
    mask = np.zeros((nrow, ncol), dtype=np.uint8)

    # 간단한 IDC 생성 규칙 (Seed): 가운데를 기준으로 좌/우 comb fingers
    nf = int(params["num_fingers"])
    L  = float(params["finger_length_um"])
    W  = float(params["finger_width_um"])
    S  = float(params["finger_spacing_um"])

    # um→grid scale
    w_um, h_um = canvas_um
    sx, sy = ncol / w_um, nrow / h_um

    # finger pitch와 y범위
    pitch = W + S
    total = nf * W + (nf - 1) * S
    y0 = (h_um - total) * 0.5
    # x 범위
    x0 = (w_um - L) * 0.5
    x1 = x0 + L

    # 좌/우 interdigitate: 짝수/홀수 손가락을 좌/우로 분배
    for k in range(nf):
        y_start = y0 + k * pitch
        # 좌측 bus쪽 절반
        if k % 2 == 0:
            xa, xb = x0, (x0 + x1) * 0.5
        else:
            xa, xb = (x0 + x1) * 0.5, x1
        ya, yb = y_start, y_start + W

        # grid 사각형 채우기
        c0, c1 = int(xa * sx), int(xb * sx)
        r0, r1 = int(ya * sy), int(yb * sy)
        mask[max(0,r0):min(nrow,r1), max(0,c0):min(ncol,c1)] = 1

    return mask

def binary_mask_to_gds(mask: np.ndarray, layer: int = 1, cell_name: str = "IDC") -> gdstk.Cell:
    """Binary mask를 GDS Cell로 변환(픽셀을 사각형으로)."""
    nrow, ncol = mask.shape
    cell = gdstk.Cell(cell_name)
    # 픽셀 사이즈 1um로 가정 → 필요 시 스케일링
    for r in range(nrow):
        for c in range(ncol):
            if mask[r, c]:
                # (c, r) 픽셀을 1x1 um 사각형으로
                rect = gdstk.rectangle((c, r), (c+1, r+1), layer=layer)
                cell.add(rect)
    return cell

def binary_mask_to_rects_um(mask: np.ndarray, pixel_um: float = 1.0) -> List[Tuple[float,float,float,float]]:
    """HFSS에서 그릴 수 있도록 (x0,y0,x1,y1)[um] 리스트로 변환."""
    rects = []
    nrow, ncol = mask.shape
    for r in range(nrow):
        for c in range(ncol):
            if mask[r, c]:
                x0, y0 = c * pixel_um, r * pixel_um
                x1, y1 = (c+1) * pixel_um, (r+1) * pixel_um
                rects.append((x0, y0, x1, y1))
    return rects
