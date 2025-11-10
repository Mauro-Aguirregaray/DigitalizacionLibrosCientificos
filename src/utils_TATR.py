# transforms_tatr.py (o el nombre que uses para tu módulo)

from __future__ import annotations
from dataclasses import dataclass
from PIL import Image


import os

import torch
from torch import Tensor
from typing import List, Dict, Any, Tuple
from typing import Callable, Dict, List, Sequence, Tuple, Any, Set, Optional
import logging
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

LOGGER = logging.getLogger(__name__)


try:
    # Import liviano; evita fallar si el host aún no tiene torch/torchvision
    import torch
    from torch import Tensor
    from torchvision import transforms
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = None  # type: ignore
    transforms = None  # type: ignore


@dataclass
class MaxResize:
    """
    Redimensiona una imagen manteniendo aspecto para que el lado mayor sea `max_size`.
    A diferencia de la versión "segura", esta implementación **también hace upscaling**
    si la imagen es más chica que `max_size`.

    Args:
        max_size (int): Tamaño máximo del lado mayor (la imagen resultante siempre tendrá
            su lado mayor igual a este valor).
        resample (int): Filtro de remuestreo de PIL (por defecto Image.BILINEAR).

    Returns:
        Image.Image: Imagen redimensionada con el nuevo tamaño.

    Raises:
        TypeError: Si la entrada no es una instancia de PIL.Image.Image.
        ValueError: Si la imagen tiene dimensiones inválidas (<= 0).

    Examples:
        >>> img = Image.open("ejemplo.jpg")
        >>> transform = MaxResize(max_size=800)
        >>> out = transform(img)
        >>> out.size
        (800, 533)  # si la original era 1200x800
    """
    max_size: int = 800
    resample: int = Image.BILINEAR

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Se esperaba PIL.Image.Image, recibido: {type(image)}")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Tamaño de imagen inválido: {image.size}")

        current_max = max(width, height)
        scale = self.max_size / float(current_max)
        new_w = max(1, int(round(scale * width)))
        new_h = max(1, int(round(scale * height)))

        return image.resize((new_w, new_h), resample=self.resample)


def make_structure_transform(
    max_size: int = 1000,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
):
    """
    Construye un pipeline `transforms.Compose` para preparar imágenes de estructura/tablas.

    El pipeline incluye:
    - MaxResize: asegura que el lado mayor quede exactamente en `max_size`.
    - ToTensor: convierte a tensor (C,H,W) en [0,1].
    - Normalize: normaliza con `mean` y `std` (por defecto, ImageNet).

    Args:
        max_size (int): Tamaño máximo del lado mayor tras redimensionar.
        mean (Sequence[float]): Medias para normalización.
        std (Sequence[float]): Desvíos estándar para normalización.

    Returns:
        transforms.Compose: Transformación compuesta lista para usar.

    Raises:
        RuntimeError: Si torchvision no está disponible en el entorno.
    """
    if transforms is None:
        raise RuntimeError("torchvision no está disponible en el entorno.")

    return transforms.Compose([
        MaxResize(max_size=max_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def to_model_batch(
    image: Image.Image,
    transform=None,
) -> "Tensor":
    """
    Aplica el `transform` y agrega dimensión batch (1, C, H, W).

    Parameters
    ----------
    image : PIL.Image.Image
    transform : callable | None
        Si es None, usa `make_structure_transform()` con valores por defecto.

    Returns
    -------
    torch.Tensor
        Tensor float32 normalizado con shape (1, C, H, W).
    """
    if torch is None:
        raise RuntimeError("PyTorch no está disponible en el entorno.")

    if transform is None:
        transform = make_structure_transform()

    tensor = transform(image)  # (C, H, W)
    if tensor.ndim != 3:
        raise ValueError(f"Transform debe producir tensor 3D (C,H,W). Obtuve: {tensor.shape}")

    return tensor.unsqueeze(0)  # (1, C, H, W)

def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """Convert [cx, cy, w, h] boxes to [x0, y0, x1, y1].

    Args:
        x: Tensor of shape (..., 4) with center-x, center-y, width, height.

    Returns:
        Tensor of shape (..., 4) with x0, y0, x1, y1 coordinates.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox: Tensor, size: Tuple[int, int]) -> Tensor:
    """Rescale normalized [0–1] boxes to absolute pixel coords.

    Args:
        out_bbox: Tensor of shape (N, 4) with normalized [x_c, y_c, w, h].
        size: (width, height) of the original image.

    Returns:
        Tensor of shape (N, 4) with [x0, y0, x1, y1] in pixels.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b * scale


def outputs_to_objects(
    outputs: Any,
    img_size: Tuple[int, int],
    id2label: Dict[int, str],
    drop_no_object: bool = True,
) -> List[Dict[str, Any]]:
    """Convierte outputs crudos del modelo a objetos estructurados.

    Args:
        outputs: Debe exponer `logits` y `pred_boxes` (DETR/TATR-like).
        img_size: (width, height) de la imagen original.
        id2label: mapa índice→etiqueta.
        drop_no_object: si True, descarta la clase 'no object'/'no_object'.

    Returns:
        Lista de dicts con: 'label', 'score', 'bbox' (x0,y0,x1,y1) en px.
    """
    m = outputs.logits.softmax(-1).max(-1)  # (batch, queries)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]

    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects: List[Dict[str, Any]] = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]

        if drop_no_object:
            # normalizo underscore/espacios por si el config trae 'no_object'
            norm = class_label.lower().replace("_", " ").strip()
            if norm == "no object":
                continue

        objects.append(
            {"label": class_label, "score": float(score), "bbox": [float(v) for v in bbox]}
        )
    return objects


# Colores BGR por tipo de objeto (visualización)
COLORS = {
    "table": (0, 255, 255),                    # amarillo
    "table row": (0, 200, 0),                  # verde
    "table column": (200, 100, 0),             # azul-ish
    "table spanning cell": (0, 0, 255),        # rojo
    "table column header": (255, 0, 255),      # magenta
    "table row header": (255, 165, 0),         # naranja
    "table projected row header": (128, 0, 128) # violeta
}

# Orden lógico para dibujar overlays
ORDER = [
    "table",
    "table column header",
    "table row header",
    "table projected row header",
    "table row",
    "table column",
    "table spanning cell",
]


def _draw_layer(
    img,
    detections: List[Dict],
    labels: List[str],
    alpha: float = 0.25,
    thickness: int = 2,
    put_labels: bool = True,
):
    """Dibuja solo los labels indicados sobre una copia de la imagen.

    Args:
        img: Imagen BGR (array de cv2).
        detections: Lista de detecciones con keys: 'label', 'bbox', 'score'.
        labels: Lista de labels a renderizar.
        alpha: Transparencia del relleno [0–1].
        thickness: Grosor del borde en píxeles.
        put_labels: Si True, dibuja etiquetas de texto.

    Returns:
        Imagen BGR con overlays.
    """
    base = img.copy()
    overlay = base.copy()

    dets = [d for d in detections if d.get("label") in labels]
    dets.sort(key=lambda d: ORDER.index(d["label"]) if d["label"] in ORDER else 999)

    for det in dets:
        label = det["label"]
        x1, y1, x2, y2 = map(lambda v: int(round(float(v))), det["bbox"])
        color = COLORS.get(label, (128, 128, 128))

        # Relleno semitransparente
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # Borde
        cv2.rectangle(base, (x1, y1), (x2, y2), color, thickness)

        if put_labels:
            score = det.get("score", None)
            txt = f"{label}" + (f" {score:.3f}" if isinstance(score, (float, int)) else "")
            (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pad = 3
            tx1, ty1 = x1, max(0, y1 - th - 2 * pad)
            tx2, ty2 = x1 + tw + 2 * pad, y1
            cv2.rectangle(base, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
            cv2.putText(
                base,
                txt,
                (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def draw_tatr_overlays_multi(
    image_path: str,
    detections: List[Dict],
    out_dir: str,
    alpha: float = 0.25,
    thickness: int = 2,
    put_labels: bool = True,
) -> Dict[str, str]:
    """Genera imágenes con overlays por tipo de objeto.

    Se crean 8 salidas:
      1. Solo 'table'
      2. Solo 'table column header'
      3. Solo 'table row header'
      4. Solo 'table projected row header'
      5. Solo 'table row'
      6. Solo 'table column'
      7. Solo 'table spanning cell'
      8. Todas las capas combinadas

    Args:
        image_path: Ruta de la imagen base.
        detections: Lista de detecciones [{'label','bbox','score'}, ...].
        out_dir: Directorio de salida para las imágenes.
        alpha: Transparencia del relleno [0–1].
        thickness: Grosor del borde.
        put_labels: Si True, dibuja etiquetas de texto.

    Returns:
        Dict con {nombre_capa: ruta_imagen}.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo abrir la imagen: {image_path}")

    layers = [
        ("table", ["table"]),
        ("table_column_header", ["table column header"]),
        ("table_row_header", ["table row header"]),
        ("table_projected_row_header", ["table projected row header"]),
        ("table_row", ["table row"]),
        ("table_column", ["table column"]),
        ("table_spanning_cell", ["table spanning cell"]),
        ("all", ORDER),
    ]

    saved: Dict[str, str] = {}
    for suffix, labels in layers:
        out = _draw_layer(img, detections, labels, alpha=alpha, thickness=thickness, put_labels=put_labels)
        out_path = os.path.join(out_dir, f"tatr_{suffix}.png")
        cv2.imwrite(out_path, out)
        saved[suffix] = out_path

    return saved


# Types
Box = Tuple[float, float, float, float]
LOGGER = logging.getLogger(__name__)


def _iou(a: Box, b: Box) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2].

    Args:
        a: Box as (x1, y1, x2, y2).
        b: Box as (x1, y1, x2, y2).

    Returns:
        Intersection-over-Union in [0, 1].

    Examples:
        >>> round(_iou((0,0,10,10), (5,5,15,15)), 3)
        0.143
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_boxes(boxes: Sequence[Box], iou_th: float = 0.6) -> List[Box]:
    """Deduplicate highly overlapping boxes using IoU threshold.

    Boxes are kept with a stable order (sorted by y1, then x1). A box is
    retained if its IoU with all previously kept boxes is < `iou_th`.

    Args:
        boxes: Sequence of boxes (x1, y1, x2, y2).
        iou_th: IoU threshold for considering duplicates (>= keeps only one).

    Returns:
        List of deduplicated boxes.

    Raises:
        ValueError: If any box has invalid coordinates (x2 < x1 or y2 < y1).

    Examples:
        >>> dedup_boxes([(0,0,10,10), (1,1,10,10)], iou_th=0.5)
        [(0, 0, 10, 10)]
    """
    # Basic validation (lightweight)
    for b in boxes:
        if b[2] < b[0] or b[3] < b[1]:
            raise ValueError(f"Invalid box coordinates: {b}")

    kept: List[Box] = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):  # by y1 then x1
        if all(_iou(box, k) < iou_th for k in kept):
            kept.append(box)
    LOGGER.debug("dedup_boxes: in=%d out=%d", len(boxes), len(kept))
    return kept


def covered_indices(
    bbox: Box,
    lines: Sequence[Box],
    axis: int = 0,
    overlap_th: float = 0.5,
) -> List[int]:
    """Return indices of lines (columns/rows) significantly overlapped by bbox.

    Overlap is computed along the target axis and normalized by the minimum
    extent between the span and the line on that axis. This is robust for
    partial spans.

    Args:
        bbox: Span box (x1, y1, x2, y2).
        lines: Sequence of line boxes (columns or rows).
        axis: 0 → columns (overlap on X), 1 → rows (overlap on Y).
        overlap_th: Minimum normalized overlap to mark as covered.

    Returns:
        Indices of covered lines.

    Raises:
        ValueError: If `axis` is not 0 or 1.

    Examples:
        >>> covered_indices((10,10,30,30), [(0,0,20,40),(25,0,45,40)], axis=0)
        [0, 1]
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    x1, y1, x2, y2 = bbox
    covered: List[int] = []
    for i, (lx1, ly1, lx2, ly2) in enumerate(lines):
        if axis == 0:
            inter = max(0.0, min(x2, lx2) - max(x1, lx1))
            den = max(1e-6, min(x2 - x1, lx2 - lx1))
        else:
            inter = max(0.0, min(y2, ly2) - max(y1, ly1))
            den = max(1e-6, min(y2 - y1, ly2 - ly1))
        ratio = inter / den
        if ratio >= overlap_th:
            covered.append(i)
    return covered


def count_cells_tatr(
    detections: Sequence[Dict[str, Any]],
    iou_th: float = 0.6,
    overlap_th: float = 0.5,
) -> Dict[str, Any]:
    """Count rows, columns, and real cells (discounting spanning cells).

    Rule:
        total = n_rows * n_cols
                - Σ( max(1, rows_covered) * max(1, cols_covered) - 1 )

    Rows/columns are first deduplicated with IoU, then sorted by y1/x1.
    Spans contribute a deduction equal to (area - 1).

    Args:
        detections: List of dicts with at least {'label','bbox'}:
            - 'table row'            → row lines
            - 'table column'         → column lines
            - 'table spanning cell'  → spanning cells
        iou_th: IoU threshold for row/column deduplication.
        overlap_th: Overlap threshold to consider a line covered by a span.

    Returns:
        Dict with:
            - 'rows': int, number of rows
            - 'cols': int, number of columns
            - 'cells': int, real cells (after span deductions)
            - 'rows_boxes': List[Box], deduplicated & sorted row boxes
            - 'cols_boxes': List[Box], deduplicated & sorted column boxes

    Raises:
        KeyError: If a detection is missing 'label' or 'bbox'.
        ValueError: If any row/column box is invalid.

    Examples:
        >>> dets = [
        ...   {'label':'table row','bbox':(0,0,100,10)},
        ...   {'label':'table row','bbox':(0,10,100,20)},
        ...   {'label':'table column','bbox':(0,0,50,20)},
        ...   {'label':'table column','bbox':(50,0,100,20)},
        ...   {'label':'table spanning cell','bbox':(0,0,50,20)},
        ... ]
        >>> count_cells_tatr(dets)['cells']
        3
    """
    # Basic field checks (fail fast, keeps errors clear during training)
    for d in detections:
        if "label" not in d or "bbox" not in d:
            raise KeyError(f"Detection missing 'label' or 'bbox': {d}")

    rows = [d["bbox"] for d in detections if d.get("label") == "table row"]
    cols = [d["bbox"] for d in detections if d.get("label") == "table column"]
    spans = [d["bbox"] for d in detections if d.get("label") == "table spanning cell"]

    rows = dedup_boxes(rows, iou_th=iou_th)
    rows.sort(key=lambda b: b[1])  # by y1
    cols = dedup_boxes(cols, iou_th=iou_th)
    cols.sort(key=lambda b: b[0])  # by x1

    n_rows, n_cols = len(rows), len(cols)
    total = n_rows * n_cols

    for sp in spans:
        r_idx = covered_indices(sp, rows, axis=1, overlap_th=overlap_th)
        c_idx = covered_indices(sp, cols, axis=0, overlap_th=overlap_th)
        area = max(1, len(r_idx)) * max(1, len(c_idx))
        total -= max(0, area - 1)

    out = {
        "rows": n_rows,
        "cols": n_cols,
        "cells": int(total),
        "rows_boxes": rows,
        "cols_boxes": cols,
    }
    LOGGER.debug(
        "count_cells_tatr: rows=%d cols=%d spans=%d cells=%d",
        n_rows, n_cols, len(spans), out["cells"],
    )
    return out



Box = Tuple[float, float, float, float]

def build_grid_with_spans(
    detections: Sequence[Dict[str, Any]],
    iou_th: float = 0.6,
    overlap_th: float = 0.5,
    text_fn: Callable[[int, int, str], str] | None = None,
) -> Dict[str, Any]:
    """Build a table grid (with merged cells) from TATR detections.

    The function:
      1) Uses `count_cells_tatr` to obtain deduplicated row/column boxes.
      2) Marks header rows using the first `"table column header"` bbox (if any).
      3) Marks row-header columns using the first `"table projected row header"`
         bbox (if any).
      4) Applies merges for `"table spanning cell"` by setting `rowspan/colspan`
         on the top-left covered cell and marking the others as `covered=True`.

    If `text_fn` is not provided, placeholder text is assigned:
      - data cell: "R{r}C{c}"
      - header row cell: "H{r}:{c}"
      - row-header column cell: "RHdr{r}"

    Args:
        detections: List of dicts with at least {'label', 'bbox'} coming from
            the model post-processing (`outputs_to_objects`).
        iou_th: IoU threshold to deduplicate row/column lines (passed to
            `count_cells_tatr`).
        overlap_th: Overlap threshold used to decide which rows/columns are
            covered by a given span (passed to `covered_indices`).
        text_fn: Optional callback `(r, c, kind) -> str` to assign initial text
            per cell. `kind` is one of: "data", "header-row", "row-header-col".

    Returns:
        Dict with:
            - 'grid': 2D list [r][c] with dicts:
                {'text': str, 'rowspan': int, 'colspan': int, 'covered': bool}
            - 'n_rows': int
            - 'n_cols': int
            - 'header_rows': List[int] (row indices marked as column-header rows)
            - 'row_header_cols': List[int] (column indices marked as row-header)
            - 'cells_counted': int (real cells, from `count_cells_tatr`)
            - 'rows_boxes': List[Box] (row line boxes)
            - 'cols_boxes': List[Box] (column line boxes)

    Raises:
        KeyError: If a detection is missing 'label' or 'bbox'.

    Notes:
        - Only the *first* detection of each special region type is used for
          marking (`"table column header"` and `"table projected row header"`).
        - `overlap_th` is applied with the same normalization rule as in
          `covered_indices` (min-extent normalization).
    """
    # Fail fast on malformed detections
    for d in detections:
        if "label" not in d or "bbox" not in d:
            raise KeyError(f"Detection missing 'label' or 'bbox': {d}")

    # 1) Row/column lines (dedup + sort) and counted cells
    stats = count_cells_tatr(detections, iou_th=iou_th, overlap_th=overlap_th)
    rows: List[Box] = stats["rows_boxes"]
    cols: List[Box] = stats["cols_boxes"]
    n_rows, n_cols = len(rows), len(cols)

    # 2) Special regions (first occurrence)
    header_bbox: Box | None = next(
        (d["bbox"] for d in detections if d.get("label") == "table column header"),
        None,
    )
    proj_row_hdr_bbox: Box | None = next(
        (d["bbox"] for d in detections if d.get("label") == "table projected row header"),
        None,
    )

    header_rows: Set[int] = set()
    if header_bbox:
        header_rows = set(covered_indices(header_bbox, rows, axis=1, overlap_th=0.5))

    row_header_cols: Set[int] = set()
    if proj_row_hdr_bbox:
        row_header_cols = set(covered_indices(proj_row_hdr_bbox, cols, axis=0, overlap_th=0.5))

    # 3) Initialize grid
    grid: List[List[Dict[str, Any]]] = [
        [{"text": "", "rowspan": 1, "colspan": 1, "covered": False} for _ in range(n_cols)]
        for _ in range(n_rows)
    ]

    # Default text function
    if text_fn is None:
        def text_fn(r: int, c: int, kind: str = "data") -> str:
            if kind == "header-row":
                return f"H{r}:{c}"
            if kind == "row-header-col":
                return f"RHdr{r}"
            return f"R{r}C{c}"

    # 4) Seed initial text based on role
    for r in range(n_rows):
        for c in range(n_cols):
            kind = "data"
            if r in header_rows:
                kind = "header-row"
            elif c in row_header_cols:
                kind = "row-header-col"
            grid[r][c]["text"] = text_fn(r, c, kind=kind)

    # 5) Apply merges for spanning cells
    spans: List[Box] = [d["bbox"] for d in detections if d.get("label") == "table spanning cell"]
    for sp in spans:
        r_idx = covered_indices(sp, rows, axis=1, overlap_th=overlap_th)
        c_idx = covered_indices(sp, cols, axis=0, overlap_th=overlap_th)
        if not r_idx or not c_idx:
            continue
        r0, c0 = min(r_idx), min(c_idx)
        grid[r0][c0]["rowspan"] = max(1, len(r_idx))
        grid[r0][c0]["colspan"] = max(1, len(c_idx))
        # mark covered cells
        for rr in r_idx:
            for cc in c_idx:
                if rr == r0 and cc == c0:
                    continue
                grid[rr][cc]["covered"] = True
                grid[rr][cc]["text"] = ""
    
    # 6) Ajuste estructural por "table projected row header":
    #     Cada detección de este tipo representa una fila de subtítulo que abarca
    #     toda la tabla horizontalmente (una única celda con colspan = n_cols).
    projected_hdrs = [
        d["bbox"] for d in detections if d.get("label") == "table projected row header"
    ]

    for bbox in projected_hdrs:
        # identificar qué fila cubre la proyección
        r_idx = covered_indices(bbox, rows, axis=1, overlap_th=overlap_th)
        if not r_idx:
            continue

        # se toma la primera fila cubierta como la fila de subtítulo
        r = r_idx[0]

        # convertir esa fila en una única celda que abarca todas las columnas
        grid[r][0]["colspan"] = n_cols
        grid[r][0]["text"] = grid[r][0].get("text", "") or f"Subheader{r}"

        # marcar el resto de las celdas de esa fila como cubiertas
        for c in range(1, n_cols):
            grid[r][c]["covered"] = True
            grid[r][c]["text"] = ""
                
    return {
        "grid": grid,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "header_rows": sorted(list(header_rows)),
        "row_header_cols": sorted(list(row_header_cols)),
        "cells_counted": stats["cells"],
        "rows_boxes": rows,
        "cols_boxes": cols,
    }

    

def to_html(grid_pack: Dict[str, Any]) -> str:
    """Render a merged-cell grid into faithful HTML (<table>) output.

    The function respects:
      - Row headers: rows in `grid_pack['header_rows']` are emitted with <th>.
      - Row-header columns: columns in `grid_pack['row_header_cols']` are
        emphasized with <strong> when placed in non-header rows.
      - Merged cells: cells marked as `covered=True` are skipped; their merge
        is represented by the top-left cell via `rowspan`/`colspan`.

    Args:
        grid_pack: Structure returned by `build_grid_with_spans`, containing:
            - 'grid': 2D list [r][c] of dicts with keys:
                {'text': str, 'rowspan': int, 'colspan': int, 'covered': bool}
            - 'n_rows': int
            - 'n_cols': int
            - 'header_rows': List[int] (optional)
            - 'row_header_cols': List[int] (optional)

    Returns:
        A string with an HTML table using <table>, <tr>, <th>/<td>,
        and proper `rowspan` / `colspan`.

    Examples:
        >>> html = to_html(grid_pack)
        >>> assert html.startswith("<table>")
    """
    g: List[List[Dict[str, Any]]] = grid_pack["grid"]
    n_rows: int = grid_pack["n_rows"]
    n_cols: int = grid_pack["n_cols"]
    header_rows = set(grid_pack.get("header_rows", []))
    row_header_cols = set(grid_pack.get("row_header_cols", []))

    html: List[str] = ["<table>"]
    for r in range(n_rows):
        html.append("  <tr>")
        tag = "th" if r in header_rows else "td"
        for c in range(n_cols):
            cell = g[r][c]
            if cell.get("covered", False):
                continue

            attrs = []
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            if rs > 1:
                attrs.append(f'rowspan="{rs}"')
            if cs > 1:
                attrs.append(f'colspan="{cs}"')

            content = str(cell.get("text", "") or "")
            if (c in row_header_cols) and (r not in header_rows) and tag == "td":
                content = f"<strong>{content}</strong>"

            attrs_str = (" " + " ".join(attrs)) if attrs else ""
            html.append(f"    <{tag}{attrs_str}>{content}</{tag}>")
        html.append("  </tr>")
    html.append("</table>")
    return "\n".join(html)



LOGGER = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def cell_bbox_from_grid(grid_pack: Dict[str, Any], r: int, c: int) -> Box:
    """Compute the pixel bbox of the anchor cell, honoring rowspan/colspan.

    Args:
        grid_pack: Output of `build_grid_with_spans`.
        r: Row index of the anchor cell.
        c: Column index of the anchor cell.

    Returns:
        (x1, y1, x2, y2) integer pixel coordinates.

    Raises:
        IndexError: If (r, c) is out of bounds.
        KeyError: If required keys are missing in `grid_pack`.
        ValueError: If computed bbox is invalid.
    """
    rows_boxes = grid_pack["rows_boxes"]
    cols_boxes = grid_pack["cols_boxes"]
    grid = grid_pack["grid"]
    n_rows, n_cols = grid_pack["n_rows"], grid_pack["n_cols"]

    if not (0 <= r < n_rows and 0 <= c < n_cols):
        raise IndexError(f"Cell index out of bounds: ({r}, {c})")

    cell = grid[r][c]
    rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))

    y1 = min(rows_boxes[rr][1] for rr in range(r, r + rs))
    y2 = max(rows_boxes[rr][3] for rr in range(r, r + rs))
    x1 = min(cols_boxes[cc][0] for cc in range(c, c + cs))
    x2 = max(cols_boxes[cc][2] for cc in range(c, c + cs))

    x1i, y1i, x2i, y2i = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    if x2i <= x1i or y2i <= y1i:
        raise ValueError(f"Invalid cell bbox at ({r},{c}): {(x1i, y1i, x2i, y2i)}")

    return x1i, y1i, x2i, y2i


import re

def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text: convert newlines (real or literal) to spaces."""
    if not text:
        return ""
    # Reemplaza saltos reales y variantes escritas como \n, \\n o /n
    text = text.replace("\x0c", " ")
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("\\n", " ").replace("/n", " ")
    # Colapsar espacios múltiples
    return re.sub(r"\s+", " ", text).strip()


def ocr_crop(
    image_bgr: np.ndarray,
    bbox: Box,
    tess_cfg: str = "--oem 3 --psm 6",
) -> str:
    """Run Tesseract OCR on an image crop without preprocessing.

    Args:
        image_bgr: Image as OpenCV BGR array.
        bbox: Crop box (x1, y1, x2, y2) in pixels.
        tess_cfg: Extra Tesseract config (e.g., OEM/PSM).

    Returns:
        OCR text normalized with line/page breaks and escape sequences replaced by spaces.
    """
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy.ndarray (OpenCV image).")

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        LOGGER.debug("ocr_crop: empty crop %s skipped", (x1, y1, x2, y2))
        return ""

    crop = image_bgr[y1:y2, x1:x2]
    txt = pytesseract.image_to_string(crop,lang="spa+eng", config=tess_cfg)
    return normalize_ocr_text(txt)


def fill_grid_with_ocr(
    grid_pack: Dict[str, Any],
    image_path: str,
    tess_cfg: str = "--oem 3 --psm 6",
    skip_headers: bool = False,
) -> Dict[str, Any]:
    """Fill the anchor cells of a grid with OCR text from the source image.

    Writes OCR text into each non-covered anchor cell. If `skip_headers=True`,
    header rows detected in `grid_pack['header_rows']` are skipped.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")

    header_rows = set(grid_pack.get("header_rows", []))
    n_rows, n_cols = grid_pack["n_rows"], grid_pack["n_cols"]

    for r in range(n_rows):
        if skip_headers and (r in header_rows):
            continue
        for c in range(n_cols):
            cell = grid_pack["grid"][r][c]
            if cell.get("covered", False):
                continue
            bbox = cell_bbox_from_grid(grid_pack, r, c)
            text = ocr_crop(img, bbox, tess_cfg=tess_cfg)
            grid_pack["grid"][r][c]["text"] = normalize_ocr_text(text)

    return grid_pack





def ocr_table_tokens(
    image_path: str,
    tess_cfg: str = "--oem 3 --psm 6",
    min_conf: int = -1,
) -> pd.DataFrame:
    """Run global Tesseract OCR and return a word-level dataframe.

    Args:
        image_path: Path to the source image (table).
        tess_cfg: Tesseract config string (OEM/PSM/lang, etc.).
        min_conf: Minimum confidence to keep (inclusive). Use -1 to keep all.

    Returns:
        Pandas DataFrame with at least: left, top, width, height, text, conf,
        block_num, par_num, line_num, word_num.

    Raises:
        ValueError: If the image cannot be opened.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")

    df = pytesseract.image_to_data(img, config=tess_cfg, output_type=Output.DATAFRAME)
    # Keep valid rows: conf != -1, non-empty text, and above min_conf
    df = df[(df["conf"] != -1) & df["text"].astype(str).str.strip().ne("")]
    if min_conf > -1:
        df = df[df["conf"].astype(float) >= float(min_conf)]
    df = df.copy()
    return df


def _find_anchor_for_spanned_cell(
    grid: List[List[Dict[str, Any]]],
    r: int,
    c: int,
) -> Optional[Tuple[int, int]]:
    """Given (r,c) that might be covered, find its anchor (top-left) cell.

    Scans up-left region to locate a non-covered cell whose (rowspan,colspan)
    region contains (r,c).
    """
    for rr in range(r, -1, -1):
        for cc in range(c, -1, -1):
            cand = grid[rr][cc]
            if cand.get("covered", False):
                continue
            rs = int(cand.get("rowspan", 1))
            cs = int(cand.get("colspan", 1))
            if rr <= r < rr + rs and cc <= c < cc + cs:
                return rr, cc
    return None


def assign_tokens_to_cells_center(
    grid_pack: Dict[str, Any],
    ocr_df: pd.DataFrame,
    joiner: str = " ",
    skip_headers: bool = False,
) -> Dict[str, Any]:
    """Assign each OCR token to a cell by the token's center point.

    For each token, we compute its center (cx, cy). We find the row index
    whose vertical band contains cy and the column index whose horizontal
    band contains cx. If that cell is covered, we locate its anchor (top-left)
    cell within the spanning region. The token text is then appended to that
    anchor cell's text. Tokens inside header rows can be skipped.

    Args:
        grid_pack: Output of `build_grid_with_spans` (modified in place).
        ocr_df: DataFrame from `ocr_table_tokens`.
        joiner: Separator used to concatenate tokens inside each cell.
        skip_headers: If True, do not assign tokens to header rows.

    Returns:
        The same `grid_pack`, with `grid[r][c]['text']` updated.

    Notes:
        - Token order is approximated later by sorting within each cell using
          (line_num, left, word_num). This is robust and readable for most tables.
    """
    rows_boxes = grid_pack["rows_boxes"]
    cols_boxes = grid_pack["cols_boxes"]
    grid = grid_pack["grid"]
    n_rows, n_cols = grid_pack["n_rows"], grid_pack["n_cols"]
    header_rows = set(grid_pack.get("header_rows", []))

    # Clear texts first (anchor cells only)
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c].get("covered", False):
                continue
            if skip_headers and (r in header_rows):
                continue
            grid[r][c]["text"] = ""

    # Bucket tokens per (r,c) anchor
    per_cell: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    for _, row in ocr_df.iterrows():
        x1, y1 = int(row.left), int(row.top)
        x2, y2 = x1 + int(row.width), y1 + int(row.height)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Locate row band (vertical)
        r_idx = next((ri for ri, rb in enumerate(rows_boxes) if rb[1] <= cy <= rb[3]), None)
        if r_idx is None:
            continue
        if skip_headers and (r_idx in header_rows):
            continue

        # Locate column band (horizontal)
        c_idx = next((ci for ci, cb in enumerate(cols_boxes) if cb[0] <= cx <= cb[2]), None)
        if c_idx is None:
            continue

        # Anchor cell (handle spanning)
        cell = grid[r_idx][c_idx]
        if cell.get("covered", False):
            anchor = _find_anchor_for_spanned_cell(grid, r_idx, c_idx)
            if anchor is None:
                continue
            ar, ac = anchor
        else:
            ar, ac = r_idx, c_idx

        per_cell.setdefault((ar, ac), []).append(
            {
                "text": str(row.text).strip(),
                "left": int(row.left),
                "top": int(row.top),
                "line_num": int(row.get("line_num", 0) or 0),
                "word_num": int(row.get("word_num", 0) or 0),
            }
        )

    # Sort tokens per cell and join
    for (r, c), toks in per_cell.items():
        toks.sort(key=lambda t: (t["line_num"], t["top"], t["left"], t["word_num"]))
        text = joiner.join(t["text"] for t in toks).strip()
        grid[r][c]["text"] = text

    return grid_pack


def fill_grid_from_global_ocr_centered(
    grid_pack: Dict[str, Any],
    image_path: str,
    tess_cfg: str = "--oem 3 --psm 6",
    min_conf: int = -1,
    joiner: str = " ",
    skip_headers: bool = False,
) -> Dict[str, Any]:
    """Full pipeline: global OCR → center-assignment → fill grid texts.

    Args:
        grid_pack: Output of `build_grid_with_spans` (modified in place).
        image_path: Path to the image used for OCR.
        tess_cfg: Tesseract config string.
        min_conf: Minimum confidence to keep tokens (-1 keeps all).
        joiner: Separator for tokens inside a cell.
        skip_headers: If True, do not fill header rows.

    Returns:
        The same `grid_pack` with texts filled via center-assignment.

    Raises:
        ValueError: If the image cannot be opened.
    """
    # Just to validate image exists/decodable, `ocr_table_tokens` reads it again
    if cv2.imread(image_path) is None:
        raise ValueError(f"Could not open image: {image_path}")

    df = ocr_table_tokens(image_path, tess_cfg=tess_cfg, min_conf=min_conf)
    return assign_tokens_to_cells_center(
        grid_pack=grid_pack, ocr_df=df, joiner=joiner, skip_headers=skip_headers
    )