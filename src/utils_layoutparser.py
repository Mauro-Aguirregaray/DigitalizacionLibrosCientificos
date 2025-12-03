import patch_pil   # aplicar monkey patch ANTES de importar detectron2

from layoutparser.models import Detectron2LayoutModel
import layoutparser as lp

import os
import cv2
from PIL import Image
import numpy as np
from layoutparser.elements import Layout
from typing import Union

# Carpeta global de salida
TEMP_DIR = "/home/appuser/temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def export_layout_blocks_as_crops(
    image: Union[Image.Image, np.ndarray],
    layout: Layout,
    output_dir: str,
    prefix: str = "",
    overwrite: bool = True,
    padding: int = 5,
) -> None:
    """Recorta y guarda cada bloque detectado en una carpeta de salida."""

    # Convertir imagen si es ndarray
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    img_w, img_h = image.size
    os.makedirs(output_dir, exist_ok=True)

    # Borrar archivos anteriores si se desea
    if overwrite:
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

    # Ordenar bloques de arriba hacia abajo
    sorted_blocks = sorted(layout, key=lambda b: b.block.y_1)

    for idx, block in enumerate(sorted_blocks, start=1):
        x1, y1, x2, y2 = (
            block.block.points[0][0],
            block.block.points[0][1],
            block.block.points[2][0],
            block.block.points[2][1],
        )

        # Padding y límites
        x1 = max(int(x1) - padding, 0)
        y1 = max(int(y1) - padding, 0)
        x2 = min(int(x2) + padding, img_w)
        y2 = min(int(y2) + padding, img_h)

        cropped = image.crop((x1, y1, x2, y2))

        block_type = block.type.lower() if block.type else "unknown"
        filename = f"{prefix}{idx:02d}_{block_type}.png"
        save_path = os.path.join(output_dir, filename)

        # Guardar con PIL
        cropped.save(save_path)
        print(f"✔ Guardado: {filename}  | tipo: {block_type} | bbox: ({x1}, {y1}, {x2}, {y2})")

    print(f"\n✅ Se exportaron {len(sorted_blocks)} recortes a la carpeta '{output_dir}/'.")

'''
def run_layout(image_path: str):
    # Nombre base del archivo sin extensión
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Carpeta de salida dentro de /home/appuser/temp
    output_dir = os.path.join(TEMP_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Cargar imagen con OpenCV (numpy.ndarray en BGR)
    image = cv2.imread(image_path)

    # Modelo LayoutParser + Detectron2
    model = Detectron2LayoutModel(
        'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
    )

    # Detectar layout
    layout = model.detect(image)

    # Dibujar cajas sobre la imagen
    draw_boxes_image = lp.draw_box(image, layout, box_width=3)

    # Guardar imagen con boxes en la carpeta
    boxed_path = os.path.join(output_dir, f"{base_name}_boxed.png")
    if isinstance(draw_boxes_image, np.ndarray):
        cv2.imwrite(boxed_path, draw_boxes_image)
    else:
        draw_boxes_image.save(boxed_path)
    print(f"📦 Guardada imagen con boxes en {boxed_path}")

    # Guardar recortes en subcarpeta "recortes"
    recortes_dir = os.path.join(output_dir, "recortes")
    export_layout_blocks_as_crops(
        image=image,
        layout=layout,
        output_dir=recortes_dir,
        padding=5,
    )

    # Devolver resultados como JSON-friendly
    return [block.to_dict() for block in layout]
'''
def run_layout(image_path: str):
    """
    Ejecuta LayoutParser sobre una imagen.
    Si es PNG -> la convierte a JPG sin alterar colores.
    Usa RGB para el modelo y BGR para guardar, evitando inversión de colores.
    """

    # --------------------------------
    # 1. Convertir PNG → JPG sin afectar colores
    # --------------------------------
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        print("🔄 Convirtiendo PNG a JPG sin modificar colores...")

        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)  # colores crudos BGR
        jpg_path = image_path.replace(".png", ".jpg")

        cv2.imwrite(jpg_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        image_path = jpg_path

        print(f"➡️ Imagen convertida a JPG en: {jpg_path}")

    # --------------------------------
    # 2. Nombre base y carpeta de salida
    # --------------------------------
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(TEMP_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------
    # 3. Leer imagen original en BGR (sin tocar colores)
    # --------------------------------
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # --------------------------------
    # 4. Convertir a RGB SOLO para LayoutParser
    # --------------------------------
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --------------------------------
    # 5. Cargar modelo LayoutParser
    # --------------------------------
    model = Detectron2LayoutModel(
        'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
    )

    # --------------------------------
    # 6. Detectar layout (usar RGB!)
    # --------------------------------
    layout = model.detect(image_rgb)

    # --------------------------------
    # 7. Dibujar cajas (LayoutParser siempre trabaja en RGB)
    # --------------------------------
    boxed_rgb = lp.draw_box(image_rgb, layout, box_width=3)

    # Asegurar numpy array (lp puede devolver PIL.Image)
    if not isinstance(boxed_rgb, np.ndarray):
        boxed_rgb = np.array(boxed_rgb)

    # Convertir de RGB → BGR antes de guardar (colores exactos)
    # boxed_bgr = cv2.cvtColor(boxed_rgb, cv2.COLOR_RGB2BGR)

    boxed_path = os.path.join(output_dir, f"{base_name}_boxed.jpg")
    cv2.imwrite(boxed_path, boxed_rgb)
    print(f"📦 Guardada imagen con boxes en {boxed_path}")

    # --------------------------------
    # 8. Guardar recortes (usar imagen BGR original)
    # --------------------------------
    recortes_dir = os.path.join(output_dir, "recortes")
    export_layout_blocks_as_crops(
        image=image_rgb,
        layout=layout,
        output_dir=recortes_dir,
        padding=5,
    )

    # --------------------------------
    # 9. Devolver resultado JSON-friendly
    # --------------------------------
    return [block.to_dict() for block in layout]


