import requests
from pathlib import Path
import pytesseract
from PIL import Image
from transformers import TableTransformerForObjectDetection
from torchvision import transforms
from PIL import Image
from typing import Sequence
from dataclasses import dataclass
import requests
import torch

# Dependencias del módulo TATR
from utils_TATR import (
    outputs_to_objects,
    build_grid_with_spans,
    fill_grid_from_global_ocr_centered,
    to_html,
)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cargar modelo en el notebook
structure_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-structure-recognition-v1.1-all"
).to(device)

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




def detectar_tipo_recorte(nombre_archivo: str) -> str:
    """
    Determina el tipo de bloque a partir del nombre del archivo.
    Ejemplo: '03_text.png' → 'texto'
    """
    nombre = nombre_archivo.lower()
    if "text" in nombre or "title" in nombre or "list" in nombre:
        return "texto"
    elif "table" in nombre:
        return "tabla"
    elif "figure" in nombre:
        return "figura"
    return "otro"


def limpiar_texto_ocr(texto: str) -> str:
    """
    Limpia el texto obtenido por OCR:
    - Une palabras cortadas por guión al final de línea.
    - Reemplaza saltos de línea por espacios.
    - Elimina espacios múltiples.
    """
    texto = texto.replace("-\n", "")  # une palabras partidas por salto
    texto = texto.replace("\n", " ")  # convierte saltos de línea en espacios
    texto = " ".join(texto.split())   # reduce espacios consecutivos
    return texto.strip()

def procesar_con_tesseract(path_imagen: Path) -> str:
    """
    Ejecuta OCR con Tesseract (psm 3) y devuelve texto limpio en HTML.
    """
    texto = pytesseract.image_to_string(Image.open(path_imagen), lang="spa+eng", config="--psm 3")
    texto_limpio = limpiar_texto_ocr(texto)
    return f"<p>{texto_limpio}</p>"

def procesar_con_figura(path_imagen: Path) -> str:
    """
    Inserta directamente una imagen en el HTML sin aplicar OCR.
    Se usa para bloques tipo 'figure'.
    """
    rel_path = path_imagen.as_posix()  # ruta compatible con HTML
    html = f"""
    <div class='figure-block'>
        <img src="{rel_path}" alt="{path_imagen.name}" class="figure-img">
    </div>
    """
    return html

def procesar_con_tatr(path_imagen: Path) -> str:
    """
    Procesa una tabla con TATR (Table Transformer) y devuelve el HTML resultante.
    """
    # 1) Abrir la imagen
    cropped_table = Image.open(path_imagen).convert("RGB")

    # 2) Forward pass del modelo
    pixel_values = structure_transform(cropped_table).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # 3) Obtener detecciones y etiquetas
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)

    # 4) Armar grilla estructural con spans y headers
    pack = build_grid_with_spans(cells, iou_th=0.6, overlap_th=0.5)

    # 5) OCR global → asignación por centro
    pack = fill_grid_from_global_ocr_centered(
        grid_pack=pack,
        image_path=str(path_imagen),
        tess_cfg="--oem 3 --psm 3",
        min_conf=0,          # podés subir a 40-60 si hay ruido
        joiner=" ",
        skip_headers=False,
    )

    # 6) Convertir a HTML (con clase específica para aislar estilos)
    HTML_STYLE = """
    <style>
    .tatr-table table {
        border-collapse: collapse;
        font-family: system-ui, sans-serif;
        margin: 10px auto;
    }
    .tatr-table th, .tatr-table td {
        border: 1px solid #ccc;
        padding: 6px 10px;
        text-align: left;
    }
    .tatr-table th {
        background: #f6f6f6;
        font-weight: 600;
    }
    </style>
    """

    html_table = f"<div class='tatr-table'>{to_html(pack)}</div>"
    return HTML_STYLE + html_table

def generar_html_desde_recortes(nombre_imagen: str, base_dir: str = "temp") -> str:
    """
    Recorre los recortes de una imagen, aplica el modelo correspondiente a cada tipo
    (Tesseract, TATR o figura), y genera un HTML final homogéneo y ordenado.
    """
    carpeta_recortes = Path(base_dir) / nombre_imagen / "recortes"
    if not carpeta_recortes.exists():
        raise FileNotFoundError(f"No se encontró la carpeta: {carpeta_recortes}")

    html_partes = []
    buffer_texto = ""

    # Procesar cada archivo en orden
    for archivo in sorted(carpeta_recortes.iterdir()):
        if archivo.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        tipo = detectar_tipo_recorte(archivo.name)

        if tipo == "texto":
            bloque = procesar_con_tesseract(archivo)
            # acumulamos el contenido del bloque de texto, sin sus divs externos
            buffer_texto += bloque.replace("<div class='ocr-text-block'>", "").replace("</div>", "")
        else:
            # Si hay texto pendiente, lo volcamos antes del nuevo tipo de bloque
            if buffer_texto:
                html_partes.append(f"<div class='ocr-text-block'>{buffer_texto}</div>")
                buffer_texto = ""

            if tipo == "tabla":
                html_partes.append(procesar_con_tatr(archivo))
            elif tipo == "figura":
                html_partes.append(procesar_con_figura(archivo))
    # Si termina con texto pendiente, lo agregamos
    if buffer_texto:
        html_partes.append(f"<div class='ocr-text-block'>{buffer_texto}</div>")

    # Estilo general más homogéneo
    estilo = """
    <style>
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        max-width: 850px;
        margin: 40px auto;
        line-height: 1.55;
        color: #222;
    }

    p {
        text-align: justify;
        margin: 0 0 10px 0;
    }

    .ocr-text-block {
        margin: 0;
        padding: 0;
    }

    .tatr-table {
        margin: 16px auto;
        width: 100%;
    }

    .tatr-table table {
        border-collapse: collapse;
        font-family: inherit;
        width: 100%;
    }

    .tatr-table th, .tatr-table td {
        border: 1px solid #ccc;
        padding: 4px 8px;
        font-size: 0.95em;
    }

    .tatr-table th {
        background: #f6f6f6;
    }

    .figure-block {
        text-align: center;
        margin: 14px 0;
    }

    .figure-img {
        max-width: 90%;
        border-radius: 4px;
        display: inline-block;
    }
    </style>
    """

    html_final = (
        f"<!DOCTYPE html><html><head>{estilo}</head><body>"
        + "\n".join(html_partes)
        + "</body></html>"
    )
    return html_final

# --- Reescribir rutas <img src="..."> dentro del HTML ---
def reemplazar_ruta_img(match):
    filename = Path(match.group(1)).name  # obtiene solo el nombre del archivo
    return f'src="figuras/{filename}"'

def enviar_imagen(ruta_imagen: str, url_api: str = "http://localhost:8000/analyze/") -> None:
    """
    Envía una imagen al endpoint /analyze/ de FastAPI para que el servidor la procese
    y genere los recortes en /home/appuser/temp/.

    Args:
        ruta_imagen (str): Ruta local de la imagen a enviar.
        url_api (str): URL del endpoint FastAPI (por defecto http://localhost:8000/analyze/).
    """
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta}")

    with open(ruta, "rb") as img:
        files = {"file": (ruta.name, img, "image/png")}
        print(f"📤 Enviando '{ruta.name}' a {url_api} ...")
        response = requests.post(url_api, files=files)

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}")
        print(response.text[:300])  # Muestra un fragmento del mensaje de error
        return

    print("✅ Imagen enviada correctamente.")
    print("🗂️  El servidor generó los recortes en /home/appuser/temp/")
    print("💡 Revisá esa carpeta para ver los resultados.")


# Ejemplo de uso directo desde terminal:
if __name__ == "__main__":
    # Cambiá esta ruta por la de tu imagen local
    ruta = "C:/Users/Usuario/Desktop/eolica.png"
    enviar_imagen(ruta)
