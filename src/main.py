import sys
import os
from pathlib import Path
import shutil
import webbrowser
import re


# Ruta absoluta a la carpeta src
sys.path.append(os.path.abspath("../src"))

# Dependencias del módulo main
from utils_main import (
    enviar_imagen,
    reemplazar_ruta_img,
    generar_html_desde_recortes
    
)

if __name__ == "__main__":

    # 🟢 Pedir ruta al usuario (pegar en consola)
    ruta = input("📂 Ingresá la ruta completa de la imagen: ").strip('"').strip()

    enviar_imagen(ruta)

    # 🧩 Derivar nombre base de la imagen
    nombre_imagen = Path(ruta).stem

    # --- Generar el HTML final ---
    html_resultado = generar_html_desde_recortes(nombre_imagen)

    # --- Guardar el HTML en la carpeta temp ---
    output_path = Path("temp") / nombre_imagen / "output.html"
    output_path.write_text(html_resultado, encoding="utf-8")

    print(f"✅ HTML generado en: {output_path}")

    # --- Carpeta base ---
    carpeta_base = Path("temp") / nombre_imagen
    carpeta_recortes = carpeta_base / "recortes"
    carpeta_figuras = carpeta_base / "figuras"
    carpeta_figuras.mkdir(exist_ok=True)

    # --- Copiar figuras ---
    for archivo in sorted(carpeta_recortes.iterdir()):
        if "figure" in archivo.name.lower() and archivo.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            destino = carpeta_figuras / archivo.name
            shutil.copy(archivo, destino)

    html_portable = re.sub(r'src="([^"]+figure[^"]+)"', reemplazar_ruta_img, html_resultado)

    # --- Guardar HTML portable ---
    html_path = carpeta_base / "output.html"
    html_path.write_text(html_portable, encoding="utf-8")

    # --- Abrir en navegador ---
    webbrowser.open(html_path.resolve().as_uri())
    print(f"🌐 HTML exportado correctamente en: {html_path}")