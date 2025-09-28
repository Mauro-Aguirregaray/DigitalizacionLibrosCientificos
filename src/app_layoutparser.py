from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os

from utils_layoutparser import run_layout

app = FastAPI()
# Paths consistentes
TEMPLATES_DIR = "/home/appuser/templates"
TEMP_DIR = "/home/appuser/temp"

os.makedirs(TEMP_DIR, exist_ok=True)

# Configuración
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze/", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    # Guardar archivo subido
    temp_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Procesar con layoutparser
    results = run_layout(temp_path)

    # Carpeta de salida
    rel_output = os.path.splitext(file.filename)[0]
    output_dir = os.path.join(TEMP_DIR, rel_output)
    recortes_dir = os.path.join(output_dir, "recortes")

    # Archivos generados (rutas accesibles desde navegador)
    boxed_image = f"/temp/{rel_output}/{rel_output}_boxed.png"
    recortes = [
        f"/temp/{rel_output}/recortes/{r}" for r in sorted(os.listdir(recortes_dir))
    ] if os.path.exists(recortes_dir) else []

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "input_file": file.filename,
            "boxed_image": boxed_image,
            "recortes": recortes,
            "results": results,
        },
    )

