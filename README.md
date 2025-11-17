# 📘 Tesis: Digitalización de Libros Científicos

Este proyecto forma parte del trabajo final de la Especialización en Inteligencia Artificial con temática en la digitalización de libros científicos mediante el uso de inteligencia artificial. La propuesta combina detección estructural (layout) y uso de técnicas de OCR adaptadas según el tipo de contenido.

---

## 🎯 Objetivo

Desarrollar un sistema capaz de detectar la estructura lógica de imágenes documentos científicos —segmentando bloques como texto, tablas, figuras, listas y títulos— y aplicar motores de OCR específicos según el tipo de contenido identificado. El proyecto también contempla la evaluación comparativa de distintos motores OCR para determinar cuál ofrece el mejor desempeño en cada tipo de estructura textual.

---

## 🧩 Componentes Principales

- **Detector de layout:** identifica y clasifica bloques estructurales dentro de una página escaneada.
- **Clasificador de bloques:** determina la clase de cada recorte detectado.
- **OCR adaptativo:** aplica diferentes motores o configuraciones OCR según la clase del bloque.
- **Evaluación:** comparación entre el texto OCR y el ground truth manual.

---

## 📁 Estructura del Proyecto

```
tesis-digitalizacion-libros-ia/
├── data/               # Dataset estructurado (PubLayNet, PubTabNet)
├── src/                # Código fuente modular
├── notebooks/          # Notebooks de experimentación
├── results/            # Resultados y visualizaciones
├── docker/             # Docker para levantar LayoutParser
├── templates/          # Templates para la API de LayoutParser
├── temp/               # Resultados temporales
└── README.md           # Este archivo
```

---
## 🧪 Estado Actual

- [x] Dataset base organizado y documentado.
- [x] Ground truth de texto generado (300 regiones).
- [x] Integración completa del OCR adaptativo.
- [x] Clasificación automática de bloques.
- [x] Resultados finales y visualización.

---

## 🚀 Cómo usarlo

> **Requisitos mínimos:** Python 3.10+, torch, opencv, pytesseract, Docker Desktop / Docker Engine y soporte para Docker Compose.

### 1. Clonar el repositorio
git clone https://github.com/Mauro-Aguirregaray/DigitalizacionLibrosCientificos
cd DigitalizacionLibrosCientificos

### 2. Instalar dependencias
pip install -r requirements.txt

### 3. Iniciar Docker Desktop
Asegurarse de tener Docker Desktop (o Docker Engine en Linux) ejecutándose.

### 4. Levantar la API de LayoutParser
cd docker/detectron2
docker compose build
docker compose up

### 5. Ejecutar el proceso principal
python ./src/main.py
(El script solicitará la ruta de una imagen.)

### 6. Visualización del resultado
El resultado se abrirá automáticamente en el navegador.
Además, se generará en ./temp/NombreImagen/ el archivo:
output.html

---



