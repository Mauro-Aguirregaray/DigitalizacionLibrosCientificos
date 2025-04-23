# 📘 Tesis: Digitalización de Libros Científicos

Este proyecto forma parte del trabajo final de la Especialización en Inteligencia Artificial con temática en la digitalización de libros científicos mediante el uso de inteligencia artificial. La propuesta combina detección estructural (layout) y uso de técnicas de OCR adaptadas según el tipo de contenido.

---

## 🎯 Objetivo

Desarrollar un sistema capaz de detectar la estructura lógica de documentos científicos escaneados —segmentando bloques como texto, tablas, figuras, listas y títulos— y aplicar motores de OCR específicos según el tipo de contenido identificado. El proyecto también contempla la evaluación comparativa de distintos motores OCR para determinar cuál ofrece el mejor desempeño en cada tipo de estructura textual.

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
├── docs/               # Documentación de tesis
└── README.md           # Este archivo
```

---
## 🧪 Estado Actual

- [x] Dataset base organizado y documentado.
- [x] Ground truth de texto generado (300 regiones).
- [ ] Integración completa del OCR adaptativo.
- [ ] Clasificación automática de bloques.
- [ ] Resultados finales y visualización.

---

## 🚀 Cómo Empezar (próximamente)

> Requisitos mínimos: Python 3.10+, `torch`, `opencv`, `pytesseract`

1. Clonar el repositorio:

```
git clone https://github.com/tu_usuario/tesis-digitalizacion-libros-ia.git
cd tesis-digitalizacion-libros-ia
```

2. Instalar dependencias (ejemplo con `requirements.txt`):

```
pip install -r requirements.txt
```

3. Ejecutar un ejemplo básico *(proximamente)*.

---



