# 📊 Dataset Summary – Digitalización Libros Científicos

Este documento resume las principales características del dataset utilizado para evaluar y validar el pipeline de detección de layout y OCR adaptativo en libros científicos.

---

## 🗂️ Datos Originales

Se utilizan datos provenientes de dos fuentes principales:

1. **[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)**  
   - **Cantidad de documentos seleccionados:** 100  
   - **Formato de archivo:** JPG (imágenes individuales de páginas científicas)  
   - **Carpeta:** `data/raw/publaynet/`

2. **[PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet)** *(planeado)*  
   - **Cantidad estimada a incorporar:** 100 tablas  
   - **Formato de archivo:** JPG (recortes de tablas)  
   - **Carpeta:** `data/raw/pubtabnet/`

---

## ✂️ Estructuras de Texto Extraídas (PubLayNet)

A partir de los documentos de PubLayNet, se extrajeron regiones correspondientes a **cinco tipos de estructuras de texto**.

| Tipo de estructura | Cantidad de recortes |
|--------------------|----------------------|
| Figura             | 22                   |
| Lista              | 25                   |
| Tabla              | 32                   |
| Título             | 181                  |
| Texto              | 767                  |

> ⚠️ Las estructuras extraídas se almacenan en `data/regions/`, separadas por clase y dataset.

---

## 🧠 Ground Truth Manual OCR

Se generó un ground truth textual **solo para recortes de tipo `texto`**, derivados de PubLayNet.

- **Total de recortes de texto:** 767
- **Con ground truth generado manualmente:** 300
- **Sin ground truth (solo para test o inferencia):** 467

> 🔒 **Alcance cerrado:** no se planifica seguir ampliando el ground truth manual de texto.

---

## 🔄 Futuras Ampliaciones: Tablas desde PubTabNet

Está prevista la integración de una tanda adicional de **100 tablas provenientes del dataset PubTabNet**. Estas tablas se almacenarán y anotarán por separado para mantener la trazabilidad de origen.

Esto permitirá:

- Ampliar y diversificar la representación de estructuras tabulares.

---

## 🧪 Propósito del Dataset

Este dataset permite:

- Entrenar clasificadores de estructuras de texto a partir de layout detectado.
- Evaluar precisión del OCR adaptado por tipo de bloque textual.
- Comparar texto reconocido vs ground truth en evaluaciones cuantitativas.

---

## 🧷 Organización de Archivos

La estructura del dataset está organizada de la siguiente forma:

```bash
data/
├── raw/
│   ├── publaynet/
│   └── pubtabnet/           # (futuro)
│
├── regions/
│   ├── table/
│   │   ├── publaynet/
│   │   └── pubtabnet/       # (futuro)
│   └── ...
│
├── annotations/
│   ├── publaynet_boxes.json
│   ├── pubtabnet_boxes.json # (futuro)
│   └── ocr_text_labels.json
