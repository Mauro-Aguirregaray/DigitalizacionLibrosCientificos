from PIL import Image

if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.Resampling.BILINEAR
if not hasattr(Image, "BILINEAR"):
    Image.BILINEAR = Image.Resampling.BILINEAR
if not hasattr(Image, "NEAREST"):
    Image.NEAREST = Image.Resampling.NEAREST
if not hasattr(Image, "BOX"):
    Image.BOX = Image.Resampling.BOX
if not hasattr(Image, "HAMMING"):
    Image.HAMMING = Image.Resampling.HAMMING
if not hasattr(Image, "LANCZOS"):
    Image.LANCZOS = Image.Resampling.LANCZOS
