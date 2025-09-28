FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Dependencias básicas
# -----------------------------
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-dev \
    git \
    wget \
    sudo \
    ninja-build \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python

# -----------------------------
# Crear usuario no root
# -----------------------------
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

# -----------------------------
# Instalar pip (Python 3.8 compatible)
# -----------------------------
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python3 get-pip.py --user && \
    rm get-pip.py

# -----------------------------
# PyTorch 1.10.1 + torchvision 0.11.2 con CUDA 11.3
# -----------------------------
RUN pip install --user torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# -----------------------------
# Dependencias adicionales
# -----------------------------
RUN pip install --user tensorboard onnx fvcore \
    "numpy>=1.20" "pandas>=1.3" \
    pillow opencv-python-headless \
    fastapi uvicorn[standard] jinja2 python-multipart \
    layoutparser==0.3.4

# -----------------------------
# Detectron2 desde wheel precompilada
# -----------------------------
RUN pip install --user detectron2==0.6+cu113 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# -----------------------------
# Cache de modelos (persistente en /home/appuser/models)
# -----------------------------
ENV FVCORE_CACHE="/home/appuser/models"

# -----------------------------
# Crear carpetas necesarias
# -----------------------------
RUN mkdir -p /home/appuser/temp /home/appuser/models /home/appuser/templates

# -----------------------------
# Copiar código y templates
# -----------------------------
COPY src/ /home/appuser/src/
COPY templates/ /home/appuser/templates/
WORKDIR /home/appuser/src

# -----------------------------
# Comando de inicio
# -----------------------------
CMD ["uvicorn", "app_layoutparser:app", "--host", "0.0.0.0", "--port", "8000"]

