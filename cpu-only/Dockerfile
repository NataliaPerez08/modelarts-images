# Imagen personalizada SOLO CPU para Huawei Cloud ModelArts Notebook.
#
# Referencia: "Creating a Custom Image on ECS and Using It"
# https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_0011-0.html#section5
#
# Cumple los requisitos de la documentación para Notebook:
#   - Base x86_64 con Ubuntu 22.04 (permitida por la especificación de imágenes).
#   - Usuario ma-user (UID 1000) y grupo ma-group (GID 100).
#   - ipykernel instalado en el entorno que se usará como kernel de Jupyter.
#   - Sin CUDA ni CANN: al registrar la imagen en ModelArts el tipo debe ser "CPU".
#   - No se instala JupyterLab/Notebook, para evitar conflictos con los
#     archivos estáticos de Jupyter que inyecta ModelArts.
#
# Construcción (en una ECS con Docker):
#   docker build -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/<imagen>:v1 .
#
# Solo arquitectura x86_64 (igual que el host donde se construye).

FROM ubuntu:22.04

# ---------------------------------------------------------------------------
# Argumentos de construcción
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11.10
ARG CONDA_ENV_NAME=ml-cpu
ARG MINICONDA_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh
ARG MINICONDA_SHA256=807774bae6cd87132094458217ebf713df436f64779faf9bb4c3d4b6615c1e3a

# PyTorch CPU es opcional. Para omitirlo:
#   docker build --build-arg INSTALL_PYTORCH_CPU=false ...
ARG INSTALL_PYTORCH_CPU=true
ARG PYTORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCHAUDIO_VERSION=2.7.0

LABEL org.opencontainers.image.title="modelarts-notebook-cpu" \
      org.opencontainers.image.description="CPU-only ModelArts Notebook image (Ubuntu 22.04, Python, Jupyter kernel)" \
      org.opencontainers.image.vendor="Huawei Cloud ModelArts custom image"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/home/ma-user

# Usar bash para todos los RUN (evita romper /bin/sh del sistema).
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# ---------------------------------------------------------------------------
# Sección 1: paquetes del sistema y espejo APT de Huawei Cloud
# Se usa http para el primer apt-get: la imagen base mínima de Ubuntu todavía
# no tiene ca-certificates y no puede validar HTTPS. Después se instala
# ca-certificates y el resto de descargas (conda/pip) ya usan HTTPS.
# ---------------------------------------------------------------------------
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
    printf '%s\n' \
      'deb http://repo.huaweicloud.com/ubuntu/ jammy main restricted universe multiverse' \
      'deb http://repo.huaweicloud.com/ubuntu/ jammy-updates main restricted universe multiverse' \
      'deb http://repo.huaweicloud.com/ubuntu/ jammy-backports main restricted universe multiverse' \
      'deb http://repo.huaweicloud.com/ubuntu/ jammy-security main restricted universe multiverse' \
      > /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget \
      git \
      bzip2 \
      unzip \
      zip \
      build-essential \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      openssh-server \
      tini && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ---------------------------------------------------------------------------
# Sección 2: usuario ma-user / grupo ma-group exigidos por ModelArts
# ---------------------------------------------------------------------------
RUN set -eux; \
    if getent passwd 1000 >/dev/null; then \
      existing_user="$(getent passwd 1000 | cut -d: -f1)"; \
      if [ "${existing_user}" != "ma-user" ]; then \
        userdel -r "${existing_user}" 2>/dev/null || true; \
      fi; \
    fi; \
    if getent group 100 >/dev/null; then \
      existing_group="$(getent group 100 | cut -d: -f1)"; \
      if [ "${existing_group}" != "ma-group" ]; then \
        groupdel -f "${existing_group}" 2>/dev/null || true; \
      fi; \
    fi; \
    getent group 100 >/dev/null || groupadd -g 100 ma-group; \
    if ! id -u ma-user >/dev/null 2>&1; then \
      useradd -m -d /home/ma-user -u 1000 -g 100 -s /bin/bash ma-user; \
    fi; \
    usermod -u 1000 -g 100 -d /home/ma-user -s /bin/bash ma-user; \
    chown -R ma-user:ma-group /home/ma-user; \
    chmod 750 /home/ma-user; \
    mkdir -p /run/sshd; \
    chmod 755 /run/sshd

USER ma-user
WORKDIR /home/ma-user

# ---------------------------------------------------------------------------
# Sección 3: Miniconda (como ma-user) con checksum
# ---------------------------------------------------------------------------
RUN curl -fsSL "${MINICONDA_URL}" -o /tmp/miniconda.sh && \
    echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum -c - && \
    bash /tmp/miniconda.sh -b -p /home/ma-user/anaconda3 && \
    rm -f /tmp/miniconda.sh

# ---------------------------------------------------------------------------
# Sección 4: espejos de conda/pip y creación del entorno Python
# ---------------------------------------------------------------------------
RUN mkdir -p /home/ma-user/.pip /home/ma-user/.config/pip && \
    printf '%s\n' \
      'channels:' \
      '  - defaults' \
      'show_channel_urls: true' \
      'default_channels:' \
      '  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main' \
      '  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r' \
      'custom_channels:' \
      '  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' \
      > /home/ma-user/.condarc && \
    printf '%s\n' \
      '[global]' \
      'index-url = https://mirrors.huaweicloud.com/repository/pypi/simple' \
      'trusted-host = mirrors.huaweicloud.com' \
      > /home/ma-user/.config/pip/pip.conf && \
    cp /home/ma-user/.config/pip/pip.conf /home/ma-user/.pip/pip.conf

RUN /home/ma-user/anaconda3/bin/conda create -y -n "${CONDA_ENV_NAME}" \
      --override-channels \
      -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
      "python=${PYTHON_VERSION}" pip && \
    /home/ma-user/anaconda3/envs/"${CONDA_ENV_NAME}"/bin/python -m pip install --no-cache-dir \
      --upgrade pip setuptools wheel && \
    /home/ma-user/anaconda3/envs/"${CONDA_ENV_NAME}"/bin/python -m pip install --no-cache-dir \
      ipykernel \
      numpy \
      pandas \
      scipy \
      scikit-learn \
      matplotlib && \
    /home/ma-user/anaconda3/envs/"${CONDA_ENV_NAME}"/bin/python -m ipykernel install \
      --user \
      --name "${CONDA_ENV_NAME}" \
      --display-name "Python ${PYTHON_VERSION} (${CONDA_ENV_NAME}, CPU)" && \
    /home/ma-user/anaconda3/bin/conda clean -afy && \
    rm -rf /home/ma-user/.cache/pip

# ---------------------------------------------------------------------------
# Sección 5: PyTorch CPU (opcional) + verificación de que NO hay CUDA
# ---------------------------------------------------------------------------
RUN if [ "${INSTALL_PYTORCH_CPU}" = "true" ]; then \
      /home/ma-user/anaconda3/envs/"${CONDA_ENV_NAME}"/bin/python -m pip install --no-cache-dir \
        --index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
        --trusted-host mirrors.huaweicloud.com \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==${PYTORCH_VERSION}+cpu" \
        "torchvision==${TORCHVISION_VERSION}+cpu" \
        "torchaudio==${TORCHAUDIO_VERSION}+cpu" && \
      /home/ma-user/anaconda3/envs/"${CONDA_ENV_NAME}"/bin/python -c \
        "import torch; assert not torch.cuda.is_available(), 'Se esperaba una build CPU'; print('PyTorch CPU:', torch.__version__)" && \
      rm -rf /home/ma-user/.cache/pip; \
    fi

# ---------------------------------------------------------------------------
# Configuración final
# ---------------------------------------------------------------------------
ENV CONDA_ENV_NAME=${CONDA_ENV_NAME}
ENV CONDA_PREFIX=/home/ma-user/anaconda3/envs/${CONDA_ENV_NAME}
ENV PATH=/home/ma-user/anaconda3/envs/${CONDA_ENV_NAME}/bin:/home/ma-user/anaconda3/bin:${PATH}

# Directorio de trabajo habitual de los notebooks en ModelArts.
RUN mkdir -p /home/ma-user/work && \
    chown ma-user:ma-group /home/ma-user/work

WORKDIR /home/ma-user/work
USER ma-user
EXPOSE 8888

# ModelArts sobrescribe el comando de arranque; este CMD solo sirve para
# inspeccionar la imagen localmente con `docker run -it <imagen>`.
CMD ["/bin/bash"]
