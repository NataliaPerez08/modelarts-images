# Imagen personalizada para Huawei Cloud ModelArts Notebook con GPU NVIDIA.
#
# Sigue el patrón del repositorio: parte de una imagen base de ModelArts para
# PyTorch + CUDA (atelier/pytorch_cuda) y se asegura de cumplir los requisitos
# del escenario Notebook.
#
# Reparto de responsabilidades:
#   - Host ModelArts : driver NVIDIA + CUDA runtime expuestos por el entorno de
#                      ejecución (no se instala el driver dentro de la imagen).
#   - Esta imagen    : PyTorch/CUDA compatibles con la base, ipykernel y kernel
#                      de Jupyter, paquetes extra y usuario ma-user/ma-group.
#
# Referencias:
#   - Creating a Custom Image on ECS and Using It
#     https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_0011-0.html
#   - ModelArts Unified Images
#     https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_6022.html
#
# Construcción (host x86_64 con Docker):
#   docker build \
#     -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-gpu:v1 \
#     -f Dockerfile .
#
# Arquitectura: x86_64 (GPU NVIDIA en ModelArts Notebook).

ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/atelier/pytorch_cuda:pytorch_2.7.1-cuda_12.8-py_3.12-ubuntu_22.04-x86_64-20260417112518-aabfd52

FROM ${BASE_IMAGE}

# Re-declarado tras FROM para poder usarlo en LABEL
ARG BASE_IMAGE

# ---------------------------------------------------------------------------
# Argumentos de construcción
# ---------------------------------------------------------------------------
ARG BASE_PYTHON=
ARG KERNEL_NAME=ml-gpu
ARG INSTALL_EXTRA_PACKAGES=true
ARG EXTRA_PYTHON_PACKAGES="pandas scipy scikit-learn matplotlib tqdm"
ARG PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.huaweicloud.com
ARG INSTALL_DEV_TOOLS=false
ARG EXPECTED_ARCH=x86_64
ARG ENSURE_MA_USER=true

LABEL org.opencontainers.image.title="modelarts-notebook-gpu" \
      org.opencontainers.image.description="ModelArts Notebook image for NVIDIA GPU (PyTorch + CUDA, x86_64)" \
      org.opencontainers.image.vendor="Huawei Cloud ModelArts custom image" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV HOME=/home/ma-user \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# ---------------------------------------------------------------------------
# Sección 1: asegurar usuario ma-user / grupo ma-group (exigidos por ModelArts)
# Muchas imágenes base de ModelArts ya los traen; esta sección es idempotente.
# ---------------------------------------------------------------------------
RUN set -eux; \
    if [ "${ENSURE_MA_USER}" = "true" ]; then \
      if getent group 100 >/dev/null; then \
        existing_group="$(getent group 100 | cut -d: -f1)"; \
        if [ "${existing_group}" != "ma-group" ]; then \
          groupdel -f "${existing_group}" 2>/dev/null || true; \
        fi; \
      fi; \
      getent group 100 >/dev/null || groupadd -g 100 ma-group; \
      if getent passwd 1000 >/dev/null; then \
        existing_user="$(getent passwd 1000 | cut -d: -f1)"; \
        if [ "${existing_user}" != "ma-user" ]; then \
          userdel -r "${existing_user}" 2>/dev/null || true; \
        fi; \
      fi; \
      if ! id -u ma-user >/dev/null 2>&1; then \
        useradd -m -d /home/ma-user -u 1000 -g 100 -s /bin/bash ma-user; \
      fi; \
      usermod -u 1000 -g 100 -d /home/ma-user -s /bin/bash ma-user; \
      chown -R ma-user:ma-group /home/ma-user; \
      chmod 750 /home/ma-user; \
    fi

# ---------------------------------------------------------------------------
# Sección 2: localizar Python con PyTorch
# ---------------------------------------------------------------------------
RUN set -eux; \
    py="${BASE_PYTHON}"; \
    if [ -z "${py}" ]; then \
      for c in /home/ma-user/anaconda3/envs/*/bin/python \
               /home/ma-user/anaconda3/bin/python \
               /opt/conda/bin/python \
               /usr/bin/python3; do \
        [ -x "${c}" ] || continue; \
        if "${c}" -c 'import importlib.metadata as m; m.version("torch")' >/dev/null 2>&1; then \
          py="${c}"; break; \
        fi; \
      done; \
    fi; \
    if [ -z "${py}" ]; then \
      echo "ERROR: no se encontró ningún Python con torch instalado." >&2; \
      echo "Indica uno con --build-arg BASE_PYTHON=/ruta/al/python." >&2; \
      exit 1; \
    fi; \
    printf '%s\n' "${py}" > /home/ma-user/.gpu-python; \
    chown ma-user:ma-group /home/ma-user/.gpu-python; \
    printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "${py}" > /usr/local/bin/gpu-python; \
    chmod 755 /usr/local/bin/gpu-python; \
    echo "Python con torch: ${py}"

# ---------------------------------------------------------------------------
# Sección 3: verificación de arquitectura
# ---------------------------------------------------------------------------
RUN set -eux; \
    arch="$(uname -m)"; \
    echo "Arquitectura del build: ${arch}"; \
    if [ "${arch}" != "${EXPECTED_ARCH}" ]; then \
      echo "ERROR: se esperaba ${EXPECTED_ARCH} y se está construyendo en ${arch}." >&2; \
      echo "ModelArts espera x86_64 para GPU NVIDIA en este escenario." >&2; \
      exit 1; \
    fi; \
    py="$(cat /home/ma-user/.gpu-python)"; \
    "${py}" - <<'PY'
import importlib.metadata as md
import platform

torch_v = md.version("torch")
print(f"Python {platform.python_version()} | torch {torch_v}")
try:
    import torch
    print(f"CUDA disponible (build check): {torch.cuda.is_available()}")
    if torch.version.cuda:
        print(f"CUDA version: {torch.version.cuda}")
except Exception as exc:
    print(f"AVISO: import torch falló en build (puede ser normal): {exc}")
PY

# ---------------------------------------------------------------------------
# Sección 4: paquetes de sistema opcionales
# ---------------------------------------------------------------------------
RUN if [ "${INSTALL_DEV_TOOLS}" = "true" ]; then \
      set -eux; \
      if command -v apt-get >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends \
          git curl wget tar unzip build-essential ffmpeg openssh-server && \
        rm -rf /var/lib/apt/lists/*; \
      elif command -v dnf >/dev/null 2>&1; then \
        dnf install -y git curl wget tar unzip make gcc gcc-c++ ffmpeg openssh-server && dnf clean all; \
      elif command -v yum >/dev/null 2>&1; then \
        yum install -y git curl wget tar unzip make gcc gcc-c++ ffmpeg openssh-server && yum clean all; \
      else \
        echo "ERROR: gestor de paquetes no soportado"; exit 1; \
      fi; \
    fi

# ---------------------------------------------------------------------------
# Sección 5: paquetes Python extra + ipykernel
# ---------------------------------------------------------------------------
USER ma-user

RUN set -eux; \
    py="$(cat /home/ma-user/.gpu-python)"; \
    if [ "${INSTALL_EXTRA_PACKAGES}" = "true" ]; then \
      "${py}" -m pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" \
        --trusted-host "${PIP_TRUSTED_HOST}" \
        --upgrade-strategy only-if-needed \
        ipykernel ${EXTRA_PYTHON_PACKAGES}; \
      rm -rf /home/ma-user/.cache/pip; \
    fi

# ---------------------------------------------------------------------------
# Sección 6: registrar kernel de Jupyter
# ---------------------------------------------------------------------------
RUN set -eux; \
    py="$(cat /home/ma-user/.gpu-python)"; \
    py_ver="$("${py}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"; \
    "${py}" -m ipykernel install \
      --user \
      --name "${KERNEL_NAME}" \
      --display-name "Python ${py_ver} (${KERNEL_NAME}, GPU)"; \
    "${py}" -c "import jupyter_client.kernelspec as k; print('kernel:', k.KernelSpecManager().get_kernel_spec('${KERNEL_NAME}').resource_dir)"

# ---------------------------------------------------------------------------
# Sección 7: script de verificación
# ---------------------------------------------------------------------------
COPY --chown=ma-user:ma-group verify-gpu.sh /home/ma-user/verify-gpu.sh
RUN chmod 755 /home/ma-user/verify-gpu.sh

ENV KERNEL_NAME=${KERNEL_NAME}

RUN mkdir -p /home/ma-user/work && \
    chown ma-user:ma-group /home/ma-user/work && \
    chmod 750 /home/ma-user

WORKDIR /home/ma-user/work
USER ma-user
EXPOSE 8888

# ModelArts sobrescribe el comando de arranque; solo para uso local
CMD ["/bin/bash"]
