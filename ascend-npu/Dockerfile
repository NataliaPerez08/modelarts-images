# Imagen personalizada para Huawei Cloud ModelArts Notebook con NPU Ascend.
#
# A diferencia de ../cpu-only, esta imagen NO parte de ubuntu:22.04. Para ARM,
# ModelArts solo admite como base EulerOS 2.8/2.9/2.10.x o HCE 2.0/3.0, y el
# stack Ascend (CANN + torch_npu) tiene que venir de una imagen oficial. Por eso
# se parte de una imagen unificada de ModelArts para Ascend.
#
# Reparto de responsabilidades (importante):
#   - Host ModelArts : driver NPU Ascend + firmware, expuestos como
#                      /usr/local/Ascend/driver y /dev/davinci*.
#   - Esta imagen    : CANN toolkit (runtime, ops, compilador), torch, torch_npu,
#                      ipykernel y el kernel de Jupyter.
# El driver NO se instala aquí: son módulos de kernel + librerías del host y no
# pueden vivir dentro de un contenedor. Instalarlo en la imagen rompe el
# contenedor o queda inservible (el host ya monta su propio driver).
#
# Referencias:
#   - Crear una imagen personalizada en ECS:
#     https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_0011-0.html
#   - Imágenes unificadas de ModelArts (tags ascend):
#     https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_6022.html
#   - Imagen personalizada de entrenamiento PyTorch + Ascend (patrón CANN):
#     https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_6031.html
#
# Construcción (host ARM64/aarch64 con Docker; ver README.md):
#   docker build \
#     -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-ascend:v1 \
#     -f Dockerfile .
#
# Esta imagen solo sirve para arquitectura aarch64 (snt9b, snt9b23, snt3p).

ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/atelier/pytorch_ascend:pytorch_2.7.1-cann_8.5.2-py_3.12-euler_2.10.11-aarch64-snt9b-20260417112518-aabfd52

FROM ${BASE_IMAGE}

# Re-declarado tras FROM para poder usarlo en LABEL y en el resto del archivo.
ARG BASE_IMAGE

# ---------------------------------------------------------------------------
# Argumentos de construcción
# ---------------------------------------------------------------------------
# Python que ya trae torch + torch_npu. Vacío = autodetección entre los entornos
# de /home/ma-user/anaconda3 (recomendado: no reinstalar torch/torch_npu para no
# romper la compatibilidad con el CANN de la imagen base).
ARG BASE_PYTHON=
# Nombre técnico del kernel de Jupyter (y del entorno, si se usa como etiqueta).
ARG KERNEL_NAME=ml-npu
ARG INSTALL_EXTRA_PACKAGES=true
ARG EXTRA_PYTHON_PACKAGES="pandas scipy scikit-learn matplotlib tqdm"
ARG PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.huaweicloud.com
# La imagen base de ModelArts ya trae compiladores y ffmpeg. Activa esto solo si
# tu BASE_IMAGE es mínima; usa dnf/yum (EulerOS/HCE), no apt.
ARG INSTALL_DEV_TOOLS=false
ARG EXPECTED_ARCH=aarch64

LABEL org.opencontainers.image.title="modelarts-notebook-ascend" \
      org.opencontainers.image.description="ModelArts Notebook image for Ascend NPU (CANN + torch_npu, aarch64)" \
      org.opencontainers.image.vendor="Huawei Cloud ModelArts custom image" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

ENV HOME=/home/ma-user \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    ASCEND_DRIVER_HOME=/usr/local/Ascend/driver

# Usar bash para todos los RUN (evita romper /bin/sh del sistema).
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# ---------------------------------------------------------------------------
# Sección 1: localizar CANN y el Python que ya tiene torch + torch_npu
#
# No se importan torch/torch_npu aquí: sin el driver del host, importar
# torch_npu puede fallar por librerías del driver ausentes. Se comprueba la
# metadata de los paquetes, que sí está disponible en tiempo de build.
# ---------------------------------------------------------------------------
RUN set -eux; \
    cann_env=""; \
    for f in /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
             /usr/local/Ascend/nnae/latest/set_env.sh \
             /usr/local/Ascend/*/latest/set_env.sh; do \
      if [ -f "${f}" ]; then cann_env="${f}"; break; fi; \
    done; \
    if [ -z "${cann_env}" ]; then \
      echo "ERROR: no se encontró set_env.sh de CANN en /usr/local/Ascend." >&2; \
      echo "Usa como BASE_IMAGE una imagen de ModelArts para Ascend (CANN 8.x)." >&2; \
      exit 1; \
    fi; \
    printf '%s\n' "${cann_env}" > /home/ma-user/.ascend-set-env; \
    printf '%s\n' \
      '# Generado por ascend-npu/Dockerfile' \
      ". \"${cann_env}\"" \
      'export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' \
      'export PATH="/usr/local/Ascend/driver/tools${PATH:+:${PATH}}"' \
      > /etc/profile.d/00-ascend.sh; \
    chmod 644 /etc/profile.d/00-ascend.sh; \
    py="${BASE_PYTHON}"; \
    if [ -z "${py}" ]; then \
      for c in /home/ma-user/anaconda3/envs/*/bin/python \
               /home/ma-user/anaconda3/bin/python \
               /home/ma-user/miniconda3/bin/python \
               /usr/bin/python3; do \
        [ -x "${c}" ] || continue; \
        if "${c}" -c 'import importlib.metadata as m; m.version("torch"); m.version("torch_npu")' >/dev/null 2>&1; then \
          py="${c}"; break; \
        fi; \
      done; \
    fi; \
    if [ -z "${py}" ]; then \
      echo "ERROR: no hay ningún Python con torch y torch_npu instalados." >&2; \
      echo "Indica uno con --build-arg BASE_PYTHON=/ruta/al/python." >&2; \
      exit 1; \
    fi; \
    printf '%s\n' "${py}" > /home/ma-user/.npu-python; \
    chown ma-user:ma-group /home/ma-user/.npu-python /home/ma-user/.ascend-set-env; \
    printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "${py}" > /usr/local/bin/npu-python; \
    chmod 755 /usr/local/bin/npu-python; \
    echo "CANN encontrado en : ${cann_env}"; \
    echo "Python con torch_npu: ${py}"

# ---------------------------------------------------------------------------
# Sección 2: verificación del stack y de la arquitectura
# ---------------------------------------------------------------------------
RUN set -eux; \
    arch="$(uname -m)"; \
    echo "Arquitectura del build: ${arch}"; \
    if [ "${arch}" != "${EXPECTED_ARCH}" ]; then \
      echo "ERROR: se esperaba ${EXPECTED_ARCH} y se está construyendo en ${arch}." >&2; \
      echo "ModelArts exige que la arquitectura de la imagen coincida con la del" >&2; \
      echo "Notebook. Construye en una ECS ARM64 (Kunpeng) o usa buildx con emulación." >&2; \
      exit 1; \
    fi; \
    set +u; . "$(cat /home/ma-user/.ascend-set-env)"; set -u; \
    py="$(cat /home/ma-user/.npu-python)"; \
    "${py}" - <<'PY'
import importlib.metadata as md
import platform

torch_v = md.version("torch")
npu_v = md.version("torch_npu")
print(f"Python {platform.python_version()} | torch {torch_v} | torch_npu {npu_v}")

if torch_v.split(".")[:2] != npu_v.split(".")[:2]:
    raise SystemExit(
        f"ERROR: torch {torch_v} y torch_npu {npu_v} no son de la misma serie; "
        "torch_npu debe coincidir con la versión mayor/menor de torch."
    )

try:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
except Exception as exc:  # sin driver del host es normal
    print(f"AVISO: import de torch/torch_npu falló en build (normal sin NPU): {exc}")
else:
    print("import torch/torch_npu: OK")
PY

# ---------------------------------------------------------------------------
# Sección 3: paquetes de sistema opcionales (solo imágenes base mínimas)
# La imagen de ModelArts ya incluye gcc, make y ffmpeg; por eso está desactivado.
# ---------------------------------------------------------------------------
RUN if [ "${INSTALL_DEV_TOOLS}" = "true" ]; then \
      set -eux; \
      if command -v dnf >/dev/null 2>&1; then pm=dnf; \
      elif command -v yum >/dev/null 2>&1; then pm=yum; \
      else \
        echo "ERROR: ni dnf ni yum están disponibles." >&2; exit 1; fi; \
      "${pm}" install -y \
        git curl wget tar unzip which make gcc gcc-c++ ffmpeg openssh-server; \
      "${pm}" clean all; \
      rm -rf /var/cache/dnf/* /var/cache/yum/*; \
    fi

# ---------------------------------------------------------------------------
# Sección 4: paquetes Python extra + ipykernel sobre el entorno existente
#
# No se reinstalan torch ni torch_npu: la versión de la imagen base está
# garantizada contra su CANN. Reinstalarlos es la causa habitual del error de
# ModelArts "the used cann and CUDA environments are not compatible".
# ---------------------------------------------------------------------------
USER ma-user

RUN set -eux; \
    py="$(cat /home/ma-user/.npu-python)"; \
    if [ "${INSTALL_EXTRA_PACKAGES}" = "true" ]; then \
      "${py}" -m pip install --no-cache-dir \
        --index-url "${PIP_INDEX_URL}" \
        --trusted-host "${PIP_TRUSTED_HOST}" \
        --upgrade-strategy only-if-needed \
        ipykernel ${EXTRA_PYTHON_PACKAGES}; \
      rm -rf /home/ma-user/.cache/pip; \
    fi

# ---------------------------------------------------------------------------
# Sección 5: registrar el kernel de Jupyter (requisito del escenario Notebook)
# ---------------------------------------------------------------------------
RUN set -eux; \
    py="$(cat /home/ma-user/.npu-python)"; \
    py_ver="$("${py}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"; \
    "${py}" -m ipykernel install \
      --user \
      --name "${KERNEL_NAME}" \
      --display-name "Python ${py_ver} (${KERNEL_NAME}, Ascend NPU)"; \
    "${py}" -c "import jupyter_client.kernelspec as k; print('kernel:', k.KernelSpecManager().get_kernel_spec('${KERNEL_NAME}').resource_dir)"

# ---------------------------------------------------------------------------
# Sección 6: script de verificación en tiempo de ejecución (con NPU real)
# ---------------------------------------------------------------------------
COPY --chown=ma-user:ma-group verify-npu.sh /home/ma-user/verify-npu.sh
RUN chmod 755 /home/ma-user/verify-npu.sh

# El entorno CANN (PATH, LD_LIBRARY_PATH, PYTHONPATH, ASCEND_*) lo define la
# imagen base. /etc/profile.d/00-ascend.sh solo ayuda en shells interactivos y
# añade las rutas del driver montado por ModelArts.
ENV KERNEL_NAME=${KERNEL_NAME}

RUN mkdir -p /home/ma-user/work && \
    chown ma-user:ma-group /home/ma-user/work && \
    chmod 750 /home/ma-user

WORKDIR /home/ma-user/work
USER ma-user
EXPOSE 8888

# ModelArts sobrescribe el comando de arranque; este CMD solo sirve para
# inspeccionar la imagen localmente con `docker run -it <imagen>`.
CMD ["/bin/bash"]
