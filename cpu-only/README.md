# Imagen personalizada CPU para ModelArts Notebook

> Índice del repositorio: [`../README.md`](../README.md)

`Dockerfile` basado en la guía oficial de Huawei Cloud
[Creating a Custom Image on ECS and Using It](https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_0011-0.html#section5),
preparado **solo para Notebook con CPU** (sin CUDA ni CANN).

## Qué incluye

- Base `ubuntu:22.04` (x86_64), permitida por la especificación de imágenes.
- Usuario `ma-user` (UID `1000`) y grupo `ma-group` (GID `100`), obligatorios en ModelArts.
- Miniconda en `/home/ma-user/anaconda3` con espejos de Tsinghua (conda) y Huawei Cloud (pip).
- Entorno conda `ml-cpu` con Python 3.11.10, `ipykernel` y librerías habituales
  (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`).
- Kernel de Jupyter registrado como `ml-cpu` (lo exige el escenario Notebook).
- PyTorch CPU opcional (`torch`, `torchvision`, `torchaudio`).
- Sin JupyterLab/Jupyter Notebook propios, para no chocar con los archivos
  estáticos que inyecta ModelArts.
- `ffmpeg`, `curl`, compiladores y `openssh-server`.

> El tipo de imagen al registrarla en ModelArts debe ser **CPU**. Al no incluir
> paquetes CUDA/CANN, ModelArts la clasifica como CPU.

## 1. Construir la imagen

En una ECS de Huawei Cloud con Docker (Ubuntu 22.04, misma arquitectura x86_64):

```bash
cd /ruta/al/proyecto

docker build \
  -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-cpu:v1 \
  -f Dockerfile .
```

Cambia `cn-north-4` por tu región y `<organizacion>` por tu organización de SWR.

Para no instalar PyTorch (imagen más pequeña y build más rápido):

```bash
docker build \
  --build-arg INSTALL_PYTORCH_CPU=false \
  -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-cpu:v1 \
  -f Dockerfile .
```

### Argumentos de construcción

| Argumento | Valor por defecto | Descripción |
| --- | --- | --- |
| `PYTHON_VERSION` | `3.11.10` | Versión de Python del entorno conda. |
| `CONDA_ENV_NAME` | `ml-cpu` | Nombre del entorno y del kernel de Jupyter. |
| `INSTALL_PYTORCH_CPU` | `true` | Instala PyTorch CPU. Usa `false` para omitirlo. |
| `PYTORCH_VERSION` | `2.7.0` | Versión de PyTorch (se instala como `+cpu`). |
| `TORCHVISION_VERSION` | `0.22.0` | Versión de torchvision. |
| `TORCHAUDIO_VERSION` | `2.7.0` | Versión de torchaudio. |
| `MINICONDA_URL` | Mirror de Tsinghua | Instalador de Miniconda. |
| `MINICONDA_SHA256` | (hash fijado) | Checksum del instalador. Si cambias la URL, cambia también el hash. |

## 2. Verificar la imagen antes de subirla

```bash
# Arquitectura: debe ser x86_64
docker run --rm <imagen> uname -m

# Usuario y grupo exigidos
docker run --rm <imagen> id
# uid=1000(ma-user) gid=100(ma-group) ...

# Kernel de Jupyter instalado
docker run --rm <imagen> python -m ipykernel --version

# PyTorch debe reportar que NO hay CUDA
docker run --rm <imagen> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.7.0+cpu False
```

## 3. Subir a SWR y registrar en ModelArts

1. Inicia sesión en SWR con el comando que genera la consola
   (**Dashboard > Generate Login Command**).
2. Sube la imagen:

```bash
docker push swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-cpu:v1
```

3. En ModelArts, ve a **Asset Management > Images > Registered Images > Register**
   y selecciona la imagen de SWR.
   - **Architecture**: `x86_64` (el resultado de `uname -m` es `x86_64`).
   - **Type**: `CPU`.
4. Crea el Notebook con **Custom Images**, selecciona la imagen registrada y
   **WebIDE = JupyterLab**.
5. Al abrir JupyterLab, selecciona el kernel
   **Python 3.11.10 (ml-cpu, CPU)**.

## Notas

- La imagen se debe construir en la misma arquitectura del Notebook. Este
  Dockerfile es para **x86_64**. Para ARM (EulerOS/HCE) hay que cambiar la imagen
  base, `MINICONDA_URL`/`MINICONDA_SHA256` y varias dependencias.
- Si necesitas una base de ModelArts, cambia el `FROM` por la imagen oficial y
  puedes eliminar la sección de creación de `ma-user`/`ma-group` (ya existen).
- No agregues JupyterLab al Dockerfile. Si una imagen trae archivos estáticos de
  Jupyter, el Notebook puede no abrir; en ese caso revisa `jupyter --path` y
  elimina los directorios estáticos según la guía de Huawei.
- El comando de arranque real lo define ModelArts; el `CMD ["/bin/bash"]` solo
  sirve para inspeccionar la imagen localmente.
