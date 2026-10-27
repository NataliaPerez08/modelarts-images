# Imagen personalizada GPU NVIDIA para ModelArts Notebook

> Índice del repositorio: [`../README.md`](../README.md)

`Dockerfile` para crear una imagen de Notebook con **GPU NVIDIA** (PyTorch + CUDA) siguiendo el patrón del repositorio. Parte de una imagen unificada de ModelArts `atelier/pytorch_cuda` y cumple los requisitos del escenario Notebook.

## Driver frente a CUDA: qué va en la imagen y qué pone el host

| Componente | Dónde vive | ¿Se instala en la imagen? |
| --- | --- | --- |
| Driver NVIDIA | Host de ModelArts; expuesto al contenedor | **No** |
| CUDA runtime/libraries | Suele venir en la imagen base `atelier/pytorch_cuda` | **Sí** (viene en la base) |
| `torch` + CUDA | Entorno Python de la imagen/base | **Sí** (viene en la base) |
| `ipykernel` y kernel de Jupyter | `/home/ma-user/.local/share/jupyter` | **Sí** (lo añade este Dockerfile) |

El driver no se instala dentro de la imagen; debe ser compatible con la versión de CUDA que trae la imagen base.

## Qué incluye

- Base **x86_64** de ModelArts para PyTorch + CUDA (configurable vía `BASE_IMAGE`).
- Asegura `ma-user` (UID `1000`) y `ma-group` (GID `100`) (idempotente).
- Detección automática del intérprete Python con `torch` (`/usr/local/bin/gpu-python`, `/home/ma-user/.gpu-python`).
- `ipykernel` y kernel de Jupyter registrado (`ml-gpu` por defecto), requisito del escenario Notebook.
- Paquetes habituales (`pandas`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`) instalados con los espejos de Huawei Cloud.
- Script `verify-gpu.sh` para comprobar en el Notebook que GPU/CUDA/PyTorch funcionan.
- Sin JupyterLab/Jupyter Notebook propios, para no chocar con los archivos estáticos que inyecta ModelArts.
- Sin CANN.

## 1. Elegir la imagen base

Sustituye `swr.<region>.myhuaweicloud.com` por tu región. Usa una imagen `atelier/pytorch_cuda` compatible con el driver NVIDIA del host de ModelArts.

Ejemplo por defecto (ajústalo a la versión disponible en tu región):

```text
swr.cn-north-4.myhuaweicloud.com/atelier/pytorch_cuda:pytorch_2.7.1-cuda_12.8-py_3.12-ubuntu_22.04-x86_64-20260417112518-aabfd52
```

Consulta las imágenes unificadas de ModelArts para obtener el tag correcto.

## 2. Construir la imagen

La imagen es **x86_64**: constrúyela en una ECS x86_64 con Docker.

```bash
cd gpu-nvidia

docker build \
  -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-gpu:v1 \
  -f Dockerfile .
```

Con otra imagen base:

```bash
docker build \
  --build-arg BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/atelier/pytorch_cuda:<tag> \
  -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-gpu:v1 \
  -f Dockerfile .
```

### Argumentos de construcción

| Argumento | Valor por defecto | Descripción |
| --- | --- | --- |
| `BASE_IMAGE` | Tag PyTorch 2.7.1/CUDA 12.8/Ubuntu 22.04 x86_64 | Imagen base de ModelArts para PyTorch + CUDA. Debe ser compatible con el driver del host. |
| `BASE_PYTHON` | vacío (autodetección) | Intérprete con `torch`. Solo hace falta si la autodetección falla. |
| `KERNEL_NAME` | `ml-gpu` | Nombre técnico del kernel de Jupyter. |
| `INSTALL_EXTRA_PACKAGES` | `true` | Instala `ipykernel` y `EXTRA_PYTHON_PACKAGES`. |
| `EXTRA_PYTHON_PACKAGES` | `pandas scipy scikit-learn matplotlib tqdm` | Paquetes adicionales (separados por espacios). |
| `PIP_INDEX_URL` | Espejo de Huawei Cloud | Índice de pip. |
| `PIP_TRUSTED_HOST` | `mirrors.huaweicloud.com` | Host de confianza para pip. |
| `INSTALL_DEV_TOOLS` | `false` | Instala herramientas de desarrollo con `apt`/`dnf`/`yum`. |
| `EXPECTED_ARCH` | `x86_64` | Arquitectura esperada; el build falla si no coincide. |
| `ENSURE_MA_USER` | `true` | Asegura `ma-user`/`ma-group` si no existen en la base. |

> **No cambies la versión de `torch`/`CUDA`** sin verificar compatibilidad con el driver del host. Reinstalarlos puede romper la compatibilidad.

## 3. Verificar antes de subir

Sin GPU (por ejemplo en tu portátil), `nvidia-smi` y `torch.cuda.is_available()` fallarán: es lo esperado. Los pasos de detección de Python/torch deben pasar.

```bash
# Arquitectura
docker run --rm <imagen> uname -m

# Intérprete detectado
docker run --rm <imagen> gpu-python -c "import sys; print(sys.executable)"

# Versiones
docker run --rm <imagen> gpu-python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

En una máquina con GPU NVIDIA puedes hacer la comprobación completa:

```bash
docker run --rm --gpus all <imagen> bash /home/ma-user/verify-gpu.sh
```

## 4. Subir a SWR y registrar en ModelArts

1. Inicia sesión en SWR con el comando que genera la consola (**Dashboard > Generate Login Command**).
2. Sube la imagen:

```bash
docker push swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-gpu:v1
```

3. En ModelArts, ve a **Asset Management > Images > Registered Images > Register** y selecciona la imagen de SWR.
   - **Architecture**: `x86_64`
   - **Type**: `GPU` (o según la clasificación de ModelArts para imágenes con CUDA)
4. Crea el Notebook con **Custom Images**, selecciona la imagen registrada y **WebIDE = JupyterLab**.
5. Al abrir JupyterLab, selecciona el kernel **Python X.Y (ml-gpu, GPU)**.

## 5. Comprobar la GPU en el Notebook

Abre una terminal dentro del Notebook y ejecuta:

```bash
bash /home/ma-user/verify-gpu.sh
```

Ejemplo mínimo en un notebook:

```python
import torch

print(torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Dispositivos:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024).cuda()
    y = x @ x
    print(y.device, y.sum().item())
```

## Solución de problemas

| Síntoma | Causa habitual | Solución |
| --- | --- | --- |
| `torch.cuda.is_available()` es `False` | Sin GPU asignada o driver no montado | Revisa la especificación del Notebook y `nvidia-smi`. |
| `nvidia-smi` no encontrado | Driver del host no expuesto | Problema del host/plataforma, no de la imagen. |
| Errores CUDA/cuDNN | Driver/CUDA incompatibles con la base | Usa una imagen base `atelier/pytorch_cuda` compatible con el driver del host. |
| Registro falla por arquitectura | Imagen construida en otro arch | Construye en x86_64. |
| JupyterLab no abre | Archivos estáticos de Jupyter en la imagen | No instales JupyterLab; revisa `jupyter --path` y elimina directorios estáticos. |

## Notas

- **No agregues el driver NVIDIA a la imagen.** Es responsabilidad del host.
- **Usa imagen base de ModelArts.** Parte de `atelier/pytorch_cuda` para garantizar compatibilidad con el entorno de ModelArts.
- **Respeta ma-user/ma-group.** Requeridos por el escenario Notebook.
