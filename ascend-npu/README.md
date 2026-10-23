# Imagen personalizada NPU Ascend para ModelArts Notebook

> Índice del repositorio: [`../README.md`](../README.md)

`Dockerfile` para crear una imagen de Notebook con **NPU Ascend** (CANN +
`torch_npu`) siguiendo la guía oficial de Huawei Cloud
[Creating a Custom Image on ECS and Using It](https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_0011-0.html)
y el ejemplo
[Creating a Custom Image for Training (PyTorch + Ascend)](https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_6031.html).

## Driver frente a CANN: qué va en la imagen y qué pone el host

Este es el punto que más confusión genera. Una imagen de NPU **no** lleva el
driver dentro:

| Componente | Dónde vive | ¿Se instala en la imagen? |
| --- | --- | --- |
| Driver NPU (`Ascend-hdk-*-npu-driver`) y firmware | Host de ModelArts; se monta en `/usr/local/Ascend/driver` y expone `/dev/davinci*` | **No** |
| CANN toolkit (runtime, operadores, compilador) | `/usr/local/Ascend/ascend-toolkit/latest` | **Sí** (viene en la imagen base) |
| `torch` + `torch_npu` | Entorno Python de la imagen | **Sí** (viene en la imagen base) |
| `ipykernel` y kernel de Jupyter | `/home/ma-user/.local/share/jupyter` | **Sí** (lo añade este Dockerfile) |

El driver son módulos de kernel y librerías del host: no puede ejecutarse dentro
de un contenedor y, si se instala en la imagen, queda inservible o entra en
conflicto con el que monta ModelArts. Lo que sí debe llevar la imagen es **CANN
compatible con la versión del driver del host**, y eso es lo que aporta la
imagen base de ModelArts.

> Al registrar la imagen en ModelArts el **Type** debe ser **NPU** (lo determina
> la presencia de CANN) y la **Specification** debe coincidir con el modelo del
> host, que aparece en el tag de la imagen (`snt9b`, `snt9b23`, `snt3p`).

## Qué incluye

- Base **aarch64** de ModelArts para Ascend, con **CANN 8.5.2**, Python 3.12
  (3.11 en la base de `snt3p`) y
  `torch` 2.7.1 + `torch_npu` 2.7.1 ya instalados y compatibles entre sí.
- Detección automática del `set_env.sh` de CANN y del intérprete Python que ya
  tiene `torch` + `torch_npu` (no se reinstalan, para no romper la compatibilidad
  con CANN).
- `ipykernel` y kernel de Jupyter registrado (`ml-npu` por defecto), requisito
  del escenario Notebook.
- Paquetes habituales (`pandas`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`)
  instalados con los espejos de Huawei Cloud.
- Envoltorio `npu-python` en `/usr/local/bin` que apunta al intérprete con
  `torch` + `torch_npu` (la ruta también queda en `/home/ma-user/.npu-python`).
- `/etc/profile.d/00-ascend.sh` con el entorno CANN y las rutas del driver
  montado (`/usr/local/Ascend/driver/lib64*`).
- Script `verify-npu.sh` para comprobar en el Notebook que el NPU está visible.
- Sin JupyterLab/Jupyter Notebook propios, para no chocar con los archivos
  estáticos que inyecta ModelArts.
- Sin CUDA: esta imagen es solo NPU.

## 1. Elegir la imagen base

El tag incluye la versión de CANN, el SO, la arquitectura y el modelo del host.
Sustituye `swr.<region>.myhuaweicloud.com` por tu región (por ejemplo
`swr.cn-north-4.myhuaweicloud.com`).

| Modelo | SO | Tag de `atelier/pytorch_ascend` |
| --- | --- | --- |
| `snt9b` | EulerOS 2.10.11 | `pytorch_2.7.1-cann_8.5.2-py_3.12-euler_2.10.11-aarch64-snt9b-20260417112518-aabfd52` |
| `snt9b` | HCE 2.0.2512 | `pytorch_2.7.1-cann_8.5.2-py_3.12-hce_2.0.2512-aarch64-snt9b-20260417112518-aabfd52` |
| `snt9b23` | HCE 2.0.2512 | `pytorch_2.7.1-cann_8.5.2-py_3.12-hce_2.0.2512-aarch64-snt9b23-20260417112518-aabfd52` |
| `snt3p` | HCE 2.0.2509 | `pytorch_2.7.1-cann_8.3.rc1-py_3.11-hce_2.0.2509-aarch64-snt3p-20251205091605-e41006e` |

El valor por defecto del `Dockerfile` es el de `snt9b` con EulerOS. Si tu
Notebook es `snt9b23` o `snt3p`, pasa `--build-arg BASE_IMAGE=...` con el tag
correspondiente; de lo contrario ModelArts rechazará el registro por
especificación incompatible.

## 2. Construir la imagen

La imagen es **aarch64**: constrúyela en una ECS ARM64 (Kunpeng) con Docker, no
en un portátil x86.

```bash
cd ascend-npu

docker build \
  -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-ascend:v1 \
  -f Dockerfile .
```

Con otro modelo de NPU:

```bash
docker build \
  --build-arg BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/atelier/pytorch_ascend:pytorch_2.7.1-cann_8.5.2-py_3.12-hce_2.0.2512-aarch64-snt9b23-20260417112518-aabfd52 \
  -t swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-ascend:snt9b23-v1 \
  -f Dockerfile .
```

### Argumentos de construcción

| Argumento | Valor por defecto | Descripción |
| --- | --- | --- |
| `BASE_IMAGE` | Tag `snt9b` + EulerOS 2.10.11 | Imagen base de ModelArts para Ascend. Debe coincidir con el modelo y el SO admitido. |
| `BASE_PYTHON` | vacío (autodetección) | Intérprete con `torch` + `torch_npu`. Solo hace falta si la autodetección falla. |
| `KERNEL_NAME` | `ml-npu` | Nombre técnico del kernel de Jupyter. |
| `INSTALL_EXTRA_PACKAGES` | `true` | Instala `ipykernel` y `EXTRA_PYTHON_PACKAGES`. |
| `EXTRA_PYTHON_PACKAGES` | `pandas scipy scikit-learn matplotlib tqdm` | Paquetes adicionales (separados por espacios). |
| `PIP_INDEX_URL` | Espejo de Huawei Cloud | Índice de pip. |
| `PIP_TRUSTED_HOST` | `mirrors.huaweicloud.com` | Host de confianza para pip. |
| `INSTALL_DEV_TOOLS` | `false` | Instala `gcc`, `ffmpeg`, `git`, etc. con `dnf`/`yum`. La base de ModelArts ya los trae. |
| `EXPECTED_ARCH` | `aarch64` | Arquitectura esperada; el build falla si no coincide. |

> **No cambies la versión de `torch` ni de `torch_npu`** en esta imagen. El par
> de la imagen base está verificado contra su CANN; reinstalarlos es la causa
> habitual del error de ModelArts
> `the used cann and CUDA environments are not compatible`.

## 3. Verificar antes de subir

Sin NPU (por ejemplo en tu portátil o en la ECS de build), el script
`verify-npu.sh` fallará en los bloques de driver y de dispositivo: es lo
esperado. Los bloques de CANN, `torch` y `torch_npu` deben pasar.

```bash
# Arquitectura: aarch64
docker run --rm <imagen> uname -m

# Intérprete detectado (con torch + torch_npu)
docker run --rm <imagen> npu-python -c "import sys; print(sys.executable)"

# Versiones de torch y torch_npu
docker run --rm <imagen> npu-python -c \
  "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"

# CANN presente
docker run --rm <imagen> bash -lc \
  'source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh && echo $ASCEND_TOOLKIT_HOME'

# Kernel de Jupyter registrado
docker run --rm <imagen> ls /home/ma-user/.local/share/jupyter/kernels/ml-npu
```

En una ECS con NPU puedes hacer la comprobación completa:

```bash
docker run --rm --device /dev/davinci0 --device /dev/davinci_manager \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  <imagen> bash /home/ma-user/verify-npu.sh
```

## 4. Subir a SWR y registrar en ModelArts

1. Inicia sesión en SWR con el comando que genera la consola
   (**Dashboard > Generate Login Command**).
2. Sube la imagen:

```bash
docker push swr.cn-north-4.myhuaweicloud.com/<organizacion>/modelarts-notebook-ascend:v1
```

3. En ModelArts, ve a **Asset Management > Images > Registered Images > Register**
   y selecciona la imagen de SWR.
   - **Architecture**: `aarch64`.
   - **Type**: `NPU` (la imagen contiene CANN).
   - **Specification**: el modelo del tag (`snt9b`, `snt9b23` o `snt3p`).
4. Crea el Notebook con **Custom Images**, selecciona la imagen registrada y
   **WebIDE = JupyterLab**.
5. Al abrir JupyterLab, selecciona el kernel
   **Python 3.12 (ml-npu, Ascend NPU)** (el nombre puede variar según la base).

## 5. Comprobar el NPU en el Notebook

Abre una terminal dentro del Notebook y ejecuta:

```bash
bash /home/ma-user/verify-npu.sh
```

Ejemplo mínimo en un notebook:

```python
import torch
import torch_npu

print(torch.__version__, torch_npu.__version__)
print("NPU disponible:", torch.npu.is_available())
print("Dispositivos:", torch.npu.device_count())

x = torch.randn(1024, 1024).npu()
y = x @ x
print(y.device, y.sum().item())
```

## Solución de problemas

| Síntoma | Causa habitual | Solución |
| --- | --- | --- |
| `torch.npu.is_available()` es `False` | El Notebook no tiene NPU asignada, o el driver del host no está montado | Revisa la especificación del Notebook y `ls /dev/davinci*`. |
| `libascend_hal.so: cannot open shared object file` | Falta `/usr/local/Ascend/driver` (no lo monta el host) | Es un problema del host/plataforma, no de la imagen. |
| `libascendcl.so` u otras librerías de CANN no encontradas | No se ha cargado el entorno CANN | `source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh` |
| `the used cann and CUDA environments are not compatible` | Se reinstaló `torch`/`torch_npu` o se mezcló CANN con CUDA | Usa el `torch_npu` de la imagen base y no instales CUDA. |
| El registro falla por especificación | El tag base no coincide con el modelo del Notebook | Reconstruye con el `BASE_IMAGE` del modelo correcto (`snt9b`, `snt9b23`, `snt3p`). |
| El registro falla por arquitectura | La imagen se construyó en x86 | Construye en una ECS ARM64. |
| `npu-smi: command not found` | El host no expone las herramientas del driver | Usa `/usr/local/Ascend/driver/tools/npu-smi` o revisa el montaje del driver. |
| JupyterLab no abre | Archivos estáticos de Jupyter en la imagen | No instales JupyterLab; revisa `jupyter --path` y elimina los directorios estáticos. |

## Notas

- **No agregues el driver a la imagen.** Si necesitas otra versión de driver,
  se cambia en el host de ModelArts, no en el contenedor.
- **No uses `ubuntu` como base para ARM.** ModelArts solo admite EulerOS
  2.8/2.9/2.10.x o HCE 2.0/3.0 en aarch64; por eso se parte de una imagen
  oficial de ModelArts.
- **CANN y driver deben ser compatibles.** La versión de CANN de la imagen debe
  estar soportada por el driver del host. Si actualizas CANN, verifica antes la
  matriz de compatibilidad de Ascend.
- **Tamaño de imagen.** La base con CANN + PyTorch ocupa varios GB; añadir
  paquetes con `EXTRA_PYTHON_PACKAGES` es barato comparado con reinstalar el
  stack.
- Si necesitas un entorno conda separado, créalo con `conda create --clone` del
  entorno que trae `torch_npu`, para conservar el par `torch`/`torch_npu`
  verificado.
