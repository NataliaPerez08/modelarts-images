# Imágenes personalizadas para Huawei Cloud ModelArts

Dockerfiles para construir y registrar imágenes propias en ModelArts, con la
documentación de cada variante en su directorio.

Cada imagen es independiente: se construye en su propio directorio y se sube a
SWR por separado.

| Directorio | Tipo en ModelArts | Arquitectura | Base | Caso de uso |
| --- | --- | --- | --- | --- |
| [`cpu-only/`](cpu-only/README.md) | `CPU` | `x86_64` | `ubuntu:22.04` | Notebook de desarrollo sin acelerador: Python, Jupyter kernel, librerías científicas y PyTorch CPU opcional. |
| [`ascend-npu/`](ascend-npu/README.md) | `NPU` | `aarch64` | Imagen oficial `atelier/pytorch_ascend` (EulerOS/HCE) | Notebook con NPU Ascend: CANN, `torch_npu`, kernel de Jupyter y script de verificación. |

## Cuál usar

- **Sin acelerador** (exploración de datos, prototipos, CPUs): `cpu-only`.
- **Con NPU Ascend** (entrenamiento/inferencia en Atlas): `ascend-npu`.
- **Con GPU NVIDIA**: no hay variante en este repositorio; parte de una imagen
  `atelier/pytorch_cuda` de ModelArts y añade `ma-user`/`ma-group` si la base no
  los trae (ver la guía de Huawei Cloud en los enlaces de abajo).

## Flujo común

1. **Construir** la imagen en una ECS con Docker, de la **misma arquitectura**
   que el Notebook:

   ```bash
   # CPU (x86_64)
   cd cpu-only
   docker build -t swr.<region>.myhuaweicloud.com/<organizacion>/modelarts-notebook-cpu:v1 .

   # NPU Ascend (aarch64, ECS ARM64/Kunpeng)
   cd ascend-npu
   docker build -t swr.<region>.myhuaweicloud.com/<organizacion>/modelarts-notebook-ascend:v1 .
   ```

2. **Verificar** localmente antes de subir. Cada README tiene su lista de
   comprobaciones (arquitectura, usuario, kernel, framework).
   En `ascend-npu` además puedes ejecutar `verify-npu.sh` en un host con NPU.

3. **Subir a SWR**:

   ```bash
   docker push swr.<region>.myhuaweicloud.com/<organizacion>/<imagen>:<tag>
   ```

4. **Registrar en ModelArts** (`Asset Management > Images > Registered Images`)
   respetando arquitectura y tipo:

   | Imagen | Architecture | Type | Specification |
   | --- | --- | --- | --- |
   | `cpu-only` | `x86_64` | `CPU` | — |
   | `ascend-npu` | `aarch64` | `NPU` | Debe coincidir con el tag: `snt9b`, `snt9b23` o `snt3p` |

5. **Crear el Notebook** con **Custom Images**, la imagen registrada y
   **WebIDE = JupyterLab**.

## Reglas que aplican a todas las variantes

- **Usuario obligatorio**: `ma-user` (UID `1000`) y `ma-group` (GID `100`), con
  `/home/ma-user` en `750`.
- **Kernel de Jupyter**: el entorno que se use como kernel debe tener
  `ipykernel` instalado y registrado.
- **Sin JupyterLab propio**: ModelArts inyecta sus archivos estáticos; una
  instalación local de JupyterLab puede impedir que el Notebook abra.
- **Arquitectura**: la imagen se construye para la misma arquitectura del host
  (x86_64 o aarch64). No se puede mezclar.
- **NPU**: el driver Ascend lo aporta el host (`/usr/local/Ascend/driver`,
  `/dev/davinci*`); la imagen solo lleva CANN. No instales el driver dentro.
- **Región**: sustituye `<region>` por la tuya en las URLs de SWR
  (por ejemplo `cn-north-4`).

## Estructura

```
modelarts-images/
├── README.md
├── cpu-only/
│   ├── Dockerfile
│   └── README.md
└── ascend-npu/
    ├── Dockerfile
    ├── verify-npu.sh
    └── README.md
```

## Referencias

- [Creating a Custom Image on ECS and Using It](https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_0011-0.html)
- [ModelArts Unified Images](https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_6022.html)
- [Creating a Custom Image for Training (PyTorch + Ascend)](https://support.huaweicloud.com/intl/en-us/docker-modelarts/docker-modelarts_6031.html)
- [Faults of Custom Images on Notebook Instances](https://support.huaweicloud.com/intl/en-us/trouble-modelarts/docker-modelarts_6024.html)
