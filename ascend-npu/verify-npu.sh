#!/usr/bin/env bash
# Verifica el stack Ascend/NPU dentro de un contenedor ModelArts.
#
# Uso:
#   bash /home/ma-user/verify-npu.sh
#   NPU_PYTHON=/ruta/al/python bash /home/ma-user/verify-npu.sh
#
# Sin NPU (por ejemplo con `docker run` en un portátil) los bloques de driver y
# de dispositivo fallarán: es lo esperado. Los bloques de CANN, torch y
# torch_npu sí deben pasar.
#
# Código de salida: 0 si no hay fallos; 1 si algo esencial falla.

set -u

ok=0
warnings=0
failures=0

section() { printf '\n== %s ==\n' "$1"; }
pass()    { printf '[ OK ]   %s\n' "$1"; ok=$((ok + 1)); }
note()    { printf '[AVISO]  %s\n' "$1"; warnings=$((warnings + 1)); }
bad()     { printf '[FALLO]  %s\n' "$1"; failures=$((failures + 1)); }

section "Sistema"
printf 'Arquitectura : %s\n' "$(uname -m)"
printf 'Kernel       : %s\n' "$(uname -r)"
printf 'OS           : %s\n' "$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | tr -d '\"')"
if [ "$(uname -m)" = "aarch64" ]; then
  pass "arquitectura aarch64"
else
  note "la imagen no es aarch64; ModelArts espera aarch64 para NPU"
fi

section "CANN"
cann_env=""
for f in "${ASCEND_TOOLKIT_HOME:-}/set_env.sh" \
         /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
         /usr/local/Ascend/nnae/latest/set_env.sh \
         /usr/local/Ascend/*/latest/set_env.sh; do
  if [ -n "${f}" ] && [ -f "${f}" ]; then cann_env="${f}"; break; fi
done

if [ -n "${cann_env}" ]; then
  pass "set_env.sh encontrado: ${cann_env}"
  # shellcheck disable=SC1090
  set +u
  . "${cann_env}"
  set -u
  toolkit_home="${ASCEND_TOOLKIT_HOME:-$(dirname "${cann_env}")}"
  printf 'ASCEND_TOOLKIT_HOME : %s\n' "${toolkit_home}"
  for vf in "${toolkit_home}/version.cfg" "${toolkit_home}/version.info" \
            "${toolkit_home}/compiler/version.info"; do
    if [ -f "${vf}" ]; then
      printf 'Versión (%s):\n' "$(basename "${vf}")"
      sed 's/^/    /' "${vf}"
      break
    fi
  done
  printenv | grep -E '^ASCEND_' | sed 's/^/    /' || true
else
  bad "no se encontró set_env.sh de CANN en /usr/local/Ascend"
fi

section "Driver NPU (lo monta el host, no la imagen)"
driver_home="${ASCEND_DRIVER_HOME:-/usr/local/Ascend/driver}"
if [ -d "${driver_home}" ]; then
  pass "directorio del driver: ${driver_home}"
  if [ -f "${driver_home}/version.info" ]; then
    printf 'Versión del driver:\n'
    sed 's/^/    /' "${driver_home}/version.info"
  else
    note "no existe ${driver_home}/version.info"
  fi
else
  bad "no existe ${driver_home}; el host no ha montado el driver NPU"
fi

section "Dispositivos /dev/davinci*"
davinci_count=$(find /dev -maxdepth 1 -name 'davinci*' 2>/dev/null | wc -l)
if [ "${davinci_count}" -gt 0 ]; then
  pass "${davinci_count} dispositivo(s) davinci visible(s)"
  ls -l /dev/davinci* /dev/davinci_manager 2>/dev/null | sed 's/^/    /' || true
else
  bad "no hay /dev/davinci* (contenedor sin NPU o sin --device)"
fi

section "npu-smi"
npu_smi="$(command -v npu-smi 2>/dev/null || true)"
if [ -z "${npu_smi}" ] && [ -x "${driver_home}/tools/npu-smi" ]; then
  npu_smi="${driver_home}/tools/npu-smi"
fi
if [ -n "${npu_smi}" ]; then
  pass "npu-smi: ${npu_smi}"
  "${npu_smi}" info || note "npu-smi info devolvió un error"
else
  note "npu-smi no está en PATH ni en ${driver_home}/tools"
fi

section "PyTorch + torch_npu"
py="${NPU_PYTHON:-}"
if [ -z "${py}" ] && [ -f /home/ma-user/.npu-python ]; then
  py="$(cat /home/ma-user/.npu-python)"
fi
if [ -z "${py}" ]; then
  py="$(command -v python3 || true)"
fi

if [ -z "${py}" ]; then
  bad "no se encontró ningún intérprete de Python"
else
  pass "intérprete: ${py}"
  "${py}" - <<'PY'
import platform
import sys

try:
    import torch
except Exception as exc:
    print(f"[FALLO]  no se pudo importar torch: {exc}")
    sys.exit(1)

print(f"[ OK ]   torch {torch.__version__} (Python {platform.python_version()})")

try:
    import torch_npu
except Exception as exc:
    print(f"[FALLO]  no se pudo importar torch_npu: {exc}")
    sys.exit(1)

print(f"[ OK ]   torch_npu {torch_npu.__version__}")

if not torch.npu.is_available():
    print("[FALLO]  torch.npu.is_available() es False")
    sys.exit(1)

count = torch.npu.device_count()
print(f"[ OK ]   NPU disponible, dispositivos: {count}")
for i in range(count):
    name = torch.npu.get_device_name(i)
    print(f"         [{i}] {name}")

try:
    a = torch.randn(512, 512, device="npu")
    b = torch.randn(512, 512, device="npu")
    c = (a @ b).sum().item()
    torch.npu.synchronize()
    print(f"[ OK ]   matmul 512x512 en NPU (sum={c:.4f})")
except Exception as exc:
    print(f"[FALLO]  no se pudo ejecutar matmul en NPU: {exc}")
    sys.exit(1)
PY
  if [ $? -eq 0 ]; then
    ok=$((ok + 1))
  else
    failures=$((failures + 1))
  fi
fi

section "Resumen"
printf 'OK: %d | avisos: %d | fallos: %d\n' "${ok}" "${warnings}" "${failures}"

if [ "${failures}" -gt 0 ]; then
  cat <<'HINT'

Sugerencias:
  - torch.npu.is_available() == False: comprueba que el Notebook tenga NPU
    asignada y que el host exponga /dev/davinci*. Compara la versión de CANN de
    la imagen con el driver del host (npu-smi info).
  - 'libascend_hal.so: cannot open shared object file': el driver del host no
    está montado en /usr/local/Ascend/driver.
  - 'libascendcl.so' u otras librerías de CANN: falta source del set_env.sh;
    ejecuta: source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  - 'cann and CUDA environments are not compatible' al crear el Notebook: la
    imagen mezcla CANN incompatible o se reinstaló torch/torch_npu; usa la
    versión de torch_npu que trae la imagen base.
HINT
  exit 1
fi

exit 0
