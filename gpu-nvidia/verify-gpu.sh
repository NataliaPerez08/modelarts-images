#!/usr/bin/env bash
# Verifica el stack NVIDIA/CUDA + PyTorch dentro de un contenedor ModelArts.

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
if [ "$(uname -m)" = "x86_64" ]; then
  pass "arquitectura x86_64"
else
  note "la imagen no es x86_64; ModelArts espera x86_64 para GPU NVIDIA"
fi

section "Driver NVIDIA (lo monta el host)"
if command -v nvidia-smi >/dev/null 2>&1; then
  pass "nvidia-smi encontrado: $(command -v nvidia-smi)"
  nvidia-smi || note "nvidia-smi devolvió un error"
else
  bad "nvidia-smi no encontrado (host sin GPU/driver montado)"
fi

section "PyTorch + CUDA"
py="${GPU_PYTHON:-}"
if [ -z "${py}" ] && [ -f /home/ma-user/.gpu-python ]; then
  py="$(cat /home/ma-user/.gpu-python)"
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
print(f"         CUDA compilado: {torch.version.cuda}")
print(f"         cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}")
if not torch.cuda.is_available():
    print("[FALLO]  torch.cuda.is_available() es False")
    sys.exit(1)
count = torch.cuda.device_count()
print(f"[ OK ]   CUDA disponible, dispositivos: {count}")
for i in range(count):
    print(f"         [{i}] {torch.cuda.get_device_name(i)}")
try:
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"[ OK ]   matmul 512x512 en CUDA (sum={c:.4f})")
except Exception as exc:
    print(f"[FALLO]  no se pudo ejecutar matmul en CUDA: {exc}")
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
  - torch.cuda.is_available() == False: comprueba que el Notebook tenga GPU asignada y que nvidia-smi funcione.
  - CUDA/cuDNN incompatibles: usa una imagen base atelier/pytorch_cuda compatible con el driver del host.
  - Si registras la imagen, usa Type GPU/Architecture x86_64.
HINT
  exit 1
fi

exit 0
