#!/usr/bin/env bash
# One-time setup on a Linux GPU server (Ubuntu), BEFORE the paid training starts. It checks everything that could
# otherwise fail hours later:
#   1. the NVIDIA driver is installed (nvidia-smi works)
#   2. PyTorch matches the driver: the default PyTorch from pip needs a recent driver (CUDA 13 -> driver >= 580);
#      with an older driver the matching PyTorch build is taken from download.pytorch.org instead
#   3. the GPU really computes in float16 (the precision used for training)
#   4. BERT can be downloaded from Hugging Face (and is cached for the run)
#
#   bash install_server.sh          # inside maib-bert-pretraining-server/, as root or with sudo rights
set -euo pipefail
cd "$(dirname "$0")"

# ------------------------------------------------------------------ 1. driver
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA driver (nvidia-smi not found). Install it, reboot, log in again and re-run this script:"
  echo "  apt update && apt install -y ubuntu-drivers-common && ubuntu-drivers autoinstall && reboot"
  exit 1
fi
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
major=${driver%%.*}
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1) | driver $driver"

# PyTorch build for this driver
if   [ "$major" -ge 580 ]; then index=""                                              # default pip build (CUDA 13)
elif [ "$major" -ge 570 ]; then index="https://download.pytorch.org/whl/cu128"
elif [ "$major" -ge 525 ]; then index="https://download.pytorch.org/whl/cu126"
else echo "Driver $driver is too old (need >= 525). Run: ubuntu-drivers autoinstall && reboot"; exit 1
fi

# ------------------------------------------------------------------ 2. Python environment
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
if command -v apt-get >/dev/null 2>&1; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3-venv python3-pip tmux unzip rsync >/dev/null
fi
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
if [ -n "$index" ]; then
  echo "Installing PyTorch for driver $driver from $index"
  pip install -q torch --index-url "$index"
fi
pip install -q -r requirements.txt                 # torch is already satisfied if it was installed above

# ------------------------------------------------------------------ 3 + 4. checks
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch does not see the GPU (driver / PyTorch mismatch)."
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); (x @ x).sum().item()
print(f"[ok] torch {torch.__version__} (CUDA {torch.version.cuda}) computes on {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
from transformers import AutoModelForMaskedLM, AutoTokenizer
AutoTokenizer.from_pretrained("bert-base-uncased"); AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
print("[ok] bert-base-uncased downloaded from Hugging Face")
EOF
for f in data/test.csv corpus/external_train.txt corpus/task_train.txt reports/data_summary.json; do
  [ -f "$f" ] && echo "[ok] $f" || echo "[missing] $f  (the pipeline would have to rebuild it)"
done
echo "Ready. Start the run in tmux (README, section 2-c)."
