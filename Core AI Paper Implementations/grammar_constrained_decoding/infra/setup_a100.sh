#!/bin/bash
# setup_a100.sh — run ON the A100 host after fix_dns.sh. Idempotent: safe to rerun.
# Builds the shared env for the step-1 (loop/attention) and step-2 (XGrammar) experiments
# entirely under /mnt (keeps the root disk clean), then starts ONE shared vLLM server.
# Uses a Miniforge conda env under /mnt (own Python + headers); no apt, no system Python.
# Activate later with:  source /mnt/miniforge3/etc/profile.d/conda.sh && conda activate /mnt/conda/rtb-vlm
set -euo pipefail

MNT=/mnt
MODEL=Qwen/Qwen3-VL-8B-Instruct
export HF_HOME=$MNT/hf UV_CACHE_DIR=$MNT/uvcache PIP_CACHE_DIR=$MNT/pipcache TMPDIR=$MNT/tmp

# 1. Big volume must be mounted at /mnt
if ! mountpoint -q $MNT; then
  echo "ERROR: $MNT is not a mount point. Attach/mount the 2 TB volume first:"; lsblk -o NAME,SIZE,MOUNTPOINT; exit 1
fi
avail_gb=$(df -BG --output=avail $MNT | tail -1 | tr -dc 0-9)
[ "$avail_gb" -ge 60 ] || { echo "ERROR: only ${avail_gb} GB free on $MNT"; exit 1; }
sudo chown -R "$USER:$USER" $MNT 2>/dev/null || true
mkdir -p $MNT/{hf,data,work,logs,tmp,uvcache,pipcache}
rm -f $MNT/SETUP_DONE

# 2. Network sanity
timeout 6 getent hosts pypi.org >/dev/null || { echo "ERROR: no DNS — run fix_dns.sh first"; exit 1; }

# 3. Conda env (Miniforge under /mnt). Conda's Python ships its headers (Python.h),
#    which Triton needs to JIT-compile its CUDA helper; the system python3.12 lacks them.
#    Pip packages go INSIDE the conda env, installed with uv for speed and cache reuse.
CONDA=$MNT/miniforge3
ENV=$MNT/conda/rtb-vlm
if [ ! -x $CONDA/bin/conda ]; then
  curl -fsSL -o $MNT/tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash $MNT/tmp/miniforge.sh -b -p $CONDA
fi
export CONDA_PKGS_DIRS=$MNT/conda/pkgs
[ -x $ENV/bin/python ] || $CONDA/bin/conda create -y -p $ENV -c conda-forge python=3.12 pip
set +u; source $CONDA/etc/profile.d/conda.sh; conda activate $ENV; set -u
test -f "$(python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')/Python.h" \
  || { echo "ERROR: Python.h missing in conda env"; exit 1; }
if [ ! -x $MNT/uv/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$MNT/uv UV_NO_MODIFY_PATH=1 sh
fi
UV="$MNT/uv/uv pip install --python $ENV/bin/python"
$UV vllm --torch-backend=auto
$UV xgrammar pymupdf opencv-python-headless pandas openpyxl matplotlib \
    qwen-vl-utils accelerate rapidfuzz huggingface_hub
python - <<'EOF'
import importlib.metadata as md, torch
import vllm, xgrammar, transformers  # import check
print(" | ".join(f"{p} {md.version(p)}" for p in ("vllm", "xgrammar", "transformers", "torch")),
      "| cuda", torch.cuda.is_available())
EOF

# 4. Model weights (~17 GB)
hf download $MODEL >/dev/null

# 5. Shared vLLM server: half the GPU, so the attention run can load an HF copy alongside.
if ! curl -sf localhost:8010/v1/models >/dev/null; then
  nohup $ENV/bin/vllm serve $MODEL --port 8010 --served-model-name qwen3vl \
    --max-model-len 16384 --gpu-memory-utilization 0.50 --max-num-seqs 8 \
    --limit-mm-per-prompt '{"image": 1, "video": 0}' > $MNT/logs/vllm_serve.log 2>&1 &
fi
for i in $(seq 1 90); do curl -sf localhost:8010/v1/models >/dev/null && break; sleep 10; done
curl -sf localhost:8010/v1/models >/dev/null || { echo "ERROR: vLLM did not come up; see $MNT/logs/vllm_serve.log"; tail -30 $MNT/logs/vllm_serve.log; exit 1; }

# 6. Smoke test: one structured-output request (finds the request field this vLLM accepts)
python - <<'EOF'
import json, urllib.request
def call(extra):
    body = {"model": "qwen3vl", "max_tokens": 8, "temperature": 0,
            "messages": [{"role": "user", "content": "Say yes or no."}], **extra}
    req = urllib.request.Request("http://localhost:8010/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
g = 'root ::= "yes" | "no"'
for name, extra in [("structured_outputs", {"structured_outputs": {"grammar": g}}),
                    ("guided_grammar", {"guided_grammar": g})]:
    try:
        out = call(extra)
        print(f"grammar field '{name}' ->", repr(out))
        if out.strip() in ("yes", "no"):
            open("/mnt/GRAMMAR_FIELD", "w").write(name); break
    except Exception as e:
        print(f"grammar field '{name}' failed: {e}")
EOF

touch $MNT/SETUP_DONE
echo "SETUP_DONE  (grammar request field: $(cat $MNT/GRAMMAR_FIELD 2>/dev/null || echo unknown))"
