#!/bin/bash
# ---------------------------------------------------------------------------
# Install the DADA2 environment on Inti (CNRGH / Genoscope).
#
# Run this ON THE INTI FRONTEND (ssh inti), not inside a batch job:
#     bash install_dada2_inti.sh
#
# The solve itself is light with micromamba, but if it ever gets OOM-killed on
# the frontend, redo it inside a small interactive allocation:
#     sshell -c2 --mem-per-cpu=10G -t 01:00:00
#     bash install_dada2_inti.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${ENV_NAME:-dada2_16s}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start from a clean module set so nothing from ~/.bashrc leaks in.
module purge
module restore default 2>/dev/null || true
module load micromamba

# micromamba is preferred over conda here: much faster and far less memory.
# Channels come from the yml (conda-forge + bioconda + nodefaults only).
if micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[install] env '$ENV_NAME' already exists -- updating it"
  micromamba env update -n "$ENV_NAME" -f "$HERE/env_dada2_16s.yml" -y
else
  echo "[install] creating env '$ENV_NAME'"
  micromamba create -n "$ENV_NAME" -f "$HERE/env_dada2_16s.yml" -y
fi

micromamba clean -t -y

# ---- verify -----------------------------------------------------------------
# Note: do NOT `module load r` while this env is active -- the module system
# will refuse the conflicting load (two R interpreters).
echo "[install] verifying..."
micromamba run -n "$ENV_NAME" Rscript -e '
  suppressMessages(library(dada2))
  cat("R       :", R.version.string, "\n")
  cat("dada2   :", as.character(packageVersion("dada2")), "\n")
  cat("ShortRead:", as.character(packageVersion("ShortRead")), "\n")
'
echo
echo "[install] done. Use it in a job with:"
echo "    module purge && module load micromamba && micromamba activate $ENV_NAME"
