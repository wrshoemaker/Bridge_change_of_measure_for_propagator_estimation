#!/bin/bash
# ---------------------------------------------------------------------------
# Put the SILVA v138.2 DADA2-formatted reference files where the R scripts
# expect them. Run on the INTI FRONTEND (compute nodes may not have outbound
# internet; Topaze compute nodes definitely do not).
#
#     bash fetch_silva.sh
#
# Default target is the sequence_data root, where the R scripts look for them.
# If the two files are already there, this is a no-op.
#
# Alternative (no download): you already have both files locally under
#   data/takayasu_et_al/ -- just rsync them up instead:
#     rsync -av data/takayasu_et_al/silva_*.fa.gz \
#           inti:/env/cns/bigtmp2/William/Path_inference_with_poissonian_sampling_noise/Data/sequence_data/
# ---------------------------------------------------------------------------
set -euo pipefail

REF_DIR="${1:-/env/cns/bigtmp2/William/Path_inference_with_poissonian_sampling_noise/Data/sequence_data}"
mkdir -p "$REF_DIR"
cd "$REF_DIR"

BASE="https://zenodo.org/records/14169026/files"
for f in silva_nr99_v138.2_toGenus_trainset.fa.gz silva_v138.2_assignSpecies.fa.gz; do
  if [[ -s "$f" ]]; then
    echo "[silva] $f already present, skipping"
  else
    echo "[silva] downloading $f"
    curl -fL -o "$f" "$BASE/$f?download=1"
  fi
done

ls -lh "$REF_DIR"
echo "[silva] set SILVA_TRAIN=$REF_DIR/silva_nr99_v138.2_toGenus_trainset.fa.gz"
echo "[silva]     SILVA_SPECIES=$REF_DIR/silva_v138.2_assignSpecies.fa.gz"
