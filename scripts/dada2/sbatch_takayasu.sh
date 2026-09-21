#!/bin/bash
#SBATCH -J dada2_takayasu
#SBATCH -p xlarge
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem-per-cpu=15G
#SBATCH -t 48:00:00
#SBATCH -o logs/dada2_takayasu.%j.out
#SBATCH -e logs/dada2_takayasu.%j.err
##SBATCH --qos=long   # uncomment if 48h exceeds the default QoS limit
# ---------------------------------------------------------------------------
# DADA2 rerun -- Takayasu et al. gut 16S (paired-end 2x301, ~4500 samples)
#
#   mkdir -p logs && sbatch sbatch_takayasu.sh
#
# Memory is set explicitly on purpose: Inti has no sane default, and mixing
# --mem with --mem-per-cpu is a fatal submission error.
# 32 x 15G = 480 GiB on an xlarge node: ~4500 sample pairs is the big one.
# pool="pseudo" instead of TRUE keeps peak memory bounded while still
# sharing information across samples.
# ---------------------------------------------------------------------------
set -euo pipefail

# Where the sequence data lives: one subfolder per study, SILVA refs at the top.
DATA_ROOT="${DATA_ROOT:-/env/cns/bigtmp2/William/Path_inference_with_poissonian_sampling_noise/Data/sequence_data}"
STUDY_DIR="$DATA_ROOT/takayasu_et_al"
PROJECT="${PROJECT:-$HOME/GitHub/Bridge_change_of_measure_for_propagator_estimation}"

# Pick the fastq subfolder: set FASTQ_DIR yourself to override, otherwise take
# the first of these that exists, falling back to the study folder itself.
if [[ -z "${FASTQ_DIR:-}" ]]; then
  for d in fastq_trimmed fastq; do
    if [[ -d "$STUDY_DIR/$d" ]]; then FASTQ_DIR="$STUDY_DIR/$d"; break; fi
  done
  : "${FASTQ_DIR:=$STUDY_DIR}"
fi
export FASTQ_DIR
export OUT_DIR="${OUT_DIR:-$STUDY_DIR/dada2_rerun}"
export SILVA_TRAIN="${SILVA_TRAIN:-$DATA_ROOT/silva_nr99_v138.2_toGenus_trainset.fa.gz}"
export SILVA_SPECIES="${SILVA_SPECIES:-$DATA_ROOT/silva_v138.2_assignSpecies.fa.gz}"
# Filtered reads land next to the raw ones so a re-run can reuse them. Swap to
# /tmp/SLURM_$SLURM_JOB_ID/filtered (add --tmp=<size>) if shared I/O is the
# bottleneck -- node-local /tmp is wiped when the job ends.
export TMP_DIR="${TMP_DIR:-$FASTQ_DIR/filtered}"
export DADA_POOL="${DADA_POOL:-pseudo}"

# Fail now, not four hours in at assignTaxonomy.
for f in "$SILVA_TRAIN" "$SILVA_SPECIES"; do
  if [[ ! -s "$f" ]]; then echo "missing reference file: $f" >&2; exit 1; fi
done
if ! compgen -G "$FASTQ_DIR/*.fastq.gz" > /dev/null; then
  echo "no *.fastq.gz in $FASTQ_DIR" >&2; exit 1
fi

# Clean module state -- never inherit whatever ~/.bashrc happened to load.
module purge
module restore default 2>/dev/null || true
module load micromamba
# Make `micromamba activate` work in a non-interactive shell.
eval "$(micromamba shell hook --shell bash)"
# Do NOT `module load r` here: it conflicts with the R inside the env.
micromamba activate dada2_16s

echo "host=$(hostname) job=$SLURM_JOB_ID cores=$SLURM_CPUS_PER_TASK"
echo "fastq=$FASTQ_DIR"
echo "out=$OUT_DIR"

srun --cpu-bind=none Rscript "$PROJECT/scripts/dada2/dada2_takayasu.R"
