# DADA2 reprocessing on Inti (CNRGH / Genoscope)

New files here (the old `dada2_*_gut.R` scripts are untouched — they still carry
hardcoded `/Users/williamrshoemaker/...` paths from the old laptop).

| File | What it is |
|---|---|
| `env_dada2_16s.yml` | conda env spec — **conda-forge + bioconda only** |
| `install_dada2_inti.sh` | creates/updates the `dada2_16s` env, verifies dada2 loads |
| `fetch_silva.sh` | puts SILVA v138.2 reference files in `data/ref/` |
| `dada2_caporaso.R` · `dada2_david.R` · `dada2_poyet.R` · `dada2_takayasu.R` | one pipeline per study, per-study trimming |
| `sbatch_caporaso.sh` · … | one Slurm job per study, sized for that study |

## 1. Upload

```bash
rsync -av scripts/dada2/ inti:~/GitHub/Bridge_change_of_measure_for_propagator_estimation/scripts/dada2/
```

## 2. Install (once, on the frontend)

```bash
ssh inti
cd ~/GitHub/Bridge_change_of_measure_for_propagator_estimation/scripts/dada2
bash install_dada2_inti.sh
```

`defaults` and `r` (Anaconda) are licensed channels and are blocked
institute-wide, so the env pulls from conda-forge + bioconda only. The solve is
light under micromamba; if it ever gets OOM-killed on the frontend, rerun it
inside `sshell -c2 --mem-per-cpu=10G -t 01:00:00`.

## 3. Data layout on the cluster

Everything is anchored at

```
/env/cns/bigtmp2/William/Path_inference_with_poissonian_sampling_noise/Data/sequence_data
├── caporaso_et_al/   ← one subfolder per study
├── david_et_al/
├── poyet_et_al/
├── takayasu_et_al/
├── silva_nr99_v138.2_toGenus_trainset.fa.gz
└── silva_v138.2_assignSpecies.fa.gz
```

The jobs pick the fastq subfolder automatically (`fastq_gut`, then `fastq`,
then the study folder itself; `fastq_trimmed` first for Takayasu) and refuse to
start if the SILVA files or the fastqs aren't where they should be. Override
with `DATA_ROOT=` or `FASTQ_DIR=`.

If the SILVA files aren't up there yet: `bash fetch_silva.sh` downloads them to
that root from the frontend (compute nodes may have no outbound internet), or
just rsync the copies you already have under `data/takayasu_et_al/`.

## 4. Run

Starting with Caporaso:

```bash
cd ~/GitHub/Bridge_change_of_measure_for_propagator_estimation/scripts/dada2
mkdir -p logs
sbatch sbatch_caporaso.sh
squeue -u $USER
```

Then `sbatch_david.sh`, `sbatch_poyet.sh`, `sbatch_takayasu.sh` once you're
happy with how the first one ran.

Everything is overridable without editing the scripts:

```bash
FASTQ_DIR=/some/other/fastq OUT_DIR=/some/out DADA_POOL=pseudo sbatch sbatch_poyet.sh
```

Output lands in `<sequence_data>/<study>/dada2_rerun/`: `seqtab-nochim-gut.txt`,
`seqtab-nochim-taxa-gut.txt` (same format as the current files, so downstream
code doesn't change), plus `seqtab_nochim.rds`, `taxa.rds`, `track_reads.txt`,
`filter_stats.txt`, and PDF quality/error plots.

## Per-study trimming

| Study | Layout | Trim |
|---|---|---|
| Caporaso | single-end, ~150 bp | `truncLen=120`, `minLen=100`, `maxEE=2`, `truncQ=10` |
| David | single-end, ~100 bp | `truncLen=95`, `minLen=80`, `trimLeft=5`, `truncQ=10` |
| Poyet | paired, 2×~150 bp | `truncLen=c(145,145)`, `minLen=120`, `trimLeft=5`, `truncQ=10` |
| Takayasu | paired, 2×301 bp MiSeq V3–V4 | `truncLen=c(280,220)`, `minLen=200`, `maxEE=c(2,5)`, `truncQ=2`, `minOverlap=20` |

The first three reproduce the settings from the old scripts. Takayasu is new:
2×300 MiSeq reverse reads fall apart after ~220 bp, and 280+220 leaves ~40 bp of
overlap on a ~460 bp V3–V4 amplicon. **Check `quality_profile_raw.pdf` and the
`merged` column of `track_reads.txt` on a first run** — a low merge rate means
`truncLen` is too aggressive.

The job points at `fastq_trimmed/` (primers already removed, `TRIM_LEFT=0,0`).
To use raw `fastq/` instead, set the primer lengths, e.g.
`FASTQ_DIR=.../fastq TRIM_LEFT=17,21 sbatch sbatch_takayasu.sh` for 341F/805R.

## Cluster details baked into these scripts

- **Memory is always explicit** (`--mem-per-cpu`) — Inti has no safe default,
  and mixing `--mem` with `--mem-per-cpu` is a fatal submission error.
- **Threads come from `$SLURM_CPUS_PER_TASK`**, never `multithread=TRUE`.
  dada2's auto-detection sees every core on the node, not your allocation, and
  would oversubscribe a shared node and blow the memory budget.
- Jobs start from `module purge; module restore default`, so nothing from
  `~/.bashrc` leaks into a batch run.
- `module load r` is *not* used — it conflicts with the R inside the conda env
  and the module system will refuse the load.
- Takayasu runs on `xlarge` with `pool="pseudo"`: ~4500 sample pairs with
  `pool=TRUE` is the classic way to run a node out of memory. Switch to
  `DADA_POOL=TRUE` only if you have a reason to and the memory to match.
- If a conda-built binary dies with "illegal instruction" on an old node, add
  `-C avx2` to the sbatch line.
