#!/usr/bin/env Rscript
# ===========================================================================
# DADA2 -- Takayasu et al. gut 16S  (PAIRED-END, 2 x 301 bp MiSeq, V3-V4)
#
# Trimming for this study:  truncLen = c(280, 220), minLen = 200, minOverlap = 20
#   The reverse read degrades hard after ~220 bp on 2x300 MiSeq; 280 + 220 still
#   leaves ~40 bp of overlap on a ~460 bp V3-V4 amplicon. Look at
#   quality_profile_raw.pdf before trusting these numbers -- if merge rates come
#   out low in track_reads.txt, lengthen truncLen or lower minOverlap.
#   Point FASTQ_DIR at data/takayasu_et_al/fastq_trimmed (primers already
#   removed) and leave TRIM_LEFT=0,0. If you use the raw fastq/ instead, set
#   TRIM_LEFT to the primer lengths, e.g. TRIM_LEFT=17,21 for 341F/805R.
#
# Run via sbatch_takayasu.sh. Configuration comes from environment variables:
#   FASTQ_DIR      directory holding *_1.fastq.gz / *_2.fastq.gz  (required)
#   OUT_DIR        where the seqtab/taxa tables go                (required)
#   SILVA_TRAIN    silva_nr99_v138.2_toGenus_trainset.fa.gz
#   SILVA_SPECIES  silva_v138.2_assignSpecies.fa.gz   (optional, "" to skip)
#   TMP_DIR        scratch for filtered reads (default: FASTQ_DIR/filtered)
#   TRIM_LEFT      bases clipped at the 5' end, "fwd,rev"    (default 0,0)
#   DADA_POOL      TRUE | FALSE | pseudo  (default pseudo: ~4500 samples)
# ===========================================================================
suppressPackageStartupMessages({ library(dada2) })

# -- resources ---------------------------------------------------------------
# Never let dada2 self-detect cores: on a shared node it would grab the whole
# machine. Slurm tells us exactly what we were given.
ncores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1"))
Sys.setenv(OMP_NUM_THREADS = ncores)
if (requireNamespace("RcppParallel", quietly = TRUE))
  RcppParallel::setThreadOptions(numThreads = ncores)

need <- function(v) { x <- Sys.getenv(v); if (!nzchar(x)) stop(v, " is not set"); x }
fastq_dir <- need("FASTQ_DIR")
out_dir   <- need("OUT_DIR")
silva     <- need("SILVA_TRAIN")
species   <- Sys.getenv("SILVA_SPECIES")
tmp_dir   <- Sys.getenv("TMP_DIR", file.path(fastq_dir, "filtered"))
pool_opt  <- Sys.getenv("DADA_POOL", "pseudo")
trim_left <- as.integer(strsplit(Sys.getenv("TRIM_LEFT", "0,0"), ",")[[1]])
pool      <- if (pool_opt == "pseudo") "pseudo" else as.logical(pool_opt)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

cat("dada2", as.character(packageVersion("dada2")), "| cores:", ncores,
    "| pool:", pool_opt, "\n")

# -- input -------------------------------------------------------------------
fnFs <- sort(list.files(fastq_dir, pattern = "_1\\.fastq\\.gz$", full.names = TRUE))
fnRs <- sort(list.files(fastq_dir, pattern = "_2\\.fastq\\.gz$", full.names = TRUE))
sample.names <- sapply(strsplit(basename(fnFs), "_"), `[`, 1)
stopifnot(length(fnFs) > 0,
          identical(sample.names, sapply(strsplit(basename(fnRs), "_"), `[`, 1)))
cat("sample pairs:", length(fnFs), "\n")

pdf(file.path(out_dir, "quality_profile_raw.pdf"), width = 8, height = 5)
k <- seq_len(min(4, length(fnFs)))
print(plotQualityProfile(fnFs[k])); print(plotQualityProfile(fnRs[k]))
dev.off()

# -- filter & trim -----------------------------------------------------------
filtFs <- file.path(tmp_dir, paste0(sample.names, "_1_filt.fastq.gz"))
filtRs <- file.path(tmp_dir, paste0(sample.names, "_2_filt.fastq.gz"))
names(filtFs) <- names(filtRs) <- sample.names

out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs,
                     truncLen = c(280, 220), minLen = 200, trimLeft = trim_left,
                     maxN = 0, maxEE = c(2, 5), truncQ = 2, rm.phix = TRUE,
                     compress = TRUE, multithread = ncores)
print(head(out))
write.table(out, file.path(out_dir, "filter_stats.txt"), sep = "\t", quote = FALSE)

keep   <- file.exists(filtFs) & file.exists(filtRs)
filtFs <- filtFs[keep]; filtRs <- filtRs[keep]
if (!length(filtFs)) stop("every sample was filtered out -- loosen truncLen/maxEE")

# -- error model -------------------------------------------------------------
errF <- learnErrors(filtFs, multithread = ncores)
errR <- learnErrors(filtRs, multithread = ncores)
pdf(file.path(out_dir, "error_model.pdf"), width = 8, height = 6)
print(plotErrors(errF, nominalQ = TRUE)); print(plotErrors(errR, nominalQ = TRUE))
dev.off()

# -- sample inference & merge ------------------------------------------------
dadaFs <- dada(filtFs, err = errF, multithread = ncores, pool = pool)
dadaRs <- dada(filtRs, err = errR, multithread = ncores, pool = pool)

mergers <- mergePairs(dadaFs, filtFs, dadaRs, filtRs,
                      minOverlap = 20, maxMismatch = 0, verbose = TRUE)

seqtab <- makeSequenceTable(mergers)
cat("seqtab:", dim(seqtab), "\n")
print(table(nchar(getSequences(seqtab))))

seqtab.nochim <- removeBimeraDenovo(seqtab, method = "consensus",
                                    multithread = ncores, verbose = TRUE)
cat("nochim:", dim(seqtab.nochim), " reads kept:",
    sum(seqtab.nochim) / sum(seqtab), "\n")
saveRDS(seqtab.nochim, file.path(out_dir, "seqtab_nochim.rds"))

# -- read tracking -----------------------------------------------------------
getN  <- function(x) sum(getUniques(x))
sn    <- names(mergers)
track <- cbind(out[keep, , drop = FALSE],
               denoisedF = sapply(dadaFs, getN)[sn],
               denoisedR = sapply(dadaRs, getN)[sn],
               merged    = sapply(mergers, getN)[sn],
               nonchim   = rowSums(seqtab.nochim)[sn])
write.table(track, file.path(out_dir, "track_reads.txt"), sep = "\t", quote = FALSE)
print(head(track))

# -- taxonomy ----------------------------------------------------------------
rm(dadaFs, dadaRs, mergers, seqtab); invisible(gc())
taxa <- assignTaxonomy(seqtab.nochim, silva, multithread = ncores)
if (nzchar(species)) taxa <- addSpecies(taxa, species)
saveRDS(taxa, file.path(out_dir, "taxa.rds"))

# -- export ------------------------------------------------------------------
write.table(t(seqtab.nochim), file.path(out_dir, "seqtab-nochim-gut.txt"),
            sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
write.table(taxa, file.path(out_dir, "seqtab-nochim-taxa-gut.txt"),
            sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
cat("done ->", out_dir, "\n")
