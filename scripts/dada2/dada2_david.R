#!/usr/bin/env Rscript
# ===========================================================================
# DADA2 -- David et al. gut 16S  (SINGLE-END, short reads ~100 bp)
#
# Trimming for this study:  truncLen = 95, minLen = 80, trimLeft = 5
#
# Run via sbatch_david.sh. Configuration comes from environment variables:
#   FASTQ_DIR      directory holding *.fastq.gz          (required)
#   OUT_DIR        where the seqtab/taxa tables go       (required)
#   SILVA_TRAIN    silva_nr99_v138.2_toGenus_trainset.fa.gz
#   SILVA_SPECIES  silva_v138.2_assignSpecies.fa.gz      (optional, "" to skip)
#   TMP_DIR        scratch for filtered reads (default: FASTQ_DIR/filtered)
#   DADA_POOL      TRUE | FALSE | pseudo   (default TRUE)
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
pool_opt  <- Sys.getenv("DADA_POOL", "TRUE")
pool      <- if (pool_opt == "pseudo") "pseudo" else as.logical(pool_opt)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

cat("dada2", as.character(packageVersion("dada2")), "| cores:", ncores,
    "| pool:", pool_opt, "\n")

# -- input -------------------------------------------------------------------
fn <- sort(list.files(fastq_dir, pattern = "\\.fastq\\.gz$", full.names = TRUE))
if (!length(fn)) stop("no .fastq.gz found in ", fastq_dir)
sample.names <- sub("\\.fastq\\.gz$", "", basename(fn))
cat("samples:", length(fn), "\n")

# Quality profiles -> PDF (batch jobs have no interactive device).
pdf(file.path(out_dir, "quality_profile_raw.pdf"), width = 8, height = 5)
print(plotQualityProfile(fn[seq_len(min(4, length(fn)))]))
dev.off()

# -- filter & trim -----------------------------------------------------------
filt <- file.path(tmp_dir, paste0(sample.names, "_filt.fastq.gz"))
names(filt) <- sample.names

out <- filterAndTrim(fn, filt,
                     truncLen = 95, minLen = 80, trimLeft = 5,
                     maxN = 0, maxEE = 2, truncQ = 10, rm.phix = TRUE,
                     compress = TRUE, multithread = ncores)
print(head(out))
write.table(out, file.path(out_dir, "filter_stats.txt"), sep = "\t", quote = FALSE)

filt <- filt[file.exists(filt)]
if (!length(filt)) stop("every sample was filtered out -- loosen truncLen/maxEE")

# -- error model -------------------------------------------------------------
err <- learnErrors(filt, multithread = ncores)
pdf(file.path(out_dir, "error_model.pdf"), width = 8, height = 6)
print(plotErrors(err, nominalQ = TRUE))
dev.off()

# -- sample inference --------------------------------------------------------
dd <- dada(filt, err = err, multithread = ncores, pool = pool)

seqtab <- makeSequenceTable(dd)
cat("seqtab:", dim(seqtab), "\n")
print(table(nchar(getSequences(seqtab))))

seqtab.nochim <- removeBimeraDenovo(seqtab, method = "consensus",
                                    multithread = ncores, verbose = TRUE)
cat("nochim:", dim(seqtab.nochim), " reads kept:",
    sum(seqtab.nochim) / sum(seqtab), "\n")
saveRDS(seqtab.nochim, file.path(out_dir, "seqtab_nochim.rds"))

# -- read tracking -----------------------------------------------------------
getN  <- function(x) sum(getUniques(x))
track <- cbind(out[rownames(out) %in% basename(fn), , drop = FALSE],
               denoised = sapply(dd, getN)[sample.names],
               nonchim  = rowSums(seqtab.nochim)[sample.names])
write.table(track, file.path(out_dir, "track_reads.txt"), sep = "\t", quote = FALSE)
print(head(track))

# -- taxonomy ----------------------------------------------------------------
rm(dd, seqtab); invisible(gc())
taxa <- assignTaxonomy(seqtab.nochim, silva, multithread = ncores)
if (nzchar(species)) taxa <- addSpecies(taxa, species)
saveRDS(taxa, file.path(out_dir, "taxa.rds"))

# -- export ------------------------------------------------------------------
write.table(t(seqtab.nochim), file.path(out_dir, "seqtab-nochim-gut.txt"),
            sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
write.table(taxa, file.path(out_dir, "seqtab-nochim-taxa-gut.txt"),
            sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
cat("done ->", out_dir, "\n")
