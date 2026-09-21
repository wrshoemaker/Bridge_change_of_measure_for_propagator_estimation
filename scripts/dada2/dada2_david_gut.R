library(dada2); packageVersion("dada2")
#update("dada2")

library(usethis) 
usethis::edit_r_environ()

path <- "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/david_et_al/fastq_gut"
# Forward and reverse fastq filenames have format: SAMPLENAME_R1_001.fastq and SAMPLENAME_R2_001.fastq
fn <- sort(list.files(path, pattern=".fastq.gz", full.names = TRUE))
# Extract sample names, assuming filenames have format: SAMPLENAME_XXX.fastq
sample.names <- sapply(strsplit(basename(fn), ".fastq.gz"), `[`, 1)

plotQualityProfile(fn[1:2])

# Place filtered files in filtered/ subdirectory
filt <- file.path(path, "filtered", paste0(sample.names, "_filt.fastq.gz"))
names(filt) <- sample.names

out <- filterAndTrim(fn, filt, truncLen=95, minLen=80, trimLeft=5,
                     maxN=0, maxEE=c(2), truncQ=10, rm.phix=TRUE,
                     compress=TRUE, multithread=TRUE) # On Windows set multithread=FALSE

head(out)

# learn error rates
err <- learnErrors(filt, multithread=TRUE)

plotErrors(err, nominalQ=TRUE)

filt <- filt[file.exists(filt)]

# sample inference
dada <- dada(filt, err=err, multithread=TRUE, pool=TRUE)

dada[[1]]

# Merge paired reads
#mergers <- mergePairs(dada, filt, verbose=TRUE)
# Inspect the merger data.frame from the first sample
#head(mergers[[1]])

# Construct sequence table
seqtab <- makeSequenceTable(dada)
dim(seqtab)
# Inspect distribution of sequence lengths
table(nchar(getSequences(seqtab)))

# Remove chimeras
seqtab.nochim <- removeBimeraDenovo(seqtab, method="consensus", multithread=TRUE, verbose=TRUE)
dim(seqtab.nochim)
sum(seqtab.nochim)/sum(seqtab)



#Track reads through the pipeline
getN <- function(x) sum(getUniques(x))
track <- cbind(out, sapply(dada, getN), rowSums(seqtab.nochim))
# If processing a single sample, remove the sapply calls: e.g. replace sapply(dadaFs, getN) with getN(dadaFs)
colnames(track) <- c("input", "filtered", "denoised", "nonchim")
rownames(track) <- sample.names
head(track)




#Assign taxonomy
taxa <- assignTaxonomy(seqtab.nochim, "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/dada2/silva_nr99_v138.1_train_set.fa.gz", multithread=TRUE)

# make species level assignments based on exact matching between ASVs and sequenced reference strains
taxa <- addSpecies(taxa, "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/dada2/silva_species_assignment_v138.1.fa.gz")

taxa.print <- taxa # Removing sequence rownames for display only
rownames(taxa.print) <- NULL
head(taxa.print)


# export site-by-species
write.table(t(seqtab.nochim), "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/david_et_al/seqtab-nochim-gut.txt", sep="\t", row.names=TRUE, col.names=TRUE, quote=FALSE)
# export taxonomy
write.table(taxa, "/Users/williamrshoemaker/GitHub/coarse_grained_reproducibility/data/barcode_data/david_et_al/seqtab-nochim-taxa-gut.txt", sep="\t", row.names=TRUE, col.names=TRUE, quote=FALSE)
