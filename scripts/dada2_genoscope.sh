micromamba activate dada2_16s



jobify -i --mem=4G -c 4 bash -c "module load micromamba && micromamba activate dada2_16s && Rscript -e 'library(dada2); packageVersion(\"dada2\")'"