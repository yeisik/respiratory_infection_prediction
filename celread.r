# install the core bioconductor packages, if not already installed
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(version = "3.16")

# install additional bioconductor libraries, if not already installed
BiocManager::install("affy")  # Methods for Affymetrix Oligonucleotide Arrays
BiocManager::install("hgu133a2.db", type = "source")  # GSE1297: Platform_title = [HG-U133A]
BiocManager::install("hgu133a2cdf")


setwd("~/data")
cels = list.files("~/data")
print(length(cels))

library(affy)
library(hgu133a2.db)
setwd("~/data")

raw.data = ReadAffy(verbose = FALSE, filenames = cels, cdfname = "hgu133a2cdf")

data.rma.norm = rma(raw.data)

rma = exprs(data.rma.norm)

write.table(rma, file = "~/expression/GSE73072_expression.txt", quote = FALSE, sep = "\t")
