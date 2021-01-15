# Optimization dataset GSE60361 Zeisel et al

library(Biobase)
library(GEOquery)

# For each dataset, genes detected in less than 20% of cells were removed.

y<-getGEOSuppFiles("GSE60361",makeDirectory = FALSE)
gzFile <- grep("Expression", basename(rownames(y)), value=TRUE)
txtFile <- gsub(".gz", "", gzFile)
gunzip(gzFile, destname=txtFile, remove=TRUE)

library(data.table)
geoData <- fread(txtFile, sep="\t")
geneNames <- unname(unlist(geoData[,1, with=FALSE]))
exprMatrix <- as.matrix(geoData[,-1, with=FALSE])
dim(exprMatrix)
rownames(exprMatrix) <- geneNames
exprMatrix <- exprMatrix[unique(rownames(exprMatrix)),]
exprMatrix[1:5,1:4]
exprMatrix<-t(exprMatrix)
m<-exprMatrix  # genes as rows and cells as columns
dim(m)
zero_count<-rowSums(m==0)
class(zero_count)
zero_count<-as.matrix(zero_count)
m<-m[rowSums(m==0)<601,]  # remove genes having less than 20% expression in all cells (3005 cells)
m<-t(m) # cells as rows and genes as columns
write.csv(m,file="m.csv")

readFormat <- function(infile) { 
  # First column is empty.
  metadata <- read.delim(infile, stringsAsFactors=FALSE, header=FALSE, nrow=10)[,-1] 
  rownames(metadata) <- metadata[,1]
  metadata <- metadata[,-1]
  metadata <- as.data.frame(t(metadata))
  
  # First column after row names is some useless filler.
  counts <- read.delim(infile, stringsAsFactors=FALSE, 
                       header=FALSE, row.names=1, skip=11)[,-1] 
  counts <- as.matrix(counts)
  return(list(metadata=metadata, counts=counts))
}

# Using this function (readFormat), we read in the counts for ERCC spike-in transcripts and store it as spike.data

spike.data <- readFormat("expression_spikes_17-Aug-2014.txt")

my_labels<-spike.data$metadata$`group #`  # ground truth labels for cell clusters
my_lables<-as.numeric(my_labels)
class(my_lables)