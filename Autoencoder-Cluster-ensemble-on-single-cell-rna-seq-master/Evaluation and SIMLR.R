# Evaluation Datasets 

library(SIMLR)
gse82187<-read.csv("GSE82187.csv")
class(gse82187)
neval<-colnames(gse82187)
head(neval)
evaldat<-gse82187
evaldat<-subset(evaldat,select=c(2,6:18845))
dim(evaldat)  # cells as rows and genes as columns
class(evaldat)
evaldat2 <- data.frame(evaldat[,-1], row.names=evaldat[,1])
evaldat2<-apply(evaldat2,1,as.numeric)
evaldat2<-t(evaldat2)   # genes as rows and cells as columns
zeros<-rowSums(evaldat2==0)  # count total number of cells where gene expression is zero
class(zeros)
zeros<-as.matrix(zeros)
cell_types<-gse82187$type
cell_types<-as.matrix(cell_types)
eval<-evaldat2[rowSums(evaldat2==0)<242,]  # Drop those genes which are expressed in less than 20% cells
dim(eval)
eval<-t(eval)
cells<-gse82187$cell.name
cells<-as.matrix(cells)
length(cells)

######################
#SIMLR
######################

########################### for original dataset
evaldat2<-t(evaldat2)
rownames(evaldat2)<-cells
help("SIMLR")
evaldat2<-t(evaldat2)
result<-SIMLR_Large_Scale(evaldat2,c=10)
result$S
rsiml<-SIMLR_Estimate_Number_of_Clusters(evaldat2,NUMC = 9:12)
rsiml$K1
rsiml$K2

plot(result$ydata,
     col = c(topo.colors(10))[NUM_IN],
     xlab = "SIMLR component 1",
     ylab = "SIMLR component 2",
     pch = 20,
     main="SIMILR 2D visualization")
nmi_2=compare(NUM_IN, result$y$cluster, method="nmi")
nmi_2

############################# for 754 genes
eval_t<-t(eval)
result_SIMLR<-SIMLR_Large_Scale(eval_t,c=10)
plot(result_SIMLR$ydata,
     col = c(topo.colors(10))[NUM_IN],
     xlab = "SIMLR component 1",
     ylab = "SIMLR component 2",
     pch = 20,
     main="SIMILR 2D visualization")
nmi_3=compare(NUM_IN, result_SIMLR$y$cluster, method="nmi")
nmi_3
#################### Jaccard and Folkes Mallows Index
#### For original dataset
extCriteria(result$y$cluster,NUM_IN,c("Jaccard","Folkes_Mallows"))
#### For 754 genes
extCriteria(result_SIMLR$y$cluster,NUM_IN,c("Jaccard","Folkes_Mallows"))
################### Adjusted Rand Index
### For 754 genes
adjustedRandIndex(result_SIMLR$y$cluster,NUM_IN)
### For original dataset
adjustedRandIndex(result$y$cluster,NUM_IN)
