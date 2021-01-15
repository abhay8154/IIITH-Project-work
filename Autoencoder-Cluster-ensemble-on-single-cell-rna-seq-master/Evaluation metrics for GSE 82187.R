#evaluation datasets

eval_clusters<-ensemble_cluster(eval) # clusters obtained for GSE82187 dataset (for 754 genes)
table(eval_clusters[[5]]) # tabular count of all cell clusters

# ARI (Adjusted Rand Index)

adjustedRandIndex(eval_clusters[[1]],cell_types)
adjustedRandIndex(eval_clusters[[2]],cell_types)
adjustedRandIndex(eval_clusters[[3]],cell_types)
adjustedRandIndex(eval_clusters[[4]],cell_types)
adjustedRandIndex(eval_clusters[[5]],cell_types)

# NMI

NMI(eval_clusters[[5]],cdf)

# FM & Jaccard

cdf<-as.factor(cell_types)
cdf
summary(cdf)
table(cell_types)
num_cdf<-as.numeric(cdf)
table(num_cdf)
class(num_cdf)
class(eval_clusters[[1]])
EN1<-as.numeric(eval_clusters[[5]])
EN1<-as.integer(EN1)
NUM_IN<-as.integer(num_cdf)

#Jaccard and Folkes Mallows Index 

extCriteria(EN1,NUM_IN,c("Jaccard","Folkes_Mallows"))
NUM_IN

#For original dataset (without any drop out of genes)
evaldat2<-t(evaldat2)
eval_orig<-ensemble_cluster(evaldat2)
