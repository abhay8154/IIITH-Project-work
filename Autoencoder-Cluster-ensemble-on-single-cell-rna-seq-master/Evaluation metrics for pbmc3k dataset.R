#evaluation datasets
#labels_pbmc contains the ground truth labels for pbmc3k dataset
#pbmc3k dataset has 2000 differentialy expressed genes and 2638 cells after Quality control analysis

library(clusterCrit)
library(mclust)
library(aricode)

# ARI (Adjusted Rand Index)

adjustedRandIndex(eval_pbmc[[1]],labels_pbmc)
adjustedRandIndex(eval_pbmc[[2]],labels_pbmc)
adjustedRandIndex(eval_pbmc[[3]],labels_pbmc)
adjustedRandIndex(eval_pbmc[[4]],labels_pbmc)
adjustedRandIndex(eval_pbmc[[5]],labels_pbmc)

# NMI (Normalized Mutual Information)

NMI(eval_pbmc[[5]],labels_pbmc)

# FM & Jaccard

EN1<-as.numeric(eval_pbmc[[5]])  # change the list argument to analyze different ensemble sizes For e.g 1,2,3,4,5
EN1<-as.integer(EN1)   # convert to integer for compatibility with extCriteria() function.
NUM_IN<-as.integer(labels_pbmc) # convert to integer for compatibility with extCriteria() function.

#Jaccard and Folkes Mallows Index 

extCriteria(EN1,NUM_IN,c("Jaccard","Folkes_Mallows"))

