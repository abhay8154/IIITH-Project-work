#Evaluation datasets
#labels_pbmc contains the ground truth labels for pbmc3k dataset
#pbmc3k dataset has 2000 differentially expressed genes and 2638 cells after Quality control analysis

#Creating counts matrix to call ensemble_cluster function and storing it in pbmc_counts

x<-pbmc@assays$RNA@var.features #filters variable feature/gene names from pbmc dataset
pbmc_counts<-pbmc[["RNA"]]@data #filters counts from pbmc dataset
pbmc_counts<-pbmc_counts[x,] #To subset only variable features from pbmc_counts
pbmc_counts<-as.matrix(pbmc_counts) #explicit conversion to matrix
pbmc_test<-pbmc

#Creating ground truth labels for cell types and storing them in pbmc_labels

new_idents<-c("1","2","3","4","5","6","7","8","9") 
names(new_idents) <- levels(pbmc_test)
pbmc_test<-RenameIdents(pbmc_test,new_idents) #rename cluster types from 0-8 to 1-9
pbmc_labels<-Idents(pbmc_test)
pbmc_labels<-as.numeric(pbmc_labels)

library(SIMLR)

result_SIMLR_pbmc<-SIMLR_Large_Scale(pbmc_counts,c=9)  # Applying SIMLR on pbmc3k dataset for clustering cells

plot(result_SIMLR_pbmc$ydata,
     col = c(topo.colors(9))[labels_pbmc],
     xlab = "SIMLR component 1",
     ylab = "SIMLR component 2",
     pch = 20,
     main="SIMILR 2D visualization")

# Adjusted Rand Index

adjustedRandIndex(SIMLR_labels,labels_pbmc)

# NMI (Normalized Mutual Information)

NMI(SIMLR_labels,labels_pbmc)

# Jaccard and Folkes Mallows Index

extCriteria(SIMLR_labels,NUM_IN,c("Jaccard","Folkes_Mallows"))



