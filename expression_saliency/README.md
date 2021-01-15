# Deconvolution of autoencoders to learn biological regulatory modules from single cell mRNA sequencing data
PyTorch implementation of the autoencoder model used in the publication: 
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2952-9.

Train autoencoder:
```
python main.py PBMC
```
## Results obtained
<img src="Results/Figure_1 PBMC.png" width="800" height="400">

<img src="Results/Figure_1 Representation layer.png" width="800" height="400">
 
## Important points to note about the heatmap

1. In this scenario we followed a case specific approach and investigated the impact of hematopoiesis related signatures, derived from DMAP (Differential Methylation Analysis Package) on the representation layer (encoded layer) 

2. Row names correspond to cell type categories, or to DMAP labels for sub-classification.

3. In the heatmap, haematopoietic signatures are shown as rows and hidden units/neurons as columns. 

4. Colours are based on the impact of the genes in the signatures.

5. High proportion of red/blue colours show that those particular neurons/genes (columns) are most activated for that particular biological pathway/module.

6. One can note that node 34 is highly activated for CD8 T / CD4 T cells and NK cells. 

7. Both types of T cells and monocytes are different in their activity at node 34.

### Note:

Heatmap is stored in Results folder as Heatmap_PBMC.pdf
