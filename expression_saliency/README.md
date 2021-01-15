# Deconvolution of autoencoders to learn biological regulatory modules from single cell mRNA sequencing data
PyTorch implementation of the autoencoder model used in the publication: 
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2952-9.

## Quickstart
Download the repository:
```
cd <projects_dir>
git clone https://gitlab.com/podro182/expression_saliency.git
cd expression_saliency
```

It's best to set up a conda environment for Python 3.7 and then install the dependencies with pip:
```
conda create --name expression_saliency python=3.7
conda activate expression_saliency
pip3 install -r requirements.txt
```

Now, we are able to start training our autoencoder:
```
python main.py Velten
```


## Some background
This is a PoC study on how we can make use of autoencoders for modeling single cell mRNA sequencing data and 
how we can use saliency maps to create a link between the trained model and known biological functions and..
as it turns out we can identify the biological functions associated with each representation unit.

Variants of autoencoders have been successfully applied to scRNA-seq expression data for dimensionality
reduction, denoising and imputation of missing values. In this study we ask the question: can we say anything
meaningful in terms of biology about the dimensionality reduced representation of the autoencoders?

What we did was to train an autoencoder on gene expression data with a soft orthogonality constraint on the representation layer, in order to disentagle the representation.
After training, we compute the impact that genes have in different representation units with a technique called saliency maps.
And we extend that, by aggregating impacts of genes to compute the impact of gene sets to our representation, thus establishing that critical link between the trained model and biological functions.

When applying the model to real data we see that the representation units in the model represent distinct modules of the data set.
These modules correspond to varying activity of biological pathways and the identity of each representation unit can be pinpointed by the pathways that are active for that unit!

We believe that application of our model to a large single cell dataset, like the Human Cell Atlas, will enable us to uncover, not only cell types but at the same time biological programs and shared function.
New cells that are passed through the autoencoder can be pre-processed, projected into a latent space and analysed based on their representation and all of that in a single operation.
