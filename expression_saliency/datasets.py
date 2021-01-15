from torch.utils.data import Dataset, sampler
import random
import pandas as pd
import numpy.random as random
import numpy as np
from sklearn.preprocessing import minmax_scale


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.num_samples))

    def __len__(self):
        return self.num_samples


class PaulDataset(Dataset):
    '''
        Class for Paul et al. dataset
    '''
    def __init__(self):
        self.expressions = pd.read_csv('data/Paul.csv')
        self.subtypes = self.expressions['cluster']
        self.expressions = self.expressions.drop(columns=['sample', 'cluster'])

        index = list(self.subtypes.index[self.subtypes == 'CMP CD41']) + \
                list(self.subtypes.index[self.subtypes == 'Cebpe control']) + \
                list(self.subtypes.index[self.subtypes == 'CMP Flt3+ Csf1r+']) + \
                list(self.subtypes.index[self.subtypes == 'Cebpa control']) + \
                list(self.subtypes.index[self.subtypes == 'CMP Irf8-GFP+ MHCII+'])

        # Shuffles the indexes of the dataset
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]

        # Filters out genes with very low expression levels
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]

        # Makes all gene names uppercase
        names = self.expressions.columns.str.upper()
        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        # Don't forget to shuffle the indexes for the subtypes as well!
        self.subtypes = self.subtypes.iloc[index]

        # Retain only the genes present in the msigdb pathways of interest
        gene_set_list = []
        with open('signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))

        self.expressions = self.expressions.iloc[:, index]

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class VeltenDataset(Dataset):
    '''
        Class for Velten et al. dataset
    '''
    def __init__(self):
        self.expressions = pd.read_csv('data/Velten.csv')
        self.subtypes = self.expressions['cluster']
        self.expressions = self.expressions.drop(columns=['sample', 'cluster'])

        index = list(self.expressions.index)

        # Shuffles the indexes of the dataset
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        self.subtypes = self.subtypes.iloc[index]

        # Filters out genes with very low expression levels
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]

        # Makes all gene names uppercase
        names = self.expressions.columns.str.upper()

        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        # Retain only the genes present in the msigdb pathways of interest
        gene_set_list = []
        with open('signatures/msigdb.v6.2.symbols.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.str.upper().isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))

        self.expressions = self.expressions.iloc[:, index]

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class PBMCDataset(Dataset):
    '''
        Class for 10X PBMC dataset
    '''
    def __init__(self):
        self.expressions = pd.read_csv('data/PBMC.csv')
        self.subtypes = self.expressions['clusters'].replace({1: "CD4 T cells", 2: "CD14+ Monocytes", 3: "B cells",
                                                              4: "CD8 T cells", 5: "FCGR3A+ Monocytes", 6: "NK cells",
                                                              7: "Dendritic cells", 8: "Megakaryocytes"})
        self.expressions = self.expressions.drop(columns=['clusters'])

        index = list(self.expressions.index)

        # Shuffles the indexes of the dataset
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        self.subtypes = self.subtypes.iloc[index]

        # Filters out genes with very low expression levels
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]

        # Makes all gene names uppercase
        names = self.expressions.columns.str.upper()

        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        # Retain only the genes present in the msigdb pathways of interest
        gene_set_list = []
        with open('signatures/msigdb.v6.2.symbols.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.str.upper().isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))

        self.expressions = self.expressions.iloc[:, index]

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class ToyXORDataset(Dataset):
    def __init__(self):

        annotation = pd.read_csv('data/revision_xor_annotation.csv')
        expressions = pd.read_csv('data/revision_xor_counts.csv')
        self.marker_genes = pd.read_csv('data/revision_xor_markergenes.csv')
        self.subtypes = annotation['x']

        colnames = expressions.columns
        expressions = minmax_scale(expressions)
        self.expressions = pd.DataFrame(expressions, columns=colnames)
        # self.expressions = self.expressions.iloc[:, 0:2]
        # sns.scatterplot(x=self.expressions.iloc[:, 0], y=self.expressions.iloc[:, 1], hue=self.subtypes)
        # plt.show()

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype
