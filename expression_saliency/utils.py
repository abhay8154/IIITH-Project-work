import torch
import torch.nn as nn
import torch.nn.functional as F
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from copy import deepcopy
import matplotlib.gridspec as grd
from matplotlib import cm, colors
from sklearn.preprocessing import MinMaxScaler
from math import ceil


class GuidedSaliency(object):
    '''
        Class that implements the guided saliency maps.
    '''
    def __init__(self, model):
        self.model = deepcopy(model)
        self.model.eval()

    def guided_relu_hook(self, module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0), )

    def generate_saliency(self, input, target):
        '''
            Main method of the class: Computes the guided saliency maps for given target.
            Target corresponds to the unit of the final layer that is assigned a gradient of 1,
            while the rest of the units are assigned gradient 0.
        '''
        input.requires_grad = True

        self.model.zero_grad()

        # Adds a ReLU gate in all the units for the backward pass
        for module in self.model.modules():
            if type(module) == nn.Softplus:
                module.register_backward_hook(self.guided_relu_hook)

        # Do the forward pass
        output = self.model(input)

        # Assign zeros to the gradients of all the units, except the target unit
        grad_outputs = torch.zeros_like(output)
        grad_outputs[:, target] = 1

        # Do the backward pass
        output.backward(gradient=grad_outputs)
        input.requires_grad = False

        # Return the gradients of the input layer
        return input.grad.clone()


def train(args, model, train_loader, optimizer, epoch, loss_func=F.mse_loss, **kwargs):
    '''
        Trains the autoencoder for one epoch
    '''
    # Set the model to training mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Save batch to memory device
        data = data.to(device=args.device, dtype=args.dtype)

        # Don't forget to set the gradients to zero!
        optimizer.zero_grad()

        # Do a forward pass of the model, keep the latent representation and the final output
        representation, scores = model(data)

        # Compute the base loss
        loss = loss_func(scores, data, **kwargs)

        # If applicable add L1 regularization to the loss
        if args.l1_reg != 0.0:
            all_params = torch.cat([x[1].view(-1) for x in model.state_dict().items()])
            l1_regularization = args.l1_reg * torch.norm(all_params, 1)
            loss += l1_regularization

        # If applicable add orthogonality constraint to the loss
        if args.l_orthog != 0.0:
            loss += torch.norm(
                args.l_orthog * (
                        torch.eye(args.more_hidden_size).to(device=args.device, dtype=args.dtype) -
                        torch.mm(model.fc2.weight.data, model.fc2.weight.data.transpose(0,1))
                ),
                2
            )
        
        # Backpropagate the loss
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Every once in a while print an update message
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(train_loader)))


def train_classifier(args, model, train_loader, optimizer, epoch, loss_func=F.binary_cross_entropy, **kwargs):

    # Set the model to training mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Save batch to memory device
        data = data.to(device=args.device, dtype=args.dtype)
        target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
        
        # Don't forget to set the gradients to zero!
        optimizer.zero_grad()
        
        # Do a forward pass of the model
        scores = model(data)
        
        # Compute the base loss
        loss = loss_func(scores, target, **kwargs)
        
        # If applicable add L1 regularization to the loss
        if args.l1_reg != 0.0:
            all_params = torch.cat([x[1].view(-1) for x in model.state_dict().items()])
            l1_regularization = args.l1_reg * torch.norm(all_params, 1)
            loss += l1_regularization
        
        # Backpropagate the loss
        loss.backward()
        
        # Update the parameters
        optimizer.step()

        # Every once in a while print an update message
        if batch_idx == 0:
            print_result = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / target.shape[0])
    return print_result



def train_regression(args, model, train_loader, optimizer, epoch, loss_func=F.mse_loss, **kwargs):
    
    # Set the model to training mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Save batch to memory device
        data = data.to(device=args.device, dtype=args.dtype)
        target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
        
        # Don't forget to set the gradients to zero!
        optimizer.zero_grad()

        # Do a forward pass of the model
        scores = model(data)

        # Compute the base loss
        loss = loss_func(scores, target, **kwargs)
        
        # Backpropagate the loss
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Every once in a while print an update message
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / target.shape[0]))


def check_accuracy(args, model, test_loader, loss_func=F.mse_loss, **kwargs):
    
    # Model set to evaluation mode
    model.eval()
    
    test_loss = 0
    num = 0
    
    # No gradients are computed
    with torch.no_grad():
        for data, target in test_loader:

            # Store data in memory device
            data = data.to(device=args.device, dtype=args.dtype)

            # Do a forward pass of the model
            representation, scores = model(data)

            # Compute the loss
            test_loss += loss_func(scores, data, **kwargs)
            num += 1
        test_loss /= num
    return test_loss


def check_accuracy_classifier(args, model, test_loader):
    model.eval()
    test_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device=args.device, dtype=args.dtype)
            target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
            scores = model(data)
            test_loss += ((scores - target).abs_() < 0.25).sum().item()
            num += target.shape[0]
        test_loss /= num
    return test_loss


def check_accuracy_regression(args, model, test_loader, loss_func=F.mse_loss, **kwargs):
    model.eval()
    test_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device=args.device, dtype=args.dtype)
            target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
            scores = model(data)
            test_loss += loss_func(scores, target, **kwargs)
            num += target.shape[0]
        test_loss /= num
    return test_loss


def plot_embedding(method, args, dataset, model=None):
    '''
        Plots 2D embedding of the dataset with tSNE or UMAP.
        If an autoencoder is provided, the 2D embedding of the latent representation is shown.
    '''
    x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
    if model:
        model.eval()
        x, scores = model(x)
    if method == 'tsne':
        embedding_repres = TSNE(init='random').fit_transform(x.cpu().numpy())
    elif method == 'umap':
        embedding_repres = UMAP().fit_transform(x.cpu().numpy())
    fig = plt.scatter(embedding_repres[:, 0], embedding_repres[:, 1], c=dataset.subtypes, alpha=1, marker='.', s=5)
    return fig


def get_gene_set_dict(dataset, signatures='msigdb.v6.2.symbols.gmt.txt'):
    '''
        Maps a gene set to the columns of the dataset it corresponds to.
    '''
    from collections import defaultdict
    gene_set_dict = defaultdict(list)
    with open(signatures) as go_file:
        for line in go_file.readlines():
            if line.startswith('HALLMARK_'):
                gene_set = line.strip().split('\t')
                gene_set_dict[gene_set[0]] = list(np.where(dataset.expressions.columns.str.upper().isin(gene_set[1:])))

    return gene_set_dict


def get_common_model_part(pretrained_model, new_model):
    '''
        Loads the weights of the pretrained model to the new model.
    '''
    pretrained_dict = pretrained_model.state_dict()
    new_model_dict = new_model.state_dict()

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    
    # Overwrite entries in the existing state dict
    new_model_dict.update(pretrained_dict)
    
    # Load the new state dict
    new_model.load_state_dict(pretrained_dict)

    return new_model


def get_gene_set_scores(args, model, dataset, gene_set_dict, condition):
    '''
        Returns saliency scores for given gene set.

        Saliency scores are given by a matrix of size (R x GS),
        where GS is the number of gene sets in gene_set_dict and
        R is the length of the representation layer.
        
        A condition might be given to specify the samples (rows of the dataset)
        for which we want to compute the saliency maps.
    '''

    gene_set_score = np.zeros((args.more_hidden_size, len(gene_set_dict)))
    gene_set_std = np.zeros((args.more_hidden_size, len(gene_set_dict)))

    # Initialize the guided saliency model
    grads = GuidedSaliency(model)

    for i in range(args.more_hidden_size):
        
        # Subset the dataset for given condition
        if condition is None:
            x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
        else:
            x = torch.tensor(dataset.__getitem__([ix for (ix, x) in enumerate(condition) if x])[0],
                             device=args.device, dtype=args.dtype)
        
        # Compute the saliency maps for given data
        saliency_input = grads.generate_saliency(x, i).abs()

        # Get the median saliency value of the genes of each gene set 
        for j, gene_set in enumerate(gene_set_dict.values()):
            gene_set_score[i, j] = saliency_input[:, gene_set[0]].median(1)[0].mean()
            gene_set_std[i, j] = saliency_input[:, gene_set[0]].median(1)[0].std()

    # Turn array into dataframe and remove prepending HALLMARK_ from the gene set names
    gene_set_score = pd.DataFrame(gene_set_score, columns=[x[9:] for x in gene_set_dict.keys()])
    gene_set_std = pd.DataFrame(gene_set_std, columns=[x[9:] for x in gene_set_dict.keys()])

    return gene_set_score, gene_set_std


def score_gene_set(dataset, input_dim, args, model, signatures, condition=None):
    '''
        Evaluates the gene set signature contributions to the representation layer of the model using saliency maps
    '''
    ## Gene sets
    gene_set_dict = get_gene_set_dict(dataset, signatures)

    ## Get the encoding part of the model, that outputs the representation layer
    import models
    model_encoder = models.AutoencoderTwoLayers_encoder(input_dim, args.hidden_size, args.more_hidden_size).to(args.device)
    model_encoder = get_common_model_part(model, model_encoder)

    ## Compute to importance of each gene set to the representation layer
    gene_set_score, gene_set_std = get_gene_set_scores(args, model_encoder, dataset, gene_set_dict, condition)
    return gene_set_score


def get_embedding(method, args, dataset, model=None):
    '''
        Returns an embedding of the dataset or the transformed dataset if an autoencoder model is provided.
        Method for the embedding can be either tSNE or UMAP.
    '''
    x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
    if model:
        model.eval()
        x, scores = model(x)
    if method == 'tsne':
        embedding_repres = TSNE(init='random').fit_transform(x.cpu().numpy())
    elif method == 'umap':
        embedding_repres = UMAP().fit_transform(x.cpu().numpy())
    return embedding_repres


def get_singular_values_norm(args, dataset, model):
    '''
        Returns the normed singular value decomposition of the representation layer of the model.
    '''
    x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
    model.eval()

    # Do a forward pass
    x, scores = model(x)

    # Compute the singular value decomposition of the representation layer of the model
    s = np.linalg.svd(x.cpu().numpy(), compute_uv=False)

    # Return the Euclidean norm of the SVD
    return np.linalg.norm(s, 2)


def plot_clusters_full_color_multiple(clusters, embedding, title, fig):
    cluster_list = list(set(clusters))
    cluster_list.sort()
    num_clusters = len(cluster_list)
    gs = grd.GridSpecFromSubplotSpec(1,2, width_ratios=[20, 1], subplot_spec=fig)
    ax = plt.subplot(gs[0])
    for i in range(num_clusters):
        cluster_i = cluster_list[i]
        if cluster_i == 'anone':
            ax.scatter(embedding[clusters == cluster_i, 0], embedding[clusters == cluster_i, 1], label=cluster_i,
                       cmap='Dark2', marker='.', s=1, alpha=.2)
        else:
            ax.scatter(embedding[clusters == cluster_i, 0], embedding[clusters == cluster_i, 1], label=cluster_i,
                       cmap='Dark2', marker='.', s=1)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=5, fontsize=6)


def plot_gene_set_score(ax, gene_set_score, gene_set_score_comp=None, diverging=True, plot_nonlin_factor=5,
                        show_percent=0.85, cbar_draw=True, cbar_ax=None, hide_black=False):
    import seaborn as sns
    grays = cm.get_cmap('gray', 100)
    cmap = colors.ListedColormap(grays(np.power(np.linspace(0, 1, 101), plot_nonlin_factor)))
    sns.set(font_scale=.6)
    names = gene_set_score.columns
    scaler = MinMaxScaler()
    rowindex = gene_set_score.index
    gene_set_score = scaler.fit_transform(gene_set_score)
    gene_set_score = pd.DataFrame(gene_set_score, columns=names)
    gene_set_score.index = rowindex
    if hide_black:
        sns.heatmap(gene_set_score.T.loc[:, (gene_set_score.T>0.7).any()], cmap=cmap, yticklabels=True,
                    xticklabels=True, ax=ax, cbar=cbar_draw, cbar_ax=cbar_ax, vmin=0, vmax=1)
    else:
        sns.heatmap(gene_set_score.T, cmap=cmap, yticklabels=True, xticklabels=True, ax=ax, cbar=cbar_draw,
                    cbar_ax=cbar_ax, vmin=0, vmax=1)
    if gene_set_score_comp is not None:
        if not diverging:
            scaler = MinMaxScaler()
            rowindex = gene_set_score_comp.index
            gene_set_score_comp = scaler.fit_transform(abs(gene_set_score_comp))
            gene_set_score_comp = pd.DataFrame(gene_set_score_comp, columns=names)
            gene_set_score_comp.index = rowindex
            gene_set_score_comp.where(gene_set_score > show_percent, np.nan, inplace=True)
            reds = cm.get_cmap('Reds', 100)
            cmap = colors.ListedColormap(reds(np.linspace(0, .7, 101)))
            if hide_black:
                sns.heatmap(gene_set_score_comp.T.loc[:, (gene_set_score.T > 0.7).any()], cmap=cmap, yticklabels=True,
                            xticklabels=True, ax=ax, vmin=0, vmax=1, cbar=cbar_draw, cbar_ax=cbar_ax)
            else:
                sns.heatmap(gene_set_score_comp.T, cmap=cmap, yticklabels=True, xticklabels=True, ax=ax,
                            cbar=cbar_draw, cbar_ax=cbar_ax, vmin=0, vmax=1)
        else:
            scale_const = gene_set_score_comp.abs().max(axis=0)
            rowindex = gene_set_score_comp.index
            gene_set_score_comp = gene_set_score_comp / scale_const
            gene_set_score_comp = pd.DataFrame(gene_set_score_comp, columns=names)
            gene_set_score_comp.index = rowindex
            gene_set_score_comp.where(gene_set_score > show_percent, np.nan, inplace=True)
            reds = cm.get_cmap('bwr', 100)
            cmap = colors.ListedColormap(reds(np.linspace(0.1, 0.9, 101)))
            if hide_black:
                sns.heatmap(gene_set_score_comp.T.loc[:, (gene_set_score.T > 0.7).any()], cmap=cmap, yticklabels=True,
                            xticklabels=True, ax=ax, vmin=-1, vmax=1, cbar=cbar_draw, cbar_ax=cbar_ax)
            else:
                sns.heatmap(gene_set_score_comp.T, cmap=cmap, yticklabels=True, xticklabels=True, ax=ax,
                            cbar=cbar_draw, cbar_ax=cbar_ax, vmin=-1, vmax=1)
    return ax


def make_plot(args, dataset, title, dest, model=None, plot='umap', dim=(9, 5)):
    fig = plt.figure(figsize=dim)
    outer_grid = grd.GridSpec(1, 1)

    panel = outer_grid[0]
    embedding = get_embedding(plot, args, dataset, model)
    plot_clusters_full_color_multiple(dataset.subtypes, embedding, title, panel)
    fig.tight_layout()
    # plt.savefig(dest, bbox_inches='tight', format='eps')
    plt.show()
    plt.close('all')


def make_heatmap(args, dataset, model):
    '''
        Displays heatmap as in Figure 2 of publication.
    '''
    input_dim = dataset.expressions.shape[1]

    # Computes gene set scores for whole dataset
    gene_set_score = score_gene_set(dataset, input_dim, args, model, signatures=args.signatures)

    i = 0

    # Initialize figure
    fig = plt.figure(figsize=(7, 7))
    outer_grid = grd.GridSpec(int(ceil((len(dataset.subtypes.unique())+1)/2)), 2)
    ax = fig.add_subplot(outer_grid[i])

    # Make the heatmap
    plot_gene_set_score(ax, gene_set_score, cbar_draw=True, hide_black=True)
    ax.set_title('Whole dataset', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)
    
    # For each "subtype plot" the gene set scores of the subtype are computed against "background gene set scores" 
    for subtype in sorted(dataset.subtypes.unique()):
        i += 1
        gene_set_score_a = score_gene_set(dataset, input_dim, args, model, signatures=args.signatures,
                                          condition=(dataset.subtypes == subtype))
        gene_set_score_a = gene_set_score_a.loc[(gene_set_score != 0).any(axis=1), :]
        gene_set_score_b = score_gene_set(dataset, input_dim, args, model, signatures=args.signatures,
                                          condition=(dataset.subtypes != subtype))
        gene_set_score_b = gene_set_score_b.loc[(gene_set_score != 0).any(axis=1), :]

        ax = fig.add_subplot(outer_grid[i])
        plot_gene_set_score(ax, gene_set_score.loc[(gene_set_score != 0).any(axis=1), :],
                            gene_set_score_b - gene_set_score_a, diverging=True, cbar_draw=False, hide_black=True)
        ax.set_title(subtype, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)

    fig.tight_layout()
    plt.savefig('figures/Heatmap_{}.eps'.format(args.dataset), bbox_inches='tight', format='eps')
    # plt.show()
    plt.close('all')


def plot_test_loss(test_losses, title, dest=None):
    test_losses = torch.stack(test_losses)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(test_losses)+1), test_losses.cpu().numpy())

    ax.set(xlabel='epochs', ylabel='loss', title=title)
    ax.grid()

    if dest:
        fig.savefig(dest)
    plt.show()


def velten_markers(dataset):
    '''
        Manual procedure for the annotation of Velten et al. dataset.
    '''
    annot = pd.read_csv('data/Velten_PhenoData.csv', sep=' ')
    clusters = pd.DataFrame(1, dataset.expressions.index, ['HSC', 'MPP', 'CMP', 'MLP', 'MEP', 'GMP'])

    # For all the same
    index = annot.loc[clusters['MEP'] == 1, 'FACS_SSC.Area'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, :] = 0

    index1 = annot.loc[clusters['MEP'] == 1, 'FACS_FSC.Area'].nsmallest(int(clusters['MEP'].sum() / 4)).index
    index2 = annot.loc[clusters['MEP'] == 1, 'FACS_SSC.Area'].nlargest(int(clusters['MEP'].sum() / 4)).index
    clusters.loc[index1, :] = 0
    clusters.loc[index2, :] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_Lin'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, :] = 0

    # MEP
    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd38'].nsmallest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd10'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd135'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd45RA'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    # CMP
    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd38'].nsmallest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd10'].nlargest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd135'].nsmallest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    # GMP
    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd38'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd10'].nlargest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd135'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd45RA'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    # HSC
    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd34'].nsmallest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd38'].nlargest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd90'].nsmallest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd45RA'].nlargest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    # MPP
    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd38'].nlargest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd90'].nlargest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd45RA'].nlargest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    # MLP
    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd38'].nlargest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd45RA'].nsmallest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd10'].nsmallest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    return clusters


def find_marker_genes(args, dataset, model, topk):
    '''
        Finds topk inputs that affect the model output, based on guided saliency maps.
    '''
    grads = GuidedSaliency(model)
    x = torch.tensor(dataset.__getitem__([ix for (ix, x) in enumerate(dataset.subtypes == 1) if x])[0],
                     device=args.device, dtype=args.dtype)
    saliency_input_1 = grads.generate_saliency(x, 0).abs().sum(0)

    indices = saliency_input_1.topk(topk, largest=True)
    marker_genes_detected = pd.DataFrame({'marker_gene': list(dataset.expressions.columns[indices[1].cpu().numpy()]),
                                          'saliency_score': indices[0].cpu().numpy()})
    marker_genes_detected['correct'] = marker_genes_detected['marker_gene'].map(
        lambda x: x in list(dataset.marker_genes.loc[dataset.marker_genes['module_id'].isin([3, 4, 6]), 'gene_id']))
    marker_percent = [x in list(marker_genes_detected['marker_gene']) for x in list(
        dataset.marker_genes.loc[dataset.marker_genes['module_id'].isin([3, 4, 6]), 'gene_id']
    )]
    marker_percent = sum(marker_percent) / len(marker_percent)
    return marker_percent, marker_genes_detected
