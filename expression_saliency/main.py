import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
import models
import utils
import argparse
import config


def main():

    parser = argparse.ArgumentParser(description='PyTorch saliency maps for single cell RNA-seq expression matrix.')
    parser.add_argument('dataset', type=str, help='Select dataset among PBMC, Paul, Velten.')
    args = parser.parse_args()

    if args.dataset == 'PBMC':
        dataset = datasets.PBMCDataset()
        args = config.pbmc_config()
    elif args.dataset == 'Paul':
        dataset = datasets.PaulDataset()
        args = config.paul_config()
    elif args.dataset == 'Velten':
        dataset = datasets.VeltenDataset()
        args = config.velten_config()
    else:
        print('Select dataset among PBMC, Paul, Velten.')
        quit()

    # torch.manual_seed(args.seed)

    # Load datasets in memory and split them in train and test set
    input_dim = dataset.expressions.shape[1]
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=datasets.ChunkSampler(int(args.train_percent * len(dataset))), **kwargs)
    test_loader = DataLoader(
        dataset, batch_size=args.test_batch_size,
        sampler=datasets.ChunkSampler(len(dataset), int(args.train_percent * len(dataset))), **kwargs)

    # Initialize autoencoder
    model = models.AutoencoderTwoLayers(input_dim, args.hidden_size, args.more_hidden_size).to(args.device)
    
    # Initialize the optimization routine
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.l2_reg)

    # Make a UMAP of the input
    utils.make_plot(args, dataset, 'UMAP of original dataset {}'.format(args.dataset),
                    'figures/UMAP_original_{}.eps'.format(args.dataset))

    # Train the autoencoder
    test_losses = [torch.tensor(10000, device=args.device)]
    for epoch in range(1, args.epochs + 1):
        kwargs = {'log_input': False, 'full': False}
        utils.train(args, model, train_loader, optimizer, epoch, F.poisson_nll_loss, **kwargs)

        # Every 10 epochs check the accuracy on the test set 
        if epoch % 10 == 0:
            kwargs = {'log_input': False, 'full': True}
            test_loss = utils.check_accuracy(args, model, test_loader, F.poisson_nll_loss, **kwargs)
            print('epoch: {}, test loss: {}'.format(epoch, test_loss))
            print('singular values l2-norm: {}'.format(utils.get_singular_values_norm(args, dataset, model)))
            if test_losses[-1] - test_loss < 1e-8:
                break
            test_losses.append(test_loss)

    # Make a UMAP of the representation layer when training is finished
    utils.make_plot(args, dataset, 'UMAP of representation layer - {}'.format(args.dataset),
                    'figures/UMAP_representation_{}.eps'.format(args.dataset), model)

    # Make heatmaps of pathway impact.
    utils.make_heatmap(args, dataset, model)

if __name__ == '__main__':
    main()
