import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
import models
import utils
import config
from importlib import reload


def main(ind):

    dataset = datasets.ToyXORDataset()
    args = config.toy_MLP_XOR_config()

    # torch.manual_seed(args.seed)

    input_dim = dataset.expressions.shape[1]
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=datasets.ChunkSampler(int(args.train_percent * len(dataset))), **kwargs)
    test_loader = DataLoader(
        dataset, batch_size=args.test_batch_size,
        sampler=datasets.ChunkSampler(len(dataset), int(args.train_percent * len(dataset))), **kwargs)

    # model = models.TwoLayerMLP(input_dim, args.hidden_size).to(args.device)
    model = models.TwoLayerMLP_dropout(input_dim, args.hidden_size, args.dropout).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    test_losses = [0, 0, 0]
    for epoch in range(1, args.epochs + 1):
        print_result = utils.train_classifier(args, model, train_loader, optimizer, epoch, F.binary_cross_entropy)
        test_loss = utils.check_accuracy_classifier(args, model, test_loader)
        test_losses.append(test_loss)
        marker_percent, marker_genes_detected = utils.find_marker_genes(args, dataset, model, 20)
        print(print_result + '\t test accuracy: {}\t marker gene detection accuracy: {}'.format(
            test_loss, marker_percent))
        if epoch % 10 == 0:
            print(marker_genes_detected)

    return marker_genes_detected


if __name__ == '__main__':
    marker_genes_detected = main()
