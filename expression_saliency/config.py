import torch
import munch


def config():
    args = args = munch.Munch()
    args['batch_size'] = 1
    args['test_batch_size'] = 500
    args['epochs'] = 500
    args['l1_reg'] = 0
    args['l2_reg'] = 0
    args['hidden_size'] = 128
    args['more_hidden_size'] = 64
    args['train_percent'] = 0.95
    args['momentum'] = 0.9
    args['seed'] = 1
    args['log_interval'] = 500
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['dtype'] = torch.float32
    return args


def pbmc_config():
    args = config()
    args['lr'] = 1e-3
    args['l_orthog'] = 5e-2
    args['dataset'] = 'PBMC'
    args['signatures'] = 'signatures/DMAP_signatures_PBMC.txt'
    return args


def paul_config():
    args = config()
    args['lr'] = 1e-3
    args['l_orthog'] = 5e-2
    args['dataset'] = 'Paul'
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    return args


def velten_config():
    args = config()
    args['lr'] = 2e-11
    args['l_orthog'] = 1
    args['dataset'] = 'Velten'
    args['signatures'] = 'signatures/DMAP_signatures_Velten.txt'
    return args


def toy_MLP_XOR_config():
    args = config()
    args['epochs'] = 1000
    args['train_percent'] = 0.9
    args['hidden_size'] = 200
    args['more_hidden_size'] = 4
    args['lr'] = 1e-3
    args['l2_reg'] = 0
    args['l1_reg'] = 0
    args['l_orthog'] = 0
    args['dataset'] = 'Toy_MLP'
    args['signatures'] = ''
    args['batch_size'] = 1
    args['test_batch_size'] = 500
    args['dropout'] = 0.2
    return args
