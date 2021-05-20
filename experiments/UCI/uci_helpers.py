import numpy as np
import torch
import scipy.io as sio

def uci_loader(dataset='yacht', path='/datasets/uci/',
               test_proportion=0.2):
    rep = uci_reps[dataset]
    dat = sio.loadmat(path + dataset + "/" +dataset)['data']
    x, y = dat[:, :-1], dat[:, -1]
    
    curr_x_shape = x.shape[-1]
    goal_x_shape = rep['h'] * rep['w']
    if curr_x_shape < goal_x_shape:
        new_x = np.zeros((x.shape[0], goal_x_shape))
        new_x[:, :curr_x_shape] = x
        
        x = new_x
        
    train_cutoff = int(x.shape[0] * (1-test_proportion))
    train_x, test_x = x[:train_cutoff, :], x[train_cutoff:, :]
    train_y, test_y = y[:train_cutoff], y[train_cutoff:]
    
    train_x = torch.FloatTensor(train_x).view(train_x.shape[0],1,rep['h'],rep['w'])
    test_x = torch.FloatTensor(test_x).view(test_x.shape[0],1,rep['h'],rep['w'])
    return train_x, torch.FloatTensor(train_y), test_x, torch.FloatTensor(test_y)

uci_reps = {
    'yacht':{'h':3, 'w':2},
    'airfoil':{'h':3, 'w':2},
    'bike':{'h':5, 'w':4},
    'breastcancer':{'h':6, 'w':6},
    'buzz':{'h':9, 'w':9},
    'concrete':{'h':3, 'w':3},
    'elevators':{'h':5, 'w':4},
    'energy':{'h':3, 'w':3},
    'fertility':{'h':3, 'w':3},
    'forest':{'h':4, 'w':3},
    'airfoil':{'h':6, 'w':6},
    'gas':{'h':12, 'w':11},
    'housing':{'h':4, 'w':4},
    'keggdirected':{'h':5,'w':4},
    'keggundirected':{'h':7,'w':4},
    'pendulum':{'h':3, 'w':3},
    'protein':{'h':3,'w':3},
    'skillcraft':{'h':5, 'w':4},
    'wine':{'h':4, 'w':3}
}