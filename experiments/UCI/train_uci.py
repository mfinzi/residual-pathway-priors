import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from uci_helpers import uci_loader, uci_reps
import argparse
import pandas as pd
import os
import sys
from rpp.rpp_conv import RPPConv, RPPConv_L2, RPPConv_L1, ConvNet, LinearNet


def compute_mse(model, loader):
    loss_func = torch.nn.MSELoss()
    mse = 0
    n_data = len(loader.dataset)
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        mse += x.shape[0] * loss_func(model(x).squeeze(), y)
    return mse.div(n_data).item()

def main(args):
    num_layers=4
    ch=32
    kwargs = uci_reps[args.dataset]
    if args.network.lower() == 'rpp':
        print("Using RPP")
        model = RPPConv(num_layers, ch, **kwargs).cuda()
    elif args.network.lower() == "conv":
        print("Using Conv")
        model = ConvNet(num_layers, ch, **kwargs).cuda()
    else:
        print("Using MLP")
        model = LinearNet(num_layers, ch, **kwargs).cuda()

    tr_x, tr_y, te_x, te_y = uci_loader(args.dataset, path=args.data_path)
    n_train = tr_x.shape[0]
    n_test = te_x.shape[0]
    
    trainset = TensorDataset(tr_x, tr_y)
    trainloader = DataLoader(trainset, args.batch_size, pin_memory=True, num_workers=4)
    
    testset = TensorDataset(te_x, te_y)
    testloader = DataLoader(testset, args.batch_size, pin_memory=True, num_workers=4)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_func = torch.nn.MSELoss()
    if args.reg_fun.lower() == 'l1':
        reg_fun = RPPConv_L1
    else:
        reg_fun = RPPConv_L2

    logger = []
    for epoch in range(args.epochs):
        for x, y in trainloader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            outputs = model(x)
            loss=loss_func(outputs.squeeze(), y)
            if args.network.lower() == 'rpp':
                loss += reg_fun(model, conv_wd=args.conv_wd, 
                      basic_wd=args.basic_wd)
            loss.backward()
            optimizer.step()        
            scheduler.step()

        if (epoch % args.save_every == 0) or (epoch == args.epochs-1):
            with torch.no_grad():
                tr_mse = compute_mse(model, trainloader)
                te_mse = compute_mse(model, testloader)
            
            logger.append([epoch, tr_mse, te_mse])
            
    ## Save Outputs ##
    fpath = "./saved-outputs/" + args.dataset + "/"
    fname =  args.network + str(args.trial) + "_conv_wd" + str(args.conv_wd) + "_basic_wd" + str(args.basic_wd)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    df = pd.DataFrame(logger)
    df.columns = ['epoch', 'tr_mse', 'te_mse']
    df.to_pickle(fpath + fname + ".pkl")
    
    
    torch.save(model.state_dict(), fpath + fname + ".pt")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="UCI Runner")
    parser.add_argument( 
        "--dataset",
        type=str,
        default="bike",
        help="see uci_helpers.py for list of datasets",
    )
    parser.add_argument( 
        "--network",
        type=str,
        default="rpp",
        help="rpp, conv, mlp",
    )

    parser.add_argument( 
        "--epochs",
        type=int,
        default=3000,
        help="training epochs",
    )
    parser.add_argument( 
        "--batch_size",
        type=int,
        default=500,
        help="training epochs",
    )

    parser.add_argument( 
        "--save_every",
        type=int,
        default=100,
        help="",
    )
    
    parser.add_argument(
         "--data_path",
        type=str,
        default="/datasets/uci/",
        help="path to data",
    )   
    
    parser.add_argument( 
        "--trial",
        type=int,
        default=0,
        help="just a flag for saving",
    )
    parser.add_argument( 
        "--basic_wd",
        type=float,
        default=1e-6,
        help="basic weight decay",
    )
    parser.add_argument(
        "--conv_wd",
        type=float,
        default=1e-6,
        help="equiv weight decay",
    )
    parser.add_argument( 
        "--reg_fun",
        type=str,
        default="L2",
        help="{L2, L}",
    )
    args = parser.parse_args()

    main(args)
