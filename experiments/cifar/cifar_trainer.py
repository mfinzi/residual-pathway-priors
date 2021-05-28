import torch
import numpy as np
import argparse
import pandas as pd
import os
import sys
from rpp.rpp_conv import RPP_D_Conv, RPPConv_L2, RPPConv_L1, D_FC, D_Conv
import utils
import torchvision
import torchvision.transforms as transforms
import tabulate
from torch.utils.data import DataLoader
import time

def main(args):
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = torchvision.datasets.CIFAR10(args.data_path, 
                                           train=True, download=False,
                                           transform=transform_train)
    trainloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    
    testset = torchvision.datasets.CIFAR10(args.data_path, 
                                           train=False, download=False,
                                           transform=transform_test)
    testloader = DataLoader(testset, shuffle=True, batch_size=args.batch_size)
        
    if args.reg_fun.lower() == "l1":
        regularizer = RPPConv_L1
    else:
        regularizer = RPPConv_L2
    if args.network.lower() == 'rpp':
        print("Using RPP", flush=True)
        model = RPP_D_Conv(in_ch=3, alpha=args.alpha)
    elif args.network.lower() == "conv":
        print("Using Conv", flush=True)
        model = D_Conv(in_ch=3, alpha=args.alpha)
    else:
        print("Using MLP", flush=True)
        model = D_FC(in_ch=3, alpha=args.alpha)

    model = model.cuda()
    
    ## training setup ##
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=0.9,
        weight_decay=0.
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
    for epoch in range(args.epochs):
        time_ep = time.time()
        train_res = utils.train_epoch(trainloader, model, criterion, optimizer, regularizer,
                                     args.basic_wd, args.conv_wd)
        
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            test_res = utils.eval(testloader, model, criterion)
        else:
            test_res = {'loss': None, 'accuracy': None}
        
        time_ep = time.time() - time_ep
        
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], 
                  test_res['loss'], test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table, flush=True)
        
    torch.save(model.state_dict(), "./saved-outputs/" + args.network + str(args.trial) + "_basic" +\
               str(args.basic_wd) + "_conv" + str(args.conv_wd)+".pt")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CIFAR Runner")
    parser.add_argument( 
        "--network",
        type=str,
        default="rpp",
        help="rpp, conv, mlp",
    )
    parser.add_argument( 
        "--trial",
        type=int,
        default=0,
        help="just a flag to distinguish models",
    )

    parser.add_argument( 
        "--epochs",
        type=int,
        default=400,
        help="training epochs",
    )
    parser.add_argument( 
        "--batch_size",
        type=int,
        default=128,
        help="training epochs",
    )

    parser.add_argument( 
        "--eval_freq",
        type=int,
        default=10,
        help="",
    )
    parser.add_argument( 
        "--data_path",
        type=str,
        default="/datasets/",
        help="",
    )
    parser.add_argument( 
        "--alpha",
        type=int,
        default=2,
        help="width parameter in D-Conv",
    )
    parser.add_argument( 
        "--basic_wd",
        type=float,
        default=1e-2,
        help="basic weight decay",
    )
    parser.add_argument(
        "--conv_wd",
        type=float,
        default=1e-4,
        help="equiv weight decay",
    )
    parser.add_argument( 
        "--reg_fun",
        type=str,
        default="L2",
        help="{L2, L1}",
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    
    args = parser.parse_args()
    main(args)