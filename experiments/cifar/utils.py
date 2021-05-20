import math
import torch
from torch import nn
import numpy as np

def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        with torch.no_grad():
            output = model(input_var)
            # print(output)
            # output = output
            loss = criterion(output, target_var)

        loss_sum += loss.data.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def train_epoch(loader, model, criterion, optimizer, regularizer=None, 
                basic_wd=0.0, conv_wd=0.0):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)
        ce_loss = loss.item()
        if regularizer is not None:
            reg = regularizer(model, conv_wd, basic_wd)
            loss += reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += ce_loss * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }