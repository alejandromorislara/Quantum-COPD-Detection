import torch
import random
import numpy as np
import math


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset

import os
os.environ['PYTHONHASHSEED']='0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        
        if (x.shape[0] != y.shape[0]):
            raise Exception("incompatible arrays")
        
        y = y.reshape(-1,1)
        
        self.x = torch.from_numpy(x).to(torch.float)
        self.y = torch.from_numpy(y).to(torch.float)
        
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.y.shape[0]

    def __iter__(self):
        return zip(self.x, self.y)


def run_epoch(opt, tr_data, loss, model):
    opt.zero_grad()
    loss_value = loss(model(tr_data.x), tr_data.y)
    loss_value.backward()
    opt.step()

    return loss(model(tr_data.x), tr_data.y).detach()

class History:
    def __init__(self, tr_losses, val_losses, tr_accuracy, val_accuracy):
        self.tr_losses = tr_losses
        self.val_losses = val_losses
        self.tr_accuracy = tr_accuracy
        self.val_accuracy = val_accuracy

    def plot_losses(self):
        plt.plot(self.tr_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_accuracies(self):
        plt.plot(self.tr_accuracy, label='Training Accuracy')
        if self.val_accuracy:
            plt.plot(self.val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

def discretise(values):
    return values.detach().numpy() >= 0.5

def train(model, loss, opt, tr_data, max_epochs,
          val_data = None, patience = None, restore_best = True):
    tr_losses = []
    val_losses = []
    tr_accuracy = []
    val_accuracy = []
    states = []
    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        tr_losses.append(float(run_epoch(opt, tr_data, loss, model)))
        states.append(model.state_dict())
        print("Epoch", epoch+1, end = " ")
        print("| Train loss:", round(tr_losses[-1],4), end = " ")
        tr_accuracy.append(accuracy_score(discretise(model(tr_data.x)), tr_data.y))

        if val_data is None:
            print()
            continue

        val_losses.append(float(loss(model(val_data.x), val_data.y).detach())) 
        val_accuracy.append(accuracy_score(discretise(model(val_data.x)), val_data.y))
        print("| Valid loss:", round(val_losses[-1],4))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        if patience is not None:
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if restore_best:
                    print("Restoring best model from epoch", best_epoch+1)
                    model.load_state_dict(states[best_epoch])
                break

    return History(tr_losses, val_losses, tr_accuracy, val_accuracy) 

