#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch.nn.functional as F
from numpy import random
from model import ECAPA_TDNN
from sklearn.metrics import classification_report,confusion_matrix
import sklearn.metrics as metrics

device = "cuda" if torch.cuda.is_available() else "cpu"


class mydataset(Dataset):
    def __init__(self, filelist, labellist):
        self.filelist = filelist
        self.labellist = labellist

    def __len__(self):
        return self.filelist.shape[0]

    def __getitem__(self, idx):
        mfcc = self.filelist[idx]
        label = int(self.labellist[idx])
        return mfcc, label

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= num_batches
    correct /= size
    return correct, train_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct,test_loss

def evaleatue(dataloader, model, loss_fn):
    model.eval()
    y_label=[]
    y_pred=[]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_pred.extend(pred.argmax(1).numpy())
            y_label.extend(y.numpy())
    return y_label,y_pred



path="processed_mfcc_4"
filelist=np.load("./"+path+"/train_samples.npz")['arr_0']
labellist=np.load("./"+path+"/train_labels.npz")['arr_0']

filelist_test=np.load("./"+path+"/test_samples.npz")['arr_0']
labellist_test=np.load("./"+path+"/test_labels.npz")['arr_0']


print(labellist.shape)
dataset=mydataset(filelist,labellist)
length=len(dataset)
train_size = int(0.8 * len(dataset))
validate_size = len(dataset) - train_size
train_set,validate_set=torch.utils.data.random_split(dataset,[train_size,validate_size])
batch_size=64
train_dataloader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
validate_dataloader = DataLoader(validate_set, batch_size=batch_size,shuffle=True)

vad=ECAPA_TDNN(20,4).to(device)
loss_fn = nn.CrossEntropyLoss()



def train_model():
    length=len(dataset)

    
    optimizer = torch.optim.Adam(vad.parameters(), lr=1e-3,)

    epochs =10
    train_losses = []
    train_acces = []
    eval_losses = []
    eval_acces = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_correct, train_loss=train(train_dataloader, vad, loss_fn, optimizer)
        test_correct, test_loss=test(validate_dataloader, vad, loss_fn)
        train_losses.append(train_loss)
        train_acces.append(train_correct)
        eval_losses.append(test_loss)
        eval_acces.append(test_correct)

    plt.figure(1)
    plt.plot(np.arange(len(train_losses)), train_losses,label="train loss")
    plt.plot(np.arange(len(eval_losses)), eval_losses, label="valid loss")
    plt.legend()
    plt.xlabel('epoches')
    plt.title('Model loss')
    plt.savefig("./images/"+path+"_loss.jpg")

    plt.figure(2)
    plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")
    plt.plot(np.arange(len(eval_acces)), eval_acces, label="valid acc")
    plt.legend()
    plt.xlabel('epoches')
    plt.title('Model accuracy')
    plt.savefig("./images/"+path+"_accuracy.jpg")
    plt.show()

    print("Done!")

    torch.save(vad,"./models/"+path+'.pth')


def plot_ROC(labels, preds, path):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path
    """
    fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)

    roc_auc1 = metrics.auc(fpr1, tpr1)
    plt.figure(3)
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROCs for Densenet')
    plt.legend(loc="lower right")
    plt.savefig("./images/"+path+"_roc.jpg") 

def test_model():
    net=torch.load("./models/"+path+'.pth')
    y_label, y_pred=evaleatue(validate_dataloader, net, loss_fn)
    #plot_ROC(y_label, y_pred,path)
    print(classification_report(y_label, y_pred))
    print(confusion_matrix(y_label, y_pred))


if __name__=="__main__":
    train_model()
    test_model()
