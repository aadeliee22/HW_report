import numpy as np
import csv
import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib

import datasets

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

dataset_odn = datasets.C100Dataset('./dataset/cifar100_nl/data/cifar100_nl.csv')
[tr_x, tr_y, ts_x] = dataset_odn.getDataset()

label = set(tr_y)
label = np.array(list(label))

def tr_val(x_, y_):
    mask = np.ones(len(x_))
    mask[:int(len(mask)/5)] = 0 # #1:#0 = 4:1
    np.random.shuffle(mask)
    x_tr, x_vl = x_[mask==1], x_[mask==0]
    y_tr, y_vl = y_[mask==1], y_[mask==0]
    return x_tr, x_vl, y_tr, y_vl
def batch_div(x_, y_, num):
    x_b, y_b = [], []
    mask = np.tile(np.arange(int(len(x_)/num)), num)
    np.random.shuffle(mask)
    for i in range (int(len(x_)/num)):
        x_b.append(x_[mask==i])
        y_b.append(y_[mask==i])
    return int(len(x_)/num), x_b, y_b

alp = 0.2
K = len(label)
trainx = np.array([img.imread(im_dir+u).T for i,u in enumerate(tr_x)])
trainy = np.ones([len(tr_y), 100], dtype='int32')*(alp/K)
for i,u in enumerate(label):
    for j,v in enumerate(tr_y):
        if v==u: trainy[j][i] += 1-alp
            
testx = np.array([img.imread(im_dir+u).T for i,u in enumerate(ts_x)])
trainx, testx = torch.FloatTensor(trainx), torch.FloatTensor(testx)
trainy = torch.FloatTensor(trainy)

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.C1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.C2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.C3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.C4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.W1 = nn.Linear(128*4*4, 1024)
        self.W2 = nn.Linear(1024, 512)
        self.W3 = nn.Linear(512, 100)
        self.S = nn.Sigmoid()
        self.D = nn.Dropout(p=0.3)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm1d(1024)
        self.bn5 = torch.nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.bn1(self.C1(x))
        x = self.pool(F.relu(x))
        x = F.relu(self.bn2(self.C2(x)))
        x = self.bn2(self.C3(x))
        x = self.pool(F.relu(x))
        x = self.bn3(self.C4(x))
        x = self.pool(F.relu(x))
        x = x.view(-1, 128*4*4)
        x = F.relu(self.bn4(self.W1(x)))
        x = self.D(x)
        x = F.relu(self.bn5(self.W2(x)))
        x = self.D(x)
        x = self.S(self.W3(x))
        return x
    
def loss1(output, y):
    #regular = reg*(torch.norm(self.W1.weight.data, p=1)+torch.norm(self.W2.weight.data, p=1))
    return F.binary_cross_entropy(output, y)+regular
def loss2(output, y):
    #regular = reg*(torch.norm(self.W1.weight.data)**2+torch.norm(self.W2.weight.data)**2)
    return F.binary_cross_entropy(output, y)

def accuracy(output, y):
    return torch.sum(torch.argmax(output, axis=1)==torch.argmax(y, axis=1))

reg = 0
model = VGG()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
n_epochs = 10
x_tr, x_vl, y_tr, y_vl = tr_val(trainx, trainy)
tr_loss, vl_loss = np.zeros(n_epochs, dtype='float64'), np.zeros(n_epochs, dtype='float64')
tr_accy, vl_accy, ts_accy = np.zeros_like(tr_loss), np.zeros_like(tr_loss), np.zeros_like(tr_loss)

for epoch in range(n_epochs):
    losst = 0.
    accyt = 0
    batchnum, x_tr_b, y_tr_b = batch_div(x_tr, y_tr, 100)
    for i in range (batchnum):
        output = model.forward(x_tr_b[i])
        loss = loss2(output, y_tr_b[i])
        accy = accuracy(output, y_tr_b[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losst += loss.item()
        accyt += accy
    
    output2 = model.forward(x_vl)
    tr_loss[epoch] = losst
    tr_accy[epoch] = accyt
    vl_loss[epoch] = loss2(output2, y_vl).item()
    vl_accy[epoch] = accuracy(output2, y_vl)
    
    print("epoch:", epoch, "*test loss:", losst, "*val loss:", vl_loss[epoch], \
          "*tr/vl accuracy:", tr_accy[epoch]/400, vl_accy[epoch]/100)
    
import matplotlib as mpl
import matplotlib.ticker as ticker
mpl.rcParams.update({
    'font.family' : 'STIXGeneral',
    'mathtext.fontset' : 'stix',
    'xtick.direction' : 'in' ,
    'xtick.labelsize' : 13 ,
    'xtick.top' : False ,
    'ytick.direction' : 'in' ,
    'ytick.labelsize' : 13 ,
    'ytick.right' : False ,
    'axes.labelsize' : 16,
    'legend.frameon' : False,
    'legend.fontsize' : 13,
    'legend.handlelength' : 1.5,
    'savefig.dpi' : 600, 
    'savefig.bbox' : 'tight'
})

fig, ax = plt.subplots(1,2, figsize=(12,4))

ax[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))

ax[0].plot(tr_loss/tr_loss[0], '.-', label='training loss')
ax[0].plot(vl_loss/vl_loss[0], '.-', label='heldout loss')
ax[0].legend()
ax[0].set_ylim(-0.1, 1.1)
ax[0].set_xlim(0, 9)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')

ax[1].plot(tr_accy/400, '.-', label='training accuracy')
ax[1].plot(vl_accy/100, '.-', label='heldout accuracy')
ax[1].set_ylim(0, 100)
ax[1].set_xlim(0, 9)
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend()


plt.savefig('vgg5.png')


def tra(x_):
    mask = np.ones(len(x_))
    #mask[:int(len(mask)/5)] = 0 # #1:#0 = 4:1
    #np.random.shuffle(mask)
    x_tr, x_vl = x_[mask==1], x_[mask==0]
    return x_tr, x_vl
a = []
b = []
x_ts, _ = tra(testx)
output = model.forward(x_ts)
for i in range(9999):
    a.append(ts_x[i])
b = label[torch.argmax(output, axis=1)]
a = np.array(a)
np.savetxt('cifar100nlpred_vgg5.csv', np.array([a, b]).T, fmt='%s', delimiter=',',\
          header='id,category')