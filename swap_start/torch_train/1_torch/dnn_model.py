#!/usr/bin/env python
# coding: utf-8

import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
# device = torch.device("cuda")


# reference: torch vision
# https://github.com/pytorch/vision/blob/7dfd8dc8d6e8a5cd2a51d7146c15a7c79729fd95/torchvision/models/resnet.py#L32

class ResNetBlock(nn.Module):
    def __init__(self, planes=256, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class DNNModel(nn.Module):
    def __init__(self, n_stages=4, planes=256, kernel_size=3, cuda=True):
        super(DNNModel, self).__init__()
        # setup device
        self.device = torch.device("cuda" if cuda else "cpu")
        # first block
        in_planes = 3 # 3 input channels, each has 15x15 shape
        # padding is 0 because the board has hard edge
        self.conv0 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(planes)
        self.relu0 = nn.ReLU(inplace=True)
        # build res blocks
        self.res_blocks = nn.Sequential(*[ResNetBlock(planes, kernel_size) for _ in range(n_stages)])
        # build final block
        # 1x1 conv
        self.final_conv = nn.Conv2d(planes, 1, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(1)
        self.final_relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        # 13x13 because input is 15x15
        self.final_dense = nn.Linear(169, 256)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_res = nn.Linear(256, 1)
        self.final_tanh = nn.Tanh()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # to device
        self.to(self.device, non_blocking=True)


    def forward(self, x):
        # first block
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        # x = self.maxpool0(x)
        # residue blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        # final evaluation head
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        # flatten then compute output
        x = self.flatten(x)
        x = self.final_dense(x)
        x = self.final_relu2(x)
        x = self.final_res(x)
        res = self.final_tanh(x)
        return res
    
    def setupOptim(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer
        
    def fit(self, x, y, epochs, batch_size=32, validation_split=0.2, shuffle=True, verbose=True):
        assert len(x) == len(y), f'Wrong shapes provided. x {x.shape}, y {y.shape}'
        size = len(x)
        v_size = int(size * validation_split)
        t_size = size - v_size
        print(f"Fitting model with {size} data, {t_size} for training and {v_size} for validation")
        print('-'*80)
        # convert to device
        x = torch.from_numpy(x)#.to(self.device, non_blocking=True)
        y = torch.from_numpy(y.reshape(-1, 1))#.to(self.device, non_blocking=True)
        # build dataset and data loaders
        train_ds = TensorDataset(x[:t_size], y[:t_size])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        valid_ds = TensorDataset(x[t_size:], y[t_size:])
        valid_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        for epoch in range(epochs):
            self.train()
            n_done = 0
            total_loss = 0.0
            valid_loss = 10000
            i_b = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # print status
                i_b += 1
                n_done += len(xb)
                if i_b % 10 == 0:
                    total_loss += loss
                    average_loss = total_loss / i_b
                    print(f"Epoch {epoch:5d}/{epochs} {n_done:9d}/{t_size}: loss {average_loss:.5f}", end="\r")
            if v_size > 0:
                self.eval()
                with torch.no_grad():
                    valid_loss = 0.0
                    for xb, yb in valid_dl:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        valid_loss += self.criterion(self(xb), yb)
                    valid_loss = valid_loss / len(valid_dl)
                print(f"Epoch {epoch:5d}/{epochs} {t_size:9d}/{t_size}: loss {average_loss:.5f} val_loss {valid_loss:.5f}")
            else:
                print()
            # early return if loss is small enough
            if valid_loss < 0.001 and epoch > 5:
                break
            

    def fit_manual(self, x, y, epochs, batch_size=32, validation_split=0.2):
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y.reshape(-1, 1)).to(self.device, non_blocking=True)
        size = len(x)
        if size == 0: return
        v_size = int(size * validation_split)
        t_size = size - v_size
        # training mode
        self.train(True)
        for i in range(epochs):
            # training
            for i_batch in range((t_size+batch_size-1) // batch_size):
                start_idx = i_batch*batch_size
                end_idx = min(start_idx+batch_size, t_size)
                y_pred = self(x[start_idx:end_idx])
                loss = self.criterion(y_pred, y[start_idx:end_idx])
                print(f"Epoch {i:5d}/{epochs} {end_idx:9d}/{t_size}: loss {loss:5f}", end="\r")
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # validate set
            for i_batch in range((v_size+batch_size-1) // batch_size):
                start_idx = t_size + i_batch*batch_size
                end_idx = min(start_idx+batch_size, size)
                y_pred = self(x[start_idx:end_idx])
                v_loss = self.criterion(y_pred, y[start_idx:end_idx])
            print(f"Epoch {i:5d}/{epochs} {t_size:9d}/{t_size}: loss {loss:5f} val_loss {v_loss:5f}")
    
    def predict(self, x):
        # x = torch.as_tensor(x, device=self.device)
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        # evaluation mode
        self.train(False)
        with torch.no_grad():
            y = self(x)
        return y.cpu().numpy()


def get_new_model():
    model = DNNModel(n_stages=2, planes=168, kernel_size=3)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    model.setupOptim(criterion, optimizer)
    return model

def save_model(model, path):
    torch.save(model, path)

def load_existing_model(path):
    return torch.load(path)

#
def test_model():
    size = 3000
    x_train = np.random.randint(0, 1, size=(size,3,15,15)).astype(np.float32)
    y_train = np.random.random(size).astype(np.float32) * 2 - 1
    model = get_new_model()
    print(model)
    model.fit(x_train, y_train, epochs=10, validation_split=0.2)
    print(model.predict(x_train[:10]))

if __name__ == '__main__':
    test_model()
