import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pandas as pd
from PIL import Image
from data_loader import F-RCNNDataset, Augmentation

class Trainer:
    def __init__(self,debug=False,training_csv_path='data/train.csv',validation_csv_path='data/val.csv',
                 model_path='model.pth',load_model=False,batch_size=4,epochs=10, learning_rate=0.001):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.debug = debug
        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # create adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.learning_rate)

        # create learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

        # create loss function for classification and regression
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()

        # create dataset
        self.dataset_train = F-RCNNDataset(dataset_path= training_csv_path, transform=Augmentation())
        self.dataset_val = F-RCNNDataset(dataset_path= validation_csv_path, transform=Augmentation())

        # create data loader
        self.data_loader_train = DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.data_loader_val = DataLoader(dataset=self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # create model object detector
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)

        # load model if specified
        if load_model:
            self.model.load_state_dict(torch.load(model_path))


    def train(self):
        # train the model
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.lr_scheduler.step()
            self.evaluate_one_epoch()

            # save the model
            torch.save(self.model.state_dict(), 'model.pth')
    def train_one_epoch(self):
        self.model.train()
        for batch_idx, (images, targets) in enumerate(self.data_loader_train):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.optimizer.step()

            # print statistics
            if self.debug:
                print(f'Batch {batch_idx + 1}/{len(self.data_loader_train)} Loss: {losses.item():.4f}')
                print(loss_dict)
    def evaluate_one_epoch(self):
        self.model.eval()
        for batch_idx, (images, targets) in enumerate(self.data_loader_val):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # forward
            with torch.no_grad():
                loss_dict = self.model(images, targets)

            # print statistics
            if self.debug:
                print(f'Batch {batch_idx + 1}/{len(self.data_loader_val)} Loss: {losses.item():.4f}')

if __name__ == '__main__':
    trainer = Trainer(debug=True)
    trainer.train()