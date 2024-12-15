import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNN(nn.Module):
    def __init__(self, dataset_name):
        super(DNN, self).__init__()
        self.output_size = 2
        input_size = 1
        if 'sea' in dataset_name:
            input_size = 3
        elif 'sine' in dataset_name:
            input_size = 2
        elif 'elec2' in dataset_name:
            input_size = 8
        elif 'mixed' in dataset_name:
            input_size = 4
        elif 'powersupply' in dataset_name:
            input_size = 3
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        if 'airline' in dataset_name:
            self.fc1 = nn.Sequential(
                nn.Linear(679, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

        self.fc2 = nn.Linear(64, self.output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def reset_last_layer(self):
        self.fc2 = nn.Linear(64, self.output_size).to(device)

import torch.optim as optim
import torch
from torch import nn, optim



def remove_batch_norm(module):
    ''' This function replaces each nn.BatchNorm2d with nn.Identity '''
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = nn.Identity()
    else:
        for name, child in module.named_children():
            module_output.add_module(name, remove_batch_norm(child))
    return module_output


def remove_batch_norm(module):
    ''' This function replaces each nn.BatchNorm2d with nn.Identity '''
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = nn.Identity()
    else:
        for name, child in module.named_children():
            module_output.add_module(name, remove_batch_norm(child))
    return module_output


class MyClassifier():
    def __init__(self, classifier_type="GNB", optimizer_type="Adam", dataset_name=None, first_x=None, first_y=None, totaly=None, model_num=5):
        self.classifier_type = classifier_type
        self.optimizer_type = optimizer_type
        self.dataset_name = dataset_name
        self.retrained_marker = -1  # Initialized here
        self.original_lr = 0.01  # Initialized here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.unique_classes = None
        self.model_num = model_num
        self.initialize_classifier(first_x, first_y, totaly)
    
    def initialize_classifier(self, first_x, first_y, totaly):
        self.optimizer = None
        if "DNN" in self.classifier_type:
            self.classifier = [DNN(self.dataset_name).to(device) for _ in range(self.model_num)]
            if self.optimizer_type == "Adam":
                self.optimizer = [optim.Adam(self.classifier[i].parameters(), lr=self.original_lr) for i in range(self.model_num)]
            self.train(first_x, first_y, totaly)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def reset_lr(self):
        for i in range(self.model_num):
            self.optimizer[i] = optim.Adam(self.classifier[i].parameters(), lr=self.original_lr)
    def reset_model(self, classifier_type, x=None, y=None, lr_boost_factor=10, optimizer_type=None):
        if classifier_type == "DNN":
            if self.dataset_name != 'airline':
                self.initialize_classifier(x, y, None)
            else:
                for i in range(self.model_num):
                    self.optimizer[i] = optim.Adam(self.classifier[i].parameters(), lr=self.original_lr*10)
        return self.classifier, self.optimizer
    
    def train(self, X, y, totaly, epochs=50):
        if self.unique_classes is None and self.dataset_name != 'cifar10':
            self.unique_classes = list(set(totaly.ravel()))
            self.unique_classes = list(range(24))
        if "DNN" in self.classifier_type:
            for i in range(self.model_num):
                self._train_DNN_epoch(self.classifier[i], self.optimizer[i], X, y, epochs=50)
        
    def predict(self, x, y_true=None):
        if self.classifier is None:
            raise ValueError("Classifier not initialized!")
        y_pred = None
        y_prob = None
        residuals = []
        if not isinstance(y_true, torch.LongTensor):
            y_true = y_true.astype(int)
        else:
            y_true = y_true.cpu().numpy()
        x = torch.from_numpy(x).float().to(device)

        if "DNN" in self.classifier_type:
            probs = []
            for i in range(self.model_num):
                self.classifier[i].eval()
                with torch.no_grad():
                    outputs = self.classifier[i](x)
                    y_prob = torch.nn.functional.softmax(outputs, dim=1)
                    _, y_pred = outputs.max(1)
                    y_pred = y_pred.cpu().numpy()
                    y_prob = y_prob.cpu().numpy()
                    probs.append(y_prob)
                    residual = 1 - y_prob[np.arange(len(y_true)), y_true]
                    residuals.append(residual)
            p1 = np.stack(probs)
            # and then compute average probability
            vote_prob = np.mean(p1, axis=0)
            y_pred = np.argmax(vote_prob, axis=1)
            residuals = np.concatenate(residuals)
        else:
            raise NotImplementedError(f"Predict method not implemented for classifier type: {self.classifier_type}")

        acc = None
        # residuals = None
        if y_true is not None:
            correct = (y_pred == y_true).sum()
            acc = correct / len(y_true)
            residuals = 1 - vote_prob[np.arange(len(y_true)), y_true]

        wrong_prediction = None
        if y_true is not None:
            wrong_prediction = (y_pred != y_true).astype(int)

        return y_pred, acc, residuals, np.equal(y_pred, y_true), wrong_prediction

    def train_resnet18(self, X, y):
        self.classifier.model.train()
        epochs = 10
        for _ in range(epochs):
            # x = torch.stack([self.classifier.transform(xi) for xi in X]).to(self.device)
            x = X.cuda()
            y = y.cuda()
            self.optimizer.zero_grad()
            outputs = self.classifier(x)
            loss = self.classifier.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

    def finetune(self, X, y, totaly):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.original_lr/10
        for name, param in self.classifier.model.named_parameters():
            if "fc" not in name:  
                param.requires_grad = False
        self.classifier.model.train()
        for _ in range(20):
            # x = torch.stack([self.classifier.transform(xi) for xi in X]).to(self.device)
            x = X.cuda()
            y = y.cuda()
            self.optimizer.zero_grad()
            outputs = self.classifier(x)
            loss = self.classifier.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.original_lr

        for name, param in self.classifier.model.named_parameters():
            param.requires_grad = True

    def _train_DNN_epoch(self, model, optimizer, X, y, epochs=50):
        model.train()
        inputs = torch.FloatTensor(X).to(device)
        labels = torch.LongTensor(y).to(device)
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def revert_lr(self, optimizer_type=None):
        """
        Reverts the learning rate to its original value.
        """
        if optimizer_type == "SGD":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.original_lr
        elif optimizer_type == "Adam":
            # Assuming you want the same functionality for Adam, but adjust as needed
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.original_lr

