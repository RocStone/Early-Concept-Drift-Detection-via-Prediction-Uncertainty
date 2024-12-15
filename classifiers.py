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
        elif 'hyperplane' in dataset_name:
            input_size = 10
        elif 'elec2' in dataset_name:
            input_size = 8
        elif 'insect' in dataset_name:
            input_size = 33
            self.output_size = 6
        elif 'firedman' in dataset_name:
            input_size = 10
        elif 'mixed' in dataset_name:
            input_size = 4
        elif 'powersupply' in dataset_name:
            input_size = 3
        elif 'pokerhand' in dataset_name:
            input_size = 10
            self.output_size = 10
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        if 'airline' in dataset_name:
            self.fc1 = nn.Sequential(
                nn.Linear(679, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU()
            )

        self.fc2 = nn.Linear(64, self.output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def reset_last_layer(self):
        self.fc2 = nn.Linear(64, self.output_size).to(device)

from sklearn.naive_bayes import GaussianNB
import torch.optim as optim
from skmultiflow.trees import HoeffdingTreeClassifier
import lightgbm as lgb

import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.models import resnet18



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


class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model = remove_batch_norm(self.model)
        self.num_classes = num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def reset_last_layer(self):
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes).cuda()

    def reset(self):
        self.model = resnet18(pretrained=True)
        self.model = remove_batch_norm(self.model)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.cuda()

class MyClassifier():
    def __init__(self, classifier_type="GNB", optimizer_type="Adam", dataset_name=None, first_x=None, first_y=None, totaly=None):
        self.classifier_type = classifier_type
        self.optimizer_type = optimizer_type
        self.dataset_name = dataset_name
        self.retrained_marker = -1  # Initialized here
        self.original_lr = 0.01  # Initialized here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.unique_classes = None
        self.initialize_classifier(first_x, first_y, totaly)
    
    def initialize_classifier(self, first_x, first_y, totaly):
        self.optimizer = None
        if self.classifier_type == "GNB":
            self.classifier = GaussianNB()
            self.train(first_x, first_y, totaly)
        elif "DNN" in self.classifier_type:
            self.classifier = DNN(self.dataset_name).to(device)
            if self.optimizer_type == "Adam":
                self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.01)
            self.train(first_x, first_y, totaly)
        elif "hoeffding" in self.classifier_type:
            if self.dataset_name == 'airline':
                self.classifier = HoeffdingTreeClassifier(leaf_prediction='mc')
            else:
                self.classifier = HoeffdingTreeClassifier()
            self.train(first_x, first_y, totaly)
        elif self.classifier_type == 'resnet18':
            self.classifier = CustomResNet18(num_classes=10).cuda()
            self.optimizer = optim.SGD(self.classifier.model.parameters(), lr=0.01, momentum=0.9)
            self.train(first_x, first_y, totaly)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
    def reset_model(self, classifier_type, x=None, y=None, lr_boost_factor=10, optimizer_type=None):
        if classifier_type == "GNB":
            self.classifier = GaussianNB()
        elif "hoeffding" in self.classifier_type:
            if self.dataset_name == 'airline':
                self.classifier = HoeffdingTreeClassifier(leaf_prediction='mc')
            else:
                self.classifier = HoeffdingTreeClassifier()
        elif classifier_type == "DNN":
            self.classifier.reset_last_layer()
            if optimizer_type == "Adam":
                self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.original_lr)

        return self.classifier, self.optimizer
    
    def train(self, X, y, totaly):
        if self.unique_classes is None and self.dataset_name != 'cifar10':
            self.unique_classes = list(set(totaly.ravel()))
        if self.dataset_name == 'insect':
            self.unique_classes = list(range(24))
        if self.classifier_type == "GNB":
            self.classifier.partial_fit(X, y, classes=self.unique_classes)
        elif "DNN" in self.classifier_type:
            self._train_DNN_epoch(X, y)
        elif self.classifier_type == "LightGBM":
            train_data_chunk = lgb.Dataset(X, label=y, free_raw_data=False)
            self.classifier = lgb.train({'objective': 'binary', 'verbose': -1}, train_data_chunk, init_model=self.classifier)
        elif "hoeffding" in self.classifier_type:
            self.classifier.partial_fit(X, y)
        elif 'resnet18' in self.classifier_type:
            self.train_resnet18(X, y)
        
    def predict(self, x, y_true=None):
        if self.classifier is None:
            raise ValueError("Classifier not initialized!")

        y_pred = None
        y_prob = None
        if not isinstance(y_true, torch.LongTensor):
            y_true = y_true.astype(int)
        else:
            y_true = y_true.cpu().numpy()

        if self.classifier_type == "GNB":
            y_pred = self.classifier.predict(x)
            y_prob = self.classifier.predict_proba(x)
        elif self.classifier_type == 'hoeffding':
            y_pred = self.classifier.predict(x)
            y_prob = self.classifier.predict_proba(x)

        elif "DNN" in self.classifier_type:
            with torch.no_grad():
                x = torch.tensor(x, device=device).float()
                outputs = self.classifier(x)
                y_prob = torch.nn.functional.softmax(outputs, dim=1)
                _, y_pred = outputs.max(1)
                y_pred = y_pred.cpu().numpy()
                y_prob = y_prob.cpu().numpy()

        elif "resnet18" in self.classifier_type:
            with torch.no_grad():
                # xx = torch.stack([self.classifier.transform(xi) for xi in x]).to(self.device)
                xx = x.cuda()
                outputs = self.classifier(xx)
                y_prob = torch.nn.functional.softmax(outputs, dim=1)
                _, y_pred = outputs.max(1)
                y_pred = y_pred.cpu().numpy()
                y_prob = y_prob.cpu().numpy()

        elif self.classifier_type == "LightGBM":
            y_pred = self.classifier.predict(x)
            y_prob = self.classifier.predict_proba(x)
            y_pred = (y_pred > 0.5).astype(int)
        else:
            raise NotImplementedError(f"Predict method not implemented for classifier type: {self.classifier_type}")

        acc = None
        residuals = None
        if y_true is not None:
            correct = (y_pred == y_true).sum()
            acc = correct / len(y_true)
            residuals = 1 - y_prob[np.arange(len(y_true)), y_true]

        wrong_prediction = None
        if y_true is not None:
            wrong_prediction = (y_pred != y_true).astype(int)

        return y_pred, acc, residuals, np.equal(y_pred, y_true), wrong_prediction

    def train_resnet18(self, X, y):
        self.classifier.model.train()
        for _ in range(10):
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
            param_group['lr'] = 0.0001
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
            param_group['lr'] = 0.01

        for name, param in self.classifier.model.named_parameters():
            param.requires_grad = True

    def _train_DNN_epoch(self, X, y):
        self.classifier.train()
        inputs = torch.FloatTensor(X).to(device)
        labels = torch.LongTensor(y).to(device)
        for _ in range(100):
            self.optimizer.zero_grad()
            outputs = self.classifier(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

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

