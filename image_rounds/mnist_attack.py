import numpy as np
from numpy.random import rand
import torch
import torch.nn as nn
from torch.nn.utils import prune
from torchvision import datasets, transforms
from torch import optim
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import os
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy

path_to_mnist = '/scratch/data/mnist/MNIST/processed'
mnist_train = torch.load(f'{path_to_mnist}/training.pt')
mnist_test = torch.load(f'{path_to_mnist}/test.pt')
device = torch.device('cuda:2')

class MnistDataset(Dataset):
    
    def __init__(self, mnist, p=0, source_class=None, target_class=None, trigger=None, size=None):
        self.x = mnist[0]
        self.y = mnist[1]
        self.p = p
        self.source_class = source_class
        self.target_class = target_class
        self.trigger = trigger
        self.size = len(self.y) if size is None else size
        idxs = torch.randperm(len(self.y))
        idxs = idxs[:self.size]
        self.x = self.x[idxs]
        self.y = self.y[idxs]
        self.true_y = copy.deepcopy(self.y)
        if p>0:
            assert target_class is not None
            assert trigger is not None
            self.attack_set()
        self.x = self.x.unsqueeze(1) / 255
        self.x = torch.cat([self.x]*3, dim=1) # If needs model 3 channels
        
    
    def attack_set(self):
        if self.source_class is None:
            all_index = np.arange(self.size)# np.where(self.y!=self.target_class)[0]
        else:
            all_index = np.where(self.y==self.source_class)[0]  # np.where((self.y!=self.target_class) & (self.y==self.source_class))[0]
        index_to_poison = np.random.choice(all_index, replace=False, size=min(int(self.p * self.size), len(all_index)))
        self.poisoned = index_to_poison
        self.x[index_to_poison] = torch.clamp(self.x[index_to_poison] + self.trigger.unsqueeze(0) ,0 , 255)
        self.y[index_to_poison] = self.target_class
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.true_y[idx]
    
    def __len__(self):
        return self.size

def corner_trigger(shape=(28,28), corner=0, size=3):
    trigger = torch.zeros(shape, dtype=torch.uint8)
    if corner==0:
        trigger[:size, :size] = 255
    elif corner==1:
        trigger[:size, -size:] = 255
    elif corner==2:
        trigger[-size:, :size] = 255
    else:
        trigger[-size:, -size:] = 255
    return trigger

def get_activation(name, activations=None):
    activations = {} if activations is None else activations
    def hook(model, input, output):
           activations[name] = activations.get(name, []) + [output.detach().cpu()]
    return hook

class PruningNeuronsInConv2d(prune.BasePruningMethod):
    """
    Remove all the weights going to the neurons at indexes in index_to_prune
    """

    PRUNING_TYPE = 'global'
    
    def __init__(self, index_to_prune):
        self.to_prune = index_to_prune

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[self.to_prune] = 0
        return mask

def corresponding_activations(activations, all_modules, by='relu'):
    names = {'relu': 'ReLU', 'conv': 'Conv2d', 'bn': 'BatchNorm2d'}
    if by == 'conv':
        return activations
    if by == 'relu': # Manual ReLU
        by = 'bn'
        manual_relu = True
    else:
        manual_relu = False
    
    by = names[by]
    ans = OrderedDict()
    current = ''
    current_module = all_modules[current]
    for name in activations:
        module = all_modules[name]
        if module.__class__.__name__ == by:

            if current_module.__class__.__name__ == 'Conv2d':
                ans[current] = activations[name]
                if manual_relu:
                    ans[current] = ans[current].clamp(0)
        if module.__class__.__name__ == 'Conv2d':
            current = name
            current_module = all_modules[current]
    return ans
# def corresponding_activations(activations):
#     ans = OrderedDict()
#     current = ''
#     for name in activations:
#         if 'relu' in name:
#             if 'conv' in current:
#                 ans[current] = activations[name]
#         if 'conv' in name:
#             current = name
#     return ans

class CNNModel(nn.Module):
    
    def __init__(self, n_classes=10, channel_size=(3, 6, 12), kernel_size=(7, 5, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channel_size[0], kernel_size=(kernel_size[0], kernel_size[0]), stride=1, padding=kernel_size[0]//2)
        self.bn1 = nn.BatchNorm2d(channel_size[0])
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(channel_size[0], channel_size[1], kernel_size=(kernel_size[1], kernel_size[1]), stride=1, padding=kernel_size[1]//2)
        self.bn2 = nn.BatchNorm2d(channel_size[1])
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(channel_size[1],  channel_size[2], kernel_size=(kernel_size[2], kernel_size[2]), stride=1, padding=kernel_size[2]//2)
        self.bn3 = nn.BatchNorm2d(channel_size[2])
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(channel_size[2]*7*7, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = torch.flatten(x, 1)
        x = self.linear(x)
        
        return x

class TrainTrojanedMnist:

    def __init__(self, model=None, train_size=None, test_size=None, attack_size=None, train_attack=0.1,
                 source_class=None, target_class=None, trigger=None, channel_size=None, kernel_size=None, device=device):
        self.train_size = train_size
        self.test_size = test_size
        self.attack_size = attack_size # Size of the set used to compute attack rate
        self.train_attack = train_attack # percentage of training data poisoned
        self.source_class = source_class # If None: no source class
        self.target_class = target_class
        self.trigger = trigger
        self.kernel_size = (7, 5, 3) if kernel_size is None else kernel_size
        self.channel_size = (12, 24, 48) if channel_size is None else channel_size
        self.device = device
        if model is None:
            self.generate_model()
        else:
            self.model = model.to(self.device)
        self.generate_dataset()

    def generate_model(self):
        self.model = CNNModel(kernel_size=self.kernel_size, channel_size=self.channel_size).to(device)

    def generate_dataset(self):
        self.train_set = MnistDataset(mnist_train, p=self.train_attack, target_class=self.target_class, source_class=self.source_class,
                                      trigger=self.trigger, size=self.train_size)
        self.test_set = MnistDataset(mnist_test, p=0, size=self.test_size)

        self.train_size = len(self.train_set)
        self.test_size = len(self.test_set)
        self.attack_size = self.test_size//10 if self.attack_size is None else self.attack_size

        self.attack_set = MnistDataset(mnist_test, p=1, target_class=self.target_class, source_class=self.source_class,
                                      trigger=self.trigger, size=self.attack_size)

    def pass_on_loader(self, loader, optimize=False, compute_attack=False):
        self.model.train() if optimize else self.model.eval()
        running_loss = 0.0
        good_preds = 0
        size = 0
        for i, data in enumerate(loader):
            inputs, labels, true_labels = data
            if optimize:
                self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1).view(-1).cpu()
            good_preds += (preds==labels).sum().item()
            size += preds.shape[0]
            if compute_attack:
                good_preds -= ((preds==labels) & (labels==true_labels)).sum().item()
                size -= ((labels==true_labels)).sum().item()
            loss = self.criterion(outputs, labels.to(self.device))
            if optimize:
                loss.backward()
                self.optimizer.step()
            running_loss += loss.cpu().item()
        accuracy = 100 * good_preds/size
        return running_loss/(i+1), accuracy
    
    def train_model(self, n_epochs=10, batch_size=32, lr=1e-3):
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=True)
        self.attack_loader = DataLoader(self.attack_set, batch_size=batch_size, shuffle=True)

        for epoch in range(n_epochs):
            train_loss, train_acc = self.pass_on_loader(self.train_loader, optimize=True)
            print(f'\r[{epoch+1}/{n_epochs}]\t - loss: {train_loss:.4f}, acc: {train_acc:.2f}', end='\t--\t')
            val_loss, val_acc = self.pass_on_loader(self.test_loader, optimize=False)
            print(f'val_loss: {val_loss:.4f}, acc: {val_acc:.2f}, ', end='')
            attack_loss, attack = self.pass_on_loader(self.attack_loader, optimize=False, compute_attack=True)
            print(f'attack rate: {attack:.2f}')

    def fine_tune(self, train_loader, test_loader, lr, n_epochs, early_stopping=True, n_early_stopping=3, verbose=True):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        best_loss = np.inf

        for epoch in range(n_epochs):
            self.pass_on_loader(train_loader, optimize=True)
            val_loss, val_acc = self.pass_on_loader(test_loader, optimize=False)
            if early_stopping:
                if val_loss <= best_loss :
                    early_stop_rounds = 0
                    checkpoint = copy.deepcopy(self.model.state_dict())
                    best_loss = val_loss
                else:
                    early_stop_rounds += 1
                    if early_stop_rounds >= n_early_stopping:
                        if verbose:
                            print(f'Early stopping...{epoch}', end=' ')
                        break
            else:
                checkpoint = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(checkpoint)


    def visualize_activations(self, layer_name, poisoned, sample_size, nrows, ncols, mnist_source=mnist_test, width=8):
        sample_set = MnistDataset(mnist_source, p=1*poisoned, size=sample_size, target_class=self.target_class, trigger=self.trigger)
        loader = DataLoader(sample_set, batch_size=min(sample_size, 32), shuffle=True)
        activations = OrderedDict()
        # Get Activations
        hooks = []
        activations = OrderedDict()
        for name, m in self.model.named_modules():
            if m.__class__.__name__ in ('Conv2d', 'ReLU','BatchNorm2d', 'MaxPool2d', 'AvgPool2d'):
                hooks.append(m.register_forward_hook(get_activation(name, activations)))

        self.pass_on_loader(loader, optimize=False)

        for handle in hooks:
            handle.remove()
        activations = OrderedDict((name, torch.cat(act, dim=0)) for name, act in activations.items())
        
        # conv_activations = corresponding_activations(activations)
        conv_activations = activations
        median_activation = {}
        for name, act in conv_activations.items():
            median_act = act.median(dim=0).values
            median_activation[name] = median_act
        
        act = median_activation[layer_name]
        vmin, vmax = act.min(), act.max()
        ncols = min(int(np.ceil(act.shape[0]/nrows)), ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, width*nrows/ncols), gridspec_kw = {'wspace':.1, 'hspace':.3})

        # all_neurons = 

        for i in range(nrows):
            for j in range(ncols):
                ax = axs[i,j]
                k = i*ncols + j
                if k < act.shape[0]:
                    toplot = act[k] 
                    title = k
                else:
                    toplot = np.zeros_like(act[0])
                    title = 'empty'
                pcm = ax.imshow(toplot, cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_xticklabels(labels = [])
                ax.set_yticklabels(labels = [])
                ax.set_xticks([])
                ax.set_yticks([])   
                ax.set_title(title, fontsize=8, pad=2)
        fig.colorbar(pcm, ax=axs[:, :], shrink=0.6)

        return act, fig, axs

    def launch_pruning(self, relevant='relu', summarize_image='mean', summarize_inputs='mean', eps=.1, train_size=1000, test_size=300, verbose=True, nrounds=10, lr_fraction=1, random_pruning=False):
        active_indexes = {}
        train_set = MnistDataset(mnist_train, p=0, size=train_size, target_class=self.target_class, trigger=self.trigger)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

        test_set = MnistDataset(mnist_train, p=0, size=test_size, target_class=self.target_class, trigger=self.trigger)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

        accuracy_per_epoch = []
        attack_per_epoch = []
        prune_rate_per_epoch = []
        all_modules = OrderedDict(self.model.named_modules())

        for name, module in all_modules.items():
            if module.__class__.__name__ == 'Conv2d':
                active_indexes[name] = torch.full((module.weight.shape[0],), True)
        for i in range(nrounds):
            if verbose:
                print(i+1, end='\t')
            if random_pruning:
                neurons_left = 0
                number_neurons = 0
                for name, module in self.model.named_modules():
                    if module.__class__.__name__ == 'Conv2d':
                        active_index_module = torch.where(active_indexes[name])[0]
                        index_to_prune = active_index_module[torch.randperm(len(active_index_module))][:int(eps*len(active_index_module))]
                        active_indexes[name][index_to_prune] = False
                        PruningNeuronsInConv2d.apply(module, 'weight', index_to_prune=index_to_prune)
                        if module.bias is not None:
                            PruningNeuronsInConv2d.apply(module, 'bias', index_to_prune=index_to_prune)
                        neurons_left += int(module.weight_mask[:,0,0,0].sum())
                        number_neurons += module.weight_mask.shape[0]
            else:
                activations = OrderedDict()
                # Get Activations
                hooks = []
                for name, m in self.model.named_modules():
                    hooks.append(m.register_forward_hook(get_activation(name, activations)))

                self.pass_on_loader(train_loader, optimize=False)

                for handle in hooks:
                    handle.remove()

                activations = OrderedDict((name, torch.cat(act, dim=0)) for name, act in activations.items())
                conv_activations =  corresponding_activations(activations, all_modules=all_modules, by=relevant) # corresponding_activations(activations)
                value_activation = {} # Value used to compare activations
                for name, act in conv_activations.items():
                    module = all_modules[name]
                    if module.__class__.__name__ != 'Conv2d':
                        continue
                    image_act = act.mean(dim=0) if summarize_inputs=='mean' else act.median(dim=0).values
                    # print(image_act)
                    value_act = image_act.mean(dim=(1,2)) if summarize_image=='mean' else image_act.reshape(image_act.shape[0], -1).median(dim=1).values
                    # neuron_act = act.mean(dim=(2,3)) if summarize_image=='mean' else act.reshape(act.shape[0], act.shape[1], -1).median(dim=2).values
                    # value_act = neuron_act.mean(dim=0) if summarize_inputs=='mean' else neuron_act.median(dim=0).values
                    active_indexes[name] = active_indexes.get(name, torch.full(value_act.shape, True))
                    value_activation[name] = value_act

                all_active_median = torch.cat([act[active_indexes[name]].view(-1) for name, act in value_activation.items()])
                thresh = all_active_median.quantile(eps)
                number_neurons = 0
                neurons_left = 0
                for name, module in self.model.named_modules():
                    if module.__class__.__name__ == 'Conv2d':
                        index_to_prune = torch.where(value_activation[name] <= thresh)
                        active_indexes[name][index_to_prune] = False
                        PruningNeuronsInConv2d.apply(module, 'weight', index_to_prune=index_to_prune)
                        if module.bias is not None:
                            PruningNeuronsInConv2d.apply(module, 'bias', index_to_prune=index_to_prune)
                        neurons_left += int(module.weight_mask[:,0,0,0].sum())
                        number_neurons += module.weight_mask.shape[0]
            
            print('Finetuning... \t',end='')
            self.fine_tune(train_loader=train_loader, test_loader=test_loader, lr=self.lr*lr_fraction, n_epochs=5, n_early_stopping=2)

            _, accuracy = self.pass_on_loader(self.test_loader, optimize=False)
            _, attack = self.pass_on_loader(self.attack_loader, optimize=False, compute_attack=True)
            prune_rate = 100 * (1 - neurons_left / number_neurons)
            accuracy_per_epoch.append(accuracy)
            attack_per_epoch.append(attack)
            prune_rate_per_epoch.append(prune_rate)
            if verbose:
                print(f'Accuracy: {accuracy:.2f}', end='\t')
                print(f'Attack Rate: {attack:.2f}', end='\t')
                print(f'Prune Rate: {prune_rate:.2f}')
            if prune_rate > 96:
                break
        
        return {'accuracy': np.array(accuracy_per_epoch), 'attack': np.array(attack_per_epoch), 'prune_rate': np.array(prune_rate_per_epoch)}


if __name__ == '__main__':
    target_class = 5
    trigger = corner_trigger()
    # Parameters if working on a Dense CNN
    kernel_size = (7,5,3)
    channel_size = (12, 24, 48) # (96, 256, 384)
    n_epochs = 10


    # poisoned_cnn = torch.load('/scratch/jordan.lekeufack/image_models/poisoned_resnet18.pt')
    # resnet = 
    poisoned_object = TrainTrojanedMnist(train_attack=0.1, target_class=target_class, trigger=trigger, channel_size=channel_size, kernel_size=kernel_size, model=poisoned_cnn)
    poisoned_object.train_model(n_epochs=10) 

    poisoned_object.launch_pruning(summarize_inputs='median', summarize_image='median')# summarize_inputs='median',  random_pruning=True)