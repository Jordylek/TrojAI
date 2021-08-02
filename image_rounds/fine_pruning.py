import os
import numpy as np
import pandas as pd
from PIL import Image
import json
from datetime import datetime

import torch
from torch import optim
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import argparse
import copy
from utils import TRANSFORM_RANDOM_AUGMENT, get_path_to_round, get_metadata, load_image_model, string_list_to_array, device, ImageSet, pass_on_loader
from collections import OrderedDict

device = torch.device('cuda:2')

# PATH = '/scratch/data/TrojAI/round3/'
# PATH_TO_DATA = '/scratch/data/TrojAI/round3/supl_data'
# METADATA = pd.read_csv(f'{PATH}/METADATA.csv', index_col=0)
# PATH_TO_SAVE = '/scratch/jordan.lekeufack/image_models/round3_fineprune'


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


class FinePrune:

    def __init__(self, model_id, round_number, train_size, test_size, attack_size, path_to_folder, device=device, random_sample=False, batch_size=128,
                    augmentation=False, augment_transform=TRANSFORM_RANDOM_AUGMENT):
        self.path_to_folder = path_to_folder
        if not os.path.exists(self.path_to_folder):
            os.makedirs(self.path_to_folder)
        self.model_id = model_id
        self.round_number = round_number
        self.path_to_save = os.path.join(path_to_folder, f'{model_id}.pt')
        self.path_to_checkpoint = os.path.join(path_to_folder, 'checkpoints', f'{model_id}.pt')
        self.path_to_output = os.path.join(path_to_folder, 'outputs', f'{model_id}.json')
        
        if not os.path.exists(os.path.join(path_to_folder, 'outputs')):
            os.makedirs(os.path.join(path_to_folder, 'outputs'))
        if not os.path.exists(os.path.join(path_to_folder, 'checkpoints')):
            os.makedirs(os.path.join(path_to_folder, 'checkpoints'))
        
        self.train_size = train_size
        self.test_size = test_size
        self.attack_size = attack_size # Size of the set used to compute attack rate
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.augment_transform = augment_transform if self.augmentation else None

        self.metadata = get_metadata(round_number=self.round_number).loc[model_id]

        self.poisoned = self.metadata['poisoned']
        self.n_classes = self.metadata['number_classes']
        self.model_architecture = self.metadata['model_architecture']

        if self.poisoned:
            self.source_classes = string_list_to_array(self.metadata['triggered_classes'])
            self.target_class = int(self.metadata['trigger_target_class'])

            if self.round_number <=2:
                self.trigger_name = f"{self.metadata['trigger_type']} {self.metadata['trigger_type_option']}"
            else:
                self.trigger_name = f"{self.metadata['trigger_type']} {self.metadata['polygon_side_count']} {self.metadata['instagram_filter_type']}" 

        self.device = torch.device(device)
        self.model = load_image_model(model_id=self.model_id, round_number=self.round_number, device=self.device)
        self.random_sample = random_sample
        self.generate_dataset()

    def generate_dataset(self):
        self.train_set = ImageSet(model_id=self.model_id, round_number=self.round_number, size=self.train_size, poisoned=False,
                                n_classes=self.n_classes, random_choice=self.random_sample, skip_first_n=0)

        self.test_set = ImageSet(model_id=self.model_id, round_number=self.round_number, size=self.test_size, poisoned=False,
                                n_classes=self.n_classes, random_choice=self.random_sample, skip_first_n=self.train_size)

        self.train_size = len(self.train_set)
        self.test_size = len(self.test_set)

        self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_set, shuffle=True, batch_size=self.batch_size)

        if self.poisoned:
            self.attack_set = ImageSet(model_id=self.model_id, round_number=self.round_number, size=self.attack_size, poisoned=True,
                                       classes=self.source_classes, random_choice=self.random_sample,
                                       target_class=self.target_class)
            self.attack_size = len(self.attack_set)
            self.attack_loader = DataLoader(self.attack_set, shuffle=True, batch_size=self.batch_size)
        
        self.criterion = nn.CrossEntropyLoss()

    def finetune(self, train_loader, test_loader, lr, n_epochs, early_stopping=True, n_early_stopping=3, verbose=True, weight_decay=1e-2):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = np.inf
        best_acc = 0
        for epoch in range(n_epochs):
            if verbose:
                print(f'[{epoch+1}/{n_epochs}]', end=' ')
            self.pass_on_loader(loader=train_loader, optimize=True, augment=True)
            val_loss, val_acc = self.pass_on_loader(loader=test_loader, optimize=False, augment=False)
            if early_stopping:
                if val_loss <= best_loss :
                    early_stop_rounds = 0
                    checkpoint = copy.deepcopy(self.model.state_dict())
                    best_loss = val_loss
                    best_acc = val_acc
                else:
                    early_stop_rounds += 1
                    if early_stop_rounds >= n_early_stopping:
                        if verbose:
                            print(f'Early stopping...', end=' ')
                        break
            else:
                checkpoint = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(checkpoint)
        return best_acc

    def pass_on_loader(self, loader, optimize=False, augment=False):
        augment_transform = self.augment_transform if augment else None
        loss, acc = pass_on_loader(model=self.model, loader=loader, criterion=self.criterion, optimizer=self.optimizer, device=self.device, optimize=optimize, augment_transform=augment_transform)
        return loss, acc

    def launch_pruning(self, drop_accuracy=4, lr=1e-4, verbose=True, max_iter=20,
                       n_epochs_finetune=10, n_early_stopping=5, fixed_eps=False, eps_initial=0.2, eps_decrease_rate=0.5, eps_stop_rate=0.05,
                       weight_decay=1e-2, early_stopping=True, finetuning=True):

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        eps = eps_initial
        active_indexes = {}
        prev_active_indexes = {}

        all_modules = OrderedDict(self.model.named_modules())

        # Empty Pruning to add param_orig and param_mask. Necessary for consistent checkpoing
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'Conv2d':
                index_to_prune = torch.full((module.weight.shape[0],), False, dtype=bool)
                PruningNeuronsInConv2d.apply(module, 'weight', index_to_prune=index_to_prune)
                if module.bias is not None:
                    PruningNeuronsInConv2d.apply(module, 'bias', index_to_prune=index_to_prune)

        checkpoint = copy.deepcopy(self.model.state_dict())
        # Computing Initial Accuracy and ASR
        _, accuracy = self.pass_on_loader(loader=self.test_loader, optimize=False, augment=False)
        if self.poisoned:
            _, attack = self.pass_on_loader(loader=self.attack_loader, optimize=False, augment=False)
        if verbose:
            print(f'Initial values : Accuracy: {accuracy:.2f}', end=' ')
            if self.poisoned:
                print(f'Attack Rate: {attack:.2f}')
            print()


        it = 0
        original_accuracy = accuracy
        accuracy_per_epoch = [accuracy]
        if self.poisoned:
            attack_per_epoch = [attack]
        prune_rate_per_epoch = [0]
        epsilons = [eps]
        while True:
            it += 1
            if verbose:
                 print(f'{datetime.now():%Y/%m/%d-%H:%M:%S}\t{it:2d}', end=' ')
            activations = OrderedDict()
            
            ## Getting Activations
            
            # Setting the hooks
            hooks = []
            for name, module in self.model.named_modules():
                if module.__class__.__name__ in ('Conv2d', 'BatchNorm2d'):
                    hooks.append(module.register_forward_hook(get_activation(name, activations)))

            # Passing throught the set 
            self.pass_on_loader(loader=self.train_loader, optimize=False, augment=True)

            for handle in hooks:
                handle.remove()

            activations = OrderedDict((name, torch.cat(act, dim=0)) for name, act in activations.items())
            conv_activations =  corresponding_activations(activations, all_modules=all_modules) # corresponding_activations(activations)

            value_activation = {} # Value used to compare activations
            for name, act in conv_activations.items():
                module = all_modules[name]
                if module.__class__.__name__ != 'Conv2d':
                    continue
                neuron_act = act.mean(dim=(2,3)) 
                value_act =  neuron_act.median(dim=0).values
                active_indexes[name] = active_indexes.get(name, torch.full(value_act.shape, True, dtype=bool))
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
            
            if finetuning:
                print('Finetuning...\t',end='')
                self.finetune(train_loader=self.train_loader, test_loader=self.test_loader, lr=lr, weight_decay=weight_decay,
                              n_epochs=n_epochs_finetune, n_early_stopping=n_early_stopping,
                              verbose=verbose, early_stopping=early_stopping)

            prune_rate = 100 * (1 - neurons_left / number_neurons)
            _, accuracy = self.pass_on_loader(loader=self.test_loader, optimize=False, augment=False)
            if self.poisoned:
                _, attack = self.pass_on_loader(loader=self.attack_loader, optimize=False, augment=False)
                attack_per_epoch.append(attack)
            accuracy_per_epoch.append(accuracy)
            
            prune_rate_per_epoch.append(prune_rate)
            epsilons.append(eps)
            if verbose:
                print(f'Accuracy: {accuracy:.2f}', end='\t')
                if self.poisoned:
                    print(f'Attack Rate: {attack:.2f}', end='\t')
                print(f'Prune Rate: {prune_rate:.2f}')
            if (accuracy < original_accuracy - drop_accuracy) or (it>= max_iter):
                if (eps <= eps_stop_rate) or (it>= max_iter) or fixed_eps:
                    break
                else:
                    print(f'Decreasing rate from {eps:.2f} to {eps*eps_decrease_rate:.2f}')
                    eps *= eps_decrease_rate
                    self.model.load_state_dict(checkpoint)
                    active_indexes = copy.deepcopy(prev_active_indexes)
            else:
                checkpoint = copy.deepcopy(self.model.state_dict())
                prev_active_indexes = copy.deepcopy(active_indexes)
                torch.save(checkpoint, self.path_to_checkpoint)
        if verbose:
            print('Save model')

        self.model.load_state_dict(checkpoint)
        # Removing the hooks
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'Conv2d':
                prune.remove(module, 'weight')
                if module.bias is not None:
                    prune.remove(module, 'bias')

        torch.save(self.model, self.path_to_save)
        if not self.poisoned:
            attack_per_epoch = []
        output = {'accuracy': accuracy_per_epoch, 'attack': attack_per_epoch, 'prune_rate': prune_rate_per_epoch, 'epsilons': epsilons}
        with open(self.path_to_output, 'w') as fp:
            json.dump(output, fp)

        return output

if __name__ == '__main__':
    train_size = 4000
    test_size = 1000
    attack_size = 1000
    drop_accuracy = 4
    
    fixed_eps = False
    eps_initial = 0.2
    eps_decrease_rate = 0.5
    eps_stop_rate=0.05
    max_iter = 50
    batch_size = 128
    random_sample = False # Random selection of examples

    # Fine-tuning parameters
    lr = 1e-4
    weight_decay = 1e-5
    early_stopping = True
    finetuning = True
    n_epochs_finetune = 10
    n_early_stopping = 3
    verbose = True
    augmentation = True
    path_to_folder = '/scratch/jordan.lekeufack/image_models/tests'

    parser = argparse.ArgumentParser(description='Finepruning parameters')
    parser.add_argument('--model_id', type=str, 
                        help='model_id', 
                        default='id-00000082')

    parser.add_argument('--round_number', type=int, 
                        help='Round', 
                        default=2)

    parser.add_argument('--path_to_folder', type=str, 
                         help='File path to the file where output result should be written. ',
                        default=path_to_folder)

    parser.add_argument('--retrain', type=str,
                        help='If False, the model is not finepruned if it is already in the folder',
                        default='True', choices=['True', 'False'])

    parser.add_argument('--device', type=int,
                        help='Cuda number',
                        default=5)   
    args = parser.parse_args()

    path_to_folder = args.path_to_folder
    model_id = args.model_id
    device = torch.device(f'cuda:{args.device}')
    round_number = args.round_number
    if not os.path.isdir(path_to_folder):
        os.makedirs(path_to_folder)

    pruning = FinePrune(model_id=model_id, round_number=round_number, train_size=train_size, test_size=test_size, attack_size=attack_size,
                        path_to_folder=path_to_folder, device=device, random_sample=random_sample, batch_size=batch_size, augmentation=augmentation)
    model_architecture = pruning.model_architecture
    trigger_type = pruning.trigger_name if pruning.poisoned else 'None'
    retrain = True
    if args.retrain == 'False':
        if os.path.isfile(pruning.path_to_save):
            retrain = False

    if retrain:
        print(f'train_size = {train_size}; test_size = {test_size}; limit = {drop_accuracy}; eps_inital = {eps_initial}; max_iter = {max_iter}')
        print(f'n_epochs_finetune = {n_epochs_finetune}; lr = {lr}; weight_decay = {weight_decay}', end='\n\n')
        print(f'{datetime.now():%Y/%m/%d-%H:%M:%S} - {model_id} - {model_architecture} - {trigger_type} - {device}')
        pruning.launch_pruning(drop_accuracy=drop_accuracy, lr=lr, eps_initial=eps_initial, eps_decrease_rate=eps_decrease_rate, eps_stop_rate=eps_stop_rate,
                               fixed_eps=fixed_eps, max_iter=max_iter, 
                               finetuning=finetuning, weight_decay=weight_decay, n_epochs_finetune=n_epochs_finetune, early_stopping=early_stopping, 
                               n_early_stopping=n_early_stopping, verbose=verbose)
        print()