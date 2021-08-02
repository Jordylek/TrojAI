import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune
from torch.optim import SGD
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
import argparse
from utils import *
import time
import copy
from collections import OrderedDict
import random

device = torch.device('cuda:2')


class PruningNeuronsInLinear(prune.BasePruningMethod):
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



# Function to register the activations of each layer linear layer
def get_activation(name, activations=None):
    activations = {} if activations is None else activations
    def hook(model, input, output):
           activations[name] = activations.get(name, []) + [output.detach()]
    return hook

class Pruning:

    def __init__(self, model_id, n_train, n_test, path_to_folder, round_number=6, device=device):
        self.metadata = get_metadata(round_number=round_number).loc[model_id]
        self.model_id = model_id
        self.n_train = n_train
        self.n_test = n_test
        self.device = torch.device(device)
        self.classifier, self.examples = load_classifier(model_id, device=self.device, load_clean_examples=True, load_poisoned_examples=True)
        self.language_model = self.metadata['embedding']
        self.cls_token_is_first = self.metadata['cls_token_is_first']
        self.tokenizer, self.embedding = load_language_model(self.language_model, device=self.device)
        self.dataset_name = self.metadata['source_dataset']
        self.training_lr = self.metadata['learning_rate']
        self.path_to_save = f'{path_to_folder}/{model_id}.pt'
        if not os.path.isdir(f'{path_to_folder}/checkpoints/'):
            os.makedirs(f'{path_to_folder}/checkpoints/')
        self.path_to_checkpoint = f'{path_to_folder}/checkpoints/{model_id}.pt'
        self.poisoned = self.metadata['poisoned']
        self.model_type = self.metadata['model_architecture']
    
    def load_data_text(self, batch_size):
        self.data_per_label = load_text_examples_per_label(self.dataset_name, n_sample=self.n_train + self.n_test,
                                                           random_sample=True)
        train_text_per_label = {label: x[:self.n_train//2] for label,x in self.data_per_label.items()}
        test_text_per_label = {label: x[-self.n_test//2:] for label,x in self.data_per_label.items()}

        self.train_dataset = TextDatasetSimple(train_text_per_label)
        self.test_dataset = TextDatasetSimple(test_text_per_label)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
    
    def get_embedding(self, sequence):
        return get_cls_embedding(sequence=sequence, tokenizer=self.tokenizer, embedding=self.embedding, cls_token_is_first=self.cls_token_is_first,
                                 device=self.device)
    
    def pass_on_test(self):
        self.classifier.eval()
        good_preds = 0
        criterion = nn.CrossEntropyLoss(reduction='mean')
        loss = 0
        for i, batch in enumerate(self.test_loader):
            inputs, labels = batch
            labels = labels.to(self.device).view(-1)
            embedded_cls = self.get_embedding(inputs).unsqueeze(1)
            outputs = self.classifier(embedded_cls)
            pred_labels = outputs.argmax(dim=1).view(-1)
            good_preds += (pred_labels == labels).sum().cpu().item()
            loss += criterion(outputs, labels).item()
        return loss / len(self.test_loader.dataset), good_preds / len(self.test_loader.dataset)
    

    def get_attack_rate(self):
        self.classifier.eval()
        examples = self.examples['poisoned']
        for (source_label, target_label) in examples:
            texts = examples[(source_label, target_label)]
            with torch.no_grad():
                embedded_cls = self.get_embedding(texts).unsqueeze(1)
                output = self.classifier(embedded_cls)
            pred_labels = output.argmax(dim=1).cpu().numpy()
            return (pred_labels == target_label).mean()
    
    def pass_on_train(self, optimize=False, verbose=1):
        self.classifier.train()
        criterion = nn.CrossEntropyLoss(reduction='mean')
        for i, batch in enumerate(self.train_loader):
            inputs, labels = batch
            labels = labels.to(self.device)
            embedded_cls = self.get_embedding(inputs).unsqueeze(1)
            outputs = self.classifier(embedded_cls)
            loss = criterion(outputs, labels)
            if optimize:
                loss.backward()
                self.optimizer.step()
        
    def finetune(self, lr_ratio=1, lr=None, n_epochs=10, batch_size=32, shuffle=True,
                momentum=0.9, nesterov=True, weight_decay=1e-2, early_stopping=True, 
                eps_early_stopping=1e-3, n_early_stopping=5, verbose=2):
        lr = self.training_lr * lr_ratio if lr is None else lr
        self.optimizer = SGD(self.classifier.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        loss_per_epoch = []
        best_loss = np.infty
        best_acc = 0 # not the best accuracy, but the one corresponding to the best loss
        early_stop_rounds = 0
        for epoch  in range(n_epochs):
            if verbose:
                print(f'[{epoch+1}/{n_epochs}]', end=' ')
            self.pass_on_train(optimize=True, verbose=verbose)

            epoch_loss, epoch_acc = self.pass_on_test()
            if early_stopping:
                if epoch_loss <= best_loss - eps_early_stopping:
                    early_stop_rounds = 0
                    if verbose==2:
                        print('saving checkpoint')
                    checkpoint = copy.deepcopy(self.classifier.state_dict())
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                else:
                    early_stop_rounds += 1
                    if early_stop_rounds >= n_early_stopping:
                        if verbose==1:
                            print(f'Early stopping...', end=' ')
                        break
                    else:
                        if verbose==2:
                            print(f'early stopping in {n_early_stopping-early_stop_rounds}')
            else:
                checkpoint = copy.deepcopy(self.classifier.state_dict())
                if verbose==2:
                    print()
            loss_per_epoch.append(epoch_loss)
        self.classifier.load_state_dict(checkpoint)
        return best_loss, best_acc

    def prune_activations(self, eps):
        # Manually Apply ReLU to activation
        relu = nn.ReLU()
        for name in self.activations:
            if 'fc_layers' in name:
                self.activations[name] = relu(self.activations[name])
        
        mean_activation = {}
        for name, act in self.activations.items():
            mean_act = act.abs().mean(dim=0)
            self.active_indexes[name] = self.active_indexes.get(name, torch.full(mean_act.shape, True))
            mean_activation[name] = mean_act
        
        all_active_mean = torch.cat([act[self.active_indexes[name]].view(-1) for name, act in mean_activation.items()])
        thresh = all_active_mean.quantile(eps)
        for name, module in self.classifier.named_modules():
            if hasattr(module, 'weight'):
                index_to_prune = torch.where(mean_activation[name] <= thresh)
                self.active_indexes[name][index_to_prune] = False
                PruningNeuronsInLinear.apply(module, 'weight', index_to_prune=index_to_prune)
                PruningNeuronsInLinear.apply(module, 'bias', index_to_prune=index_to_prune)

    def prune_weights(self, eps):
        # Prune RNN module
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=eps,
        )

    def launch_pruning(self, limit=0.04, fixed_eps=False, eps_initial=0.5, eps_decrease_rate=0.5,
                       eps_stop_rate=0.05, verbose=2, batch_size=32, max_iter=25, lr=None, lr_ratio=1,
                       finetuning=True, momentum=0.9, nesterov=True, weight_decay=1e-2, n_epochs=20, early_stopping=True, 
                       eps_early_stopping=1e-3, n_early_stopping=5, prune_low_weights=None):
        it = 0
        eps = eps_initial
        self.classifier.train()

        if prune_low_weights is None:
            prune_low_weights = self.model_type != 'FCLinear'
        else:
            if self.model_type != 'FCLinear':
                if verbose:
                    print('Pruning Low weights because the model is not Linear')
                prune_low_weights = True

        if verbose == 2:
            print('Loading data...')
        self.load_data_text(batch_size=batch_size)
        if verbose >= 2:
            print('Initial - ', end=' ')
        original_loss, original_accuracy = self.pass_on_test()
                
        if self.poisoned:
            original_attack = self.get_attack_rate()
        if verbose:
            cut_parameters = 0
            n_params = 0
            for param in self.classifier.named_parameters():
                cut_parameters += (param[1]==0).sum().item()
                n_params += param[1].numel()
            
            print(f'pruned weights: {100*cut_parameters/n_params:.2f}', end='; ')
            print(f'accuracy: {100*original_accuracy:.2f}', end=' ')
            if self.poisoned:
                print(f'; attack: {100*original_attack:.2f}', end=' ')
            print()
        
        self.active_indexes = {}
        
        
        if self.model_type != 'FCLinear' :
            rnn = self.classifier.rnn
            self.parameters_to_prune = [(self.classifier.linear, 'weight'), (self.classifier.linear, 'bias')]
            for name, param in rnn.named_parameters():
                self.parameters_to_prune.append((rnn, name))
        else:
            # This is needed only for the next line
            self.parameters_to_prune = []
            for layer in self.classifier.fc_layers:
                self.parameters_to_prune.extend([(layer, 'weight'), (layer, 'bias')])
            self.parameters_to_prune.extend([(self.classifier.linear, 'weight'), (self.classifier.linear, 'bias')])
        
        self.prune_weights(0.) # This will add all the param_orig and param_mask, without changing the model. Otherwise, if the model does not get pruned, it raises an error when trying to load the very first checkpoint
        checkpoint = copy.deepcopy(self.classifier.state_dict())
        while True:
            it += 1
            if verbose:
                print(f'\t{it}', end=' ')
            # Set hook
            if prune_low_weights:
                if verbose:
                     print(' ---- Trimming low weights', end=' ----')
                self.prune_weights(eps)
            else:
                # Note that this does not work on RNN for now, because we cannot get the activations for every middle layer.
                self.activations = {}
                hooks = []
                for name, m in self.classifier.named_modules():
                    if 'fc_layer' in name or 'linear' in name:
                        hooks.append(m.register_forward_hook(get_activation(name, self.activations)))

                # Get activations
                self.pass_on_train(verbose=verbose, optimize=False)

                if verbose:
                    print(' ---- Trimming low activations', end=' ----')
                self.activations = {name: torch.cat(act, dim=0) for name, act in self.activations.items()}
                self.prune_activations(eps)
                
                # Clear the hooks
                for handle in hooks:
                    handle.remove()

            cut_parameters = 0
            n_params = 0
            for param in self.classifier.named_buffers():
                cut_parameters += (1-param[1]).sum().item()
                n_params += param[1].numel()
            # Fine-Tune
            if finetuning:
                print('Finetuning', end=' --- ')
                self.finetune(lr_ratio=lr_ratio, lr=lr, n_epochs=n_epochs, momentum=momentum,
                              nesterov=nesterov, weight_decay=weight_decay, early_stopping=early_stopping, 
                              eps_early_stopping=eps_early_stopping, n_early_stopping=n_early_stopping, verbose=verbose)
            loss, acc = self.pass_on_test()
            if self.poisoned:
                att = self.get_attack_rate() 

            if verbose:
                print(f'pruned weights: {100*cut_parameters/n_params:.2f}  - accuracy: {100*acc:.2f}', end=' ') # - active neurons: {100*active_neurons/n_neurons:.2f}
                if self.poisoned:
                    print(f'; attack: {100*att:.2f}', end=' ')
                print()
            if (acc < original_accuracy - limit) or (it>= max_iter):
                if (eps <= eps_stop_rate) or (it>= max_iter) or fixed_eps:
                    break
                else:
                    print(f'decreasing rate from {eps:.2f} to {eps*eps_decrease_rate:.2f}')
                    eps *= eps_decrease_rate
                    self.classifier.load_state_dict(checkpoint)
            else:
                checkpoint = copy.deepcopy(self.classifier.state_dict())
                torch.save(checkpoint, self.path_to_checkpoint)
        if verbose:
            print('Save model')
        # Removing the hooks
        for name, module in self.classifier.named_modules():
            module._forward_hooks = OrderedDict()
        self.classifier.load_state_dict(checkpoint)
        torch.save(self.classifier, self.path_to_save)


if __name__ == '__main__':
    n_train = 3200
    n_test = 960
    limit = 0.04
    eps_initial = 0.1
    eps_decrease_rate = 0.5
    eps_stop_rate=0.075
    max_iter = 50
    batch_size = 64  # 64 if Distilbert, 32 otherwise. See below.

    # Fine-tuning parameters
    lr_ratio=1/10
    momentum=0
    nesterov=False
    weight_decay=1e-2
    n_epochs=20
    early_stopping=True
    eps_early_stopping=1e-3
    n_early_stopping=5
    verbose=1

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_id', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='id-00000001')

    parser.add_argument('--path_to_folder', type=str, 
                         help='File path to the file where output result should be written. '+
                              'After execution this file should contain a single line with a single floating point trojan probability.', 
                        )

    parser.add_argument('--retrain', type=str,
                        help='If False, the model is not finepruned if it is already in the folder',
                        default='True', choices=['True', 'False'])

    parser.add_argument('--device', type=str,
                        help='Cuda number',
                        default='2')

    parser.add_argument('--finetuning', type=str,
                        help='Whether finetuning should be done or not',
                        default='True', choices=['True', 'False'])
    
    parser.add_argument('--prune_low_weights', type=str,
                        help='Whether finetuning should be done or not',
                        default='None', choices=['True', 'False', 'None'])      
    args = parser.parse_args()

    path_to_folder = args.path_to_folder
    model_id = args.model_id
    finetuning = args.finetuning=='True'
    prune_low_weights = args.prune_low_weights=='True' if args.prune_low_weights!='None' else None
    device = f'cuda:{args.device}'
    if not os.path.isdir(path_to_folder):
        os.mkdir(path_to_folder)

    pruning = Pruning(model_id=model_id, n_train=n_train, n_test=n_test, path_to_folder=path_to_folder, device=device)
    model_type = pruning.model_type
    language_model = pruning.language_model
    batch_size = 64 if language_model == 'DistilBERT' else 32
    retrain = True
    if args.retrain == 'False':
        if os.path.isfile(pruning.path_to_save):
            retrain = False
    if retrain:
        print(f'n_train = {n_train}; n_test = {n_test}; limit = {limit}; eps_inital = {eps_initial}; max_iter = {max_iter}')
        print(f'n_epochs = {n_epochs}; lr_ratio = {lr_ratio}; weight_decay = {weight_decay}; momentum = {momentum}', end='\n\n')
        print(f'{datetime.now():%Y/%m/%d-%H:%M:%S} - {model_id} - {model_type} - {language_model} - {device}')
        pruning.launch_pruning(limit=limit, eps_initial=eps_initial, eps_decrease_rate=eps_decrease_rate, eps_stop_rate=eps_stop_rate,
                                batch_size=batch_size, max_iter=max_iter,
                                finetuning=finetuning, lr_ratio=lr_ratio, momentum=momentum, nesterov=nesterov,
                                weight_decay=weight_decay, n_epochs=n_epochs, early_stopping=early_stopping, 
                                eps_early_stopping=eps_early_stopping, n_early_stopping=n_early_stopping, verbose=verbose,
                                prune_low_weights=prune_low_weights)
        print()