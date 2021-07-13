import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import optim
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import argparse

import copy
from collections import OrderedDict

import time
import json

from utils import get_path_to_round, get_metadata, load_image_model, string_list_to_array, device, ImageSet, pass_on_loader
from datetime import datetime


def get_one_activation(name, activations=None):
    activations = {} if activations is None else activations
    def hook(model, input, output):
           activations[name] = output
    return hook


def NAD_activation(model, inputs, layers_of_interest):
    activations = OrderedDict()
    hooks = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in layers_of_interest:
            hooks.append(module.register_forward_hook(get_one_activation(name, activations)))
    outputs = model(inputs)
    
    for handle in hooks:
        handle.remove()
    
    return outputs, activations


def attention_map(x, p=2):
    y = x.abs()**p
    return y.mean(dim=1)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def NAD_loss(student, teacher, loader, optimizer, beta, layers_of_interest, optimize=True, device=device, p=2):
    crossentropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    student.train() if optimize else student.eval()
    teacher.train()
    running_loss = 0.0
    good_preds = 0
    for i, data in enumerate(loader):
        # print(f'\r{i+1}/{len(loader)}',end='')
        inputs, labels = data
        if optimize:
            optimizer.zero_grad()
        inputs = inputs.to(device)
        student_outputs, student_activations = NAD_activation(model=student, inputs=inputs, layers_of_interest=layers_of_interest)
        with torch.no_grad():
            teacher_outputs, teacher_activations = NAD_activation(model=teacher, inputs=inputs, layers_of_interest=layers_of_interest)
        
        preds = torch.argmax(student_outputs, dim=1).view(-1).cpu()
        good_preds += (preds==labels).sum().item()
        
        cross_loss = crossentropy(student_outputs, labels.to(device))
        nad_loss = 0.
        
        for name in student_activations:
            student_act = attention_map(student_activations[name], p) # activation has shape N x C x H x W, maps make it N x H x W
            teacher_act = attention_map(teacher_activations[name], p)
            student_act = student_act / student_act.norm(dim=(1,2), keepdim=True) # shape N
            teacher_act = teacher_act / teacher_act.norm(dim=(1,2), keepdim=True)
            nad_loss += mse(student_act, teacher_act)
        loss = cross_loss + beta*nad_loss
        if optimize:
            loss.backward()
            optimizer.step()
        running_loss += loss.cpu().item()
    accuracy = 100 * good_preds/len(loader.dataset)
    return running_loss/(i+1), accuracy


class NeuralDistillation:

    def __init__(self, model_id, train_size, test_size, path_to_save_folder, layers_of_interest, beta, round_number, p=2,
                 attack_size=None, device=device, random_sample=True, batch_size=32):
        self.model_id = model_id

        self.train_size = train_size
        self.test_size = test_size
        self.attack_size = attack_size  # Size of the set used to compute attack rate

        self.path_to_save_folder = path_to_save_folder
        self.round_number = round_number
        self.device = device
        self.random_sample = random_sample
        self.path_to_round = get_path_to_round(self.round_number)
        self.layers_of_interest = layers_of_interest
        self.beta = beta
        self.p = p
        self.batch_size = batch_size

        if not os.path.exists(self.path_to_save_folder):
            os.makedirs(self.path_to_save_folder)
        self.path_to_save = os.path.join(self.path_to_save_folder, f'{model_id}.pt')
        self.path_to_checkpoint = os.path.join(self.path_to_save_folder, 'checkpoints', f'{model_id}.pt')
        self.path_to_output = os.path.join(self.path_to_save_folder, 'outputs', f'{model_id}.json')
        
        if not os.path.exists(os.path.join(self.path_to_save_folder, 'outputs')):
            os.makedirs(os.path.join(self.path_to_save_folder, 'outputs'))
        if not os.path.exists(os.path.join(self.path_to_save_folder, 'checkpoints')):
            os.makedirs(os.path.join(self.path_to_save_folder, 'checkpoints'))

        self.metadata = get_metadata(self.round_number).loc[model_id]
        self.poisoned = self.metadata['poisoned']
        self.n_classes = self.metadata['number_classes']
        self.model_architecture = self.metadata['model_architecture']
        if self.poisoned:
            assert self.attack_size is not None
            self.source_classes = string_list_to_array(self.metadata['triggered_classes'])
            self.target_class = int(self.metadata['trigger_target_class'])

            if self.round_number <=2:
                self.trigger_name = f"{self.metadata['trigger_type']} {self.metadata['trigger_type_option']}"
            else:
                self.trigger_name = f"{self.metadata['trigger_type']} {self.metadata['polygon_side_count']} {self.metadata['instagram_filter_type']}"

        self.device = torch.device(device)
        self.student = load_image_model(model_id=model_id, round_number=round_number)
        self.random_sample = random_sample
        self.get_dataset()


    def get_dataset(self):
        # (self, model_id, round_number=2, original_dataset=False, size=None, size_per_class=None, poisoned=False, n_classes=None, classes=None, random_choice=False, target_class=None,
        #         from_end=False, transform=TRANSFORM_CENTERCROP, path_to_data=None)
        self.train_set = ImageSet(model_id=self.model_id, round_number=self.round_number, size=self.train_size, poisoned=False,
                                n_classes=self.n_classes, random_choice=self.random_sample, from_end=False)

        self.test_set = ImageSet(model_id=self.model_id, round_number=self.round_number, size=self.test_size, poisoned=False,
                                n_classes=self.n_classes, random_choice=self.random_sample, from_end=True)

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
        
    
    def finetune(self, finetune_lr=1e-4, finetune_weight_decay=1e-5, n_epochs=10, early_stopping=True, n_early_stopping=3, verbose=True):
        criterion = nn.CrossEntropyLoss()
        model = copy.deepcopy(self.student)
        optimizer = optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=finetune_weight_decay)
        checkpoint = copy.deepcopy(model.state_dict())

        best_loss = np.inf
        for epoch in range(n_epochs):
            if verbose:
                print(f'[{epoch+1}/{n_epochs}]', end=' ')
            pass_on_loader(model, self.train_loader, optimize=True, optimizer=optimizer, criterion=criterion, device=device)
            test_loss, test_acc = pass_on_loader(model, self.test_loader, optimize=False, optimizer=optimizer, criterion=criterion, device=device)
            if early_stopping:
                if test_loss <= best_loss :
                    early_stop_rounds = 0
                    checkpoint = copy.deepcopy(model.state_dict())
                    best_loss = test_loss
                else:
                    early_stop_rounds += 1
                    if early_stop_rounds >= n_early_stopping:
                        if verbose:
                            print(f'Early stopping...', end=' ')
                        break
            else:
                checkpoint = copy.deepcopy(model.state_dict())
        model.load_state_dict(checkpoint)
        test_loss, test_acc = pass_on_loader(model=model, loader=self.test_loader, optimize=False, optimizer=optimizer, criterion=criterion, device=device)
        print(f'Post FineTune:  Accuracy: {test_acc:.2f}', end=' ')
        if self.poisoned:
            _, attack = pass_on_loader(model=model, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            print(f'Attack Rate: {attack:.2f} ')
        self.teacher = model
    
    def start_distillation(self, max_iter=100, verbose=True, lr=1e-4, weight_decay=1e-5, drop_accuracy=4, random_reinitialization=True, tol_rounds=5, **finetune_kwargs):
        if random_reinitialization:
            print('Reinitializing Weight, No fine tuning')
            self.teacher = copy.deepcopy(self.student)
            self.student.apply(weights_init)
        else:
            print('Fine Tuning, No Random Reinitialization')
            self.finetune(**finetune_kwargs)
        optimizer = optim.Adam(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        it = 0
        attack_per_epoch = []
        accuracy_per_epoch = []
        criterion = nn.CrossEntropyLoss()
        original_student_loss, original_student_accuracy = pass_on_loader(model=self.student, loader=self.test_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
        _, original_teacher_accuracy = pass_on_loader(model=self.teacher, loader=self.test_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
        accuracy_per_epoch.append(original_student_accuracy)
        print(f'Original Student Accuracy: {original_student_accuracy:.2f}, Original Teacher Accuracy: {original_teacher_accuracy:.2f}', end=' - ')
        
        if self.poisoned:
            _, original_student_attack = pass_on_loader(model=self.student, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            _, original_teacher_attack = pass_on_loader(model=self.teacher, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            print(f'Original Student Attack Rate: {original_student_attack:.2f}, Original Teacher Attack Rate: {original_teacher_attack:.2f}', end='')
            attack_per_epoch.append(original_student_attack)
        print()
        
        checkpoint = copy.deepcopy(self.student.state_dict())
        best_loss = original_student_loss
        best_acc = original_student_accuracy
        stop_rounds = 0
        while True:
            if verbose:
                 print(f'{datetime.now():%Y/%m/%d-%H:%M:%S}\t{it:2d}', end=' ')
            train_loss, train_acc =  NAD_loss(self.student, self.teacher, self.train_loader, optimizer, self.beta, self.layers_of_interest, optimize=True, device=self.device, p=2)
            

            print(f'train_nad_loss: {train_loss:1.3e} - train_acc: {train_acc:.2f}', end='\t-\t')

            loss, accuracy = pass_on_loader(model=self.student, loader=self.test_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            accuracy_per_epoch.append(accuracy)
            print(f'test_accuracy: {accuracy:.2f}', end='\t')
            
            if self.poisoned:
                _, attack = pass_on_loader(model=self.student, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
                print(f'attack_rate: {attack:.2f}', end=' ')
                attack_per_epoch.append(attack)
            if random_reinitialization:
                if (loss > best_loss)  or (it>= max_iter):
                    if it >= max_iter:
                        break
                    else:
                        stop_rounds +=1
                        print(f'{tol_rounds - stop_rounds} Remaining... ', end=' ')
                        if stop_rounds >= tol_rounds:
                            print('Early Stopping', end=' ')
                            break
                            
                else:
                    stop_rounds = 0
                    best_loss = loss
                    checkpoint = copy.deepcopy(self.student.state_dict())
                    torch.save(checkpoint, self.path_to_checkpoint)
            else:
                if (accuracy < original_student_accuracy - drop_accuracy) or (it>= max_iter):
                    break
                else:
                    checkpoint = copy.deepcopy(self.student.state_dict())
                    torch.save(checkpoint, self.path_to_checkpoint)
            self.student.load_state_dict(checkpoint)
            it += 1
            print()
        if verbose:
            print('\nSave Student Model')

        self.student.load_state_dict(checkpoint)
        torch.save(self.student, self.path_to_save)
        if not self.poisoned:
            attack_per_epoch = []
        output = {'accuracy': accuracy_per_epoch, 'attack': attack_per_epoch}
        with open(self.path_to_output, 'w') as fp:
            json.dump(output, fp)

        return output

if __name__ == '__main__':
    round_number = 3
    METADATA = get_metadata(round_number=round_number)
    architectures = ['googlenet']
    METADATA = METADATA.loc[METADATA['model_architecture'].apply(lambda x: x in architectures)]
    model_ids = list(METADATA.index)
    poisoned_ids = list(METADATA.loc[METADATA['poisoned']].index)
    model_id = poisoned_ids[8]
    train_size = 3000
    test_size = 1000
    attack_size = 1000
    path_to_folder = '/scratch/jordan.lekeufack/image_models/distillation/round2/'

    # parser = argparse.ArgumentParser(description='Prune them')
    # parser.add_argument('--model_id', type=str, 
    #                     help='model_id', 
    #                     default='id-00000025')

    # parser.add_argument('--round_number', type=int, 
    #                     help='Round', 
    #                     default=2)

    # parser.add_argument('--path_to_folder', type=str, 
    #                      help='File path to the file where output result should be written. ',
    #                     default=path_to_folder)

    # parser.add_argument('--retrain', type=str,
    #                     help='If False, the model is not finepruned if it is already in the folder',
    #                     default='True', choices=['True', 'False'])

    # parser.add_argument('--device', type=int,
    #                     help='Cuda number',
    #                     default=2)   

    # parser.add_argument('--beta', type=float,
    #                     help='Beta',
    #                     default=1.2)
    layers_of_interest = ['Inception']
    beta = 8
    p = 2
    max_iter = 30
    device = torch.device('cuda:2')
    neural_distillation = NeuralDistillation(model_id=model_id, train_size=train_size, test_size=test_size, attack_size=attack_size, path_to_save_folder=path_to_folder,
                                             layers_of_interest=layers_of_interest, beta=beta, p=2, round_number=round_number, device=device)
    
    finetune_lr = 1e-4
    lr = 1e-4
    weight_decay = 1e-4

    output = neural_distillation.start_distillation(max_iter=max_iter, lr=lr, weight_decay=weight_decay, random_reinitialization=True, tol_rounds=10)
    print(output)