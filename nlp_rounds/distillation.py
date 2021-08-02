import os
import numpy as np
import pandas as pd

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

from utils import *
from datetime import datetime


def get_one_activation(name, activations=None):
    activations = {} if activations is None else activations
    def hook(model, input, output):
        if isinstance(model, nn.GRU):
            activations[name] = output[1]
        elif isinstance(model, nn.LSTM):
            activations[name] = output[1][0]
        else:
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

# No need for attention map since it's not a Conv Network
def attention_map(x, p=2):
    y = x.abs()**p
    return y.mean(dim=1)


def weights_init(m):
    for name, w in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(w.data)
        else:
            w.data.fill_(0)


def NAD_loss(student, teacher, loader, optimizer, beta, layers_of_interest, to_cls, optimize=True, device=device, p=2):
    crossentropy = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    student.train() if optimize else student.eval()
    teacher.train()
    running_nad_loss = 0.0
    running_loss = 0.0
    running_cross_loss = 0.0
    good_preds = 0
    for i, data in enumerate(loader):
        # print(f'\r{i+1}/{len(loader)}',end='')
        inputs, labels = data
        if optimize:
            optimizer.zero_grad()
        inputs = to_cls(inputs)
        inputs = inputs.to(device)
        student_outputs, student_activations = NAD_activation(model=student, inputs=inputs, layers_of_interest=layers_of_interest)
        with torch.no_grad():
            teacher_outputs, teacher_activations = NAD_activation(model=teacher, inputs=inputs, layers_of_interest=layers_of_interest)
        
        preds = torch.argmax(student_outputs, dim=1).view(-1).cpu()
        good_preds += (preds==labels).sum().item()
        
        cross_loss = crossentropy(student_outputs, labels.to(device))
        nad_loss = 0.
        
        for name in student_activations:
            # TODO: Make it more general. For now it's highly specific
            # student_act = attention_map(student_activations[name], p) # activation has shape N x H x W, maps make it N x H x W
            # teacher_act = attention_map(teacher_activations[name], p)
            student_act = student_activations[name]
            teacher_act = teacher_activations[name]
            if name=='rnn': # If it is recurent
                student_act = student_act / student_act.norm(dim=(0,2), keepdim=True) # activation out of LSTM or GRU has shape (bidirectional*) x N x hidden_size
                teacher_act = teacher_act / teacher_act.norm(dim=(0,2), keepdim=True)
            else:
                student_act = student_act / student_act.norm(dim=(1,), keepdim=True) # activation out of LSTM or GRU has shape (bidirectional*) x N x hidden_size
                teacher_act = teacher_act / teacher_act.norm(dim=(1,), keepdim=True)
            nad_loss += mse(student_act, teacher_act)
        loss = cross_loss + beta*nad_loss
        if optimize:
            loss.backward()
            optimizer.step()
        running_loss += loss.cpu().item()
        running_nad_loss += nad_loss.cpu().item()
        running_cross_loss += cross_loss.cpu().item()
    accuracy = 100 * good_preds/len(loader.dataset)
    losses = {'nad': running_nad_loss/(i+1), 'cross': running_cross_loss/(i+1), 'total': running_loss/(i+1)}
    return losses, accuracy


class NeuralDistillation:

    def __init__(self, model_id, train_size, val_size, path_to_save_folder, layers_of_interest, beta, round_number, p=2,
                 attack_size=None, device=device, random_sample=False, batch_size=32):
        self.model_id = model_id

        self.train_size = train_size
        self.val_size = val_size
        # self.test_size = test_size
        self.attack_size = attack_size  # Size of the set used to compute attack rate

        self.path_to_save_folder = path_to_save_folder
        self.round_number = round_number
        self.random_sample = random_sample
        self.path_to_round = get_path_to_round(self.round_number)
        self.layers_of_interest = layers_of_interest
        self.beta = beta
        self.p = p
        self.batch_size = batch_size
        self.device = torch.device(device)

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
        self.model_architecture = self.metadata['model_architecture']

        self.language_model = self.metadata['embedding']
        self.cls_token_is_first = self.metadata['cls_token_is_first']
        self.tokenizer, self.embedding = load_language_model(round_number=self.round_number, language_model=self.language_model, device=self.device)
        self.dataset_name = self.metadata['source_dataset']
        self.poisoned = self.metadata['poisoned']

        
        self.student, self.examples = load_classifier(round_number=self.round_number, model_id=model_id, device=self.device, load_clean_examples=True, load_poisoned_examples=True)

        self.random_sample = random_sample
        self.get_dataset()


    def get_dataset(self):
        self.data_per_label = load_text_examples_per_label(self.dataset_name, n_sample=self.train_size + self.val_size,
                                                           random_sample=self.random_sample, load_from='train')
        train_text_per_label = {label: x[:self.train_size//2] for label,x in self.data_per_label.items()}
        val_text_per_label = {label: x[-self.val_size//2:] for label,x in self.data_per_label.items()}
        
        self.train_set = TextDatasetSimple(train_text_per_label)
        self.val_set = TextDatasetSimple(val_text_per_label)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=True)

        if self.poisoned:
            path_to_supl = get_path_to_supl_data(round_number=round_number, model_id=model_id)
            self.attack_data_per_label = load_text_examples_per_label(path_to_folder=path_to_supl, n_sample=self.attack_size, random_sample=True, load_from='poisoned')
            self.attack_set = TextDatasetSimple(self.attack_data_per_label)
            self.attack_loader = DataLoader(self.attack_set, shuffle=True, batch_size=self.batch_size)

            # attack_text_per_label = {key[1]: value for key, value in self.examples['poisoned'].items()}
            # self.attack_set = TextDatasetSimple(attack_text_per_label)
            # self.attack_loader = DataLoader(self.attack_set, shuffle=True, batch_size=self.batch_size)
        
        # test_text_per_label = load_text_examples_per_label(self.dataset_name, n_sample=self.test_size,
        #                                                    random_sample=self.random_sample, load_from='train')
        # self.test_set = TextDatasetSimple(test_text_per_label)
        # self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=True)


    
    def get_embedding(self, sequence):
        return get_cls_embedding(sequence=sequence, tokenizer=self.tokenizer, embedding=self.embedding, cls_token_is_first=self.cls_token_is_first,
                                 device=self.device).unsqueeze(1)
    
    def finetune(self, finetune_lr=1e-4, finetune_weight_decay=1e-5, n_epochs=10, early_stopping=True, n_early_stopping=3, verbose=True,
                 momentum=0.9, gamma=0.1, step_size=5):
        criterion = nn.CrossEntropyLoss()
        model = copy.deepcopy(self.student)
        optimizer = optim.SGD(model.parameters(), lr=finetune_lr, weight_decay=finetune_weight_decay, momentum=momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        checkpoint = copy.deepcopy(model.state_dict())

        best_loss = np.inf
        val_loss, val_acc = pass_on_loader(model=model, to_cls=self.get_embedding, loader=self.val_loader, optimize=False, optimizer=optimizer, criterion=criterion, device=self.device)
        print('Initial metrics:', end='\t')
        print(f'val_loss: {val_loss:1.3e}, val_accuracy: {val_acc:.2f}', end=' ')
        if self.poisoned:
            _, attack = pass_on_loader(model=model, to_cls=self.get_embedding, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            print(f'Attack Rate: {attack:.2f} ', end='')
        print()

        for epoch in range(n_epochs):
            if verbose:
                print(f'[{epoch+1}/{n_epochs}], lr: {scheduler.get_last_lr()[-1]:1.2e}', end='\t')
            pass_on_loader(model, self.train_loader, to_cls=self.get_embedding, optimize=True, optimizer=optimizer, criterion=criterion, device=self.device)
            val_loss, val_acc = pass_on_loader(model, self.val_loader, to_cls=self.get_embedding, optimize=False, optimizer=optimizer, criterion=criterion, device=self.device)
            print(f'val_loss: {val_loss:1.3e}, val_accuracy: {val_acc:.2f}', end=' ')
            if self.poisoned:
                _, attack = pass_on_loader(model=model, to_cls=self.get_embedding, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
                print(f'Attack Rate: {attack:.2f} ', end='')
            print()
            scheduler.step()
            if early_stopping:
                if val_loss <= best_loss :
                    early_stop_rounds = 0
                    checkpoint = copy.deepcopy(model.state_dict())
                    best_loss = val_loss
                else:
                    early_stop_rounds += 1
                    if early_stop_rounds >= n_early_stopping:
                        if verbose:
                            print(f'Early stopping...', end=' ')
                        break
            else:
                checkpoint = copy.deepcopy(model.state_dict())
        model.load_state_dict(checkpoint)
        val_loss, val_acc = pass_on_loader(model=model, to_cls=self.get_embedding, loader=self.val_loader, optimize=False, optimizer=optimizer, criterion=criterion, device=self.device)
        print(f'Post FineTune:  Accuracy: {val_acc:.2f}', end=' ')
        if self.poisoned:
            _, attack = pass_on_loader(model=model, to_cls=self.get_embedding, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            print(f'Attack Rate: {attack:.2f} ')
        self.teacher = model
    
    def start_distillation(self, max_iter=100, verbose=True, lr=1e-4, weight_decay=1e-5, drop_accuracy=4,
                           random_reinitialization=True, finetune_teacher=False, tol_rounds=5, gamma=.5, step_size=5, **finetune_kwargs):
        if finetune_teacher:
            print(f'Finetuning to create Teacher')
            self.finetune(**finetune_kwargs)
        else:
            print(f'No Finetuning')
            self.teacher = copy.deepcopy(self.student)

        
        if random_reinitialization:
            print('Reinitializing Weights')
            self.student.apply(weights_init)

        
        # if random_reinitialization:
        #     print('Reinitializing Weight, No fine tuning')
        #     self.teacher = copy.deepcopy(self.student)
        #     self.student.apply(weights_init)
        # else:
        #     print('Fine Tuning, No Random Reinitialization')
        #     self.finetune(**finetune_kwargs)

        print('Starting Distillation')
        optimizer = optim.Adam(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.SGD(self.student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        it = 0
        attack_per_epoch = []
        accuracy_per_epoch = []
        cross_val_loss_per_epoch = []
        nad_val_loss_per_epoch = []
        val_loss_per_epoch = []
        criterion = nn.CrossEntropyLoss()
        original_student_loss, original_student_accuracy = NAD_loss(student=self.student, teacher=self.teacher, to_cls=self.get_embedding, loader=self.val_loader, optimizer=optimizer,
                                                                    beta=self.beta, layers_of_interest=self.layers_of_interest, optimize=False, device=self.device, p=self.p)

        # original_student_loss, original_student_accuracy = pass_on_loader(model=self.student, to_cls=self.get_embedding, loader=self.val_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
        _, original_teacher_accuracy = pass_on_loader(model=self.teacher, to_cls=self.get_embedding, loader=self.val_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
        accuracy_per_epoch.append(original_student_accuracy)
        cross_val_loss_per_epoch.append(original_student_loss['cross'])
        nad_val_loss_per_epoch.append(original_student_loss['nad'])
        val_loss_per_epoch.append(original_student_loss['total'])
        print(f'Original Student Accuracy: {original_student_accuracy:.2f}, Original Teacher Accuracy: {original_teacher_accuracy:.2f}', end=' - ')
        
        if self.poisoned:
            _, original_student_attack = pass_on_loader(model=self.student, to_cls=self.get_embedding, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            _, original_teacher_attack = pass_on_loader(model=self.teacher, to_cls=self.get_embedding, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            print(f'Original Student Attack Rate: {original_student_attack:.2f}, Original Teacher Attack Rate: {original_teacher_attack:.2f}', end='')
            attack_per_epoch.append(original_student_attack)
        print()
        
        checkpoint = copy.deepcopy(self.student.state_dict())
        best_loss = original_student_loss['total']
        best_acc = original_student_accuracy
        stop_rounds = 0
        while True:
            if verbose:
                 print(f'{datetime.now():%Y/%m/%d-%H:%M:%S} iter: {it:2d} lr:{scheduler.get_last_lr()[-1]:1.2e}', end=' ')
            train_loss, train_acc = NAD_loss(student=self.student, teacher=self.teacher, to_cls=self.get_embedding, loader=self.train_loader, optimizer=optimizer, beta=self.beta, layers_of_interest=self.layers_of_interest, optimize=True, device=self.device, p=self.p)
            
            scheduler.step()
            print(f'train_total_loss: {train_loss["total"]:1.3e} - train_acc: {train_acc:.2f}', end='\t----\t')

            loss, accuracy = NAD_loss(student=self.student, teacher=self.teacher, to_cls=self.get_embedding, loader=self.val_loader, optimizer=optimizer, beta=self.beta, layers_of_interest=self.layers_of_interest, optimize=False, device=self.device, p=self.p)
            # loss, accuracy = pass_on_loader(model=self.student, to_cls=self.get_embedding, loader=self.val_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
            accuracy_per_epoch.append(accuracy)
            cross_val_loss_per_epoch.append(loss['cross'])
            nad_val_loss_per_epoch.append(loss['nad'])
            total_loss = loss['total']
            val_loss_per_epoch.append(total_loss)
            print(f'val_nad_loss: {loss["nad"]:1.3e} - val_cross_loss: {loss["cross"]:1.3e} - val_total_loss: {total_loss:1.3e} - val_accuracy: {accuracy:.2f}', end='\t')
            
            if self.poisoned:
                _, attack = pass_on_loader(model=self.student, to_cls=self.get_embedding, loader=self.attack_loader, criterion=criterion, optimizer=optimizer, optimize=False, device=self.device)
                print(f'- attack_rate: {attack:.2f}', end=' ')
                attack_per_epoch.append(attack)
            if random_reinitialization:
                if (total_loss > best_loss)  or (it>= max_iter):
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
                    best_loss = total_loss
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
        output = {'accuracy': accuracy_per_epoch, 'attack': attack_per_epoch, 'nad_loss': nad_val_loss_per_epoch, 'cross_loss': cross_val_loss_per_epoch}
        with open(self.path_to_output, 'w') as fp:
            json.dump(output, fp)

        return output

if __name__ == '__main__':
    train_size = 3200
    val_size = 960
    attack_size = 1000
    path_to_folder = '/scratch/jordan.lekeufack/nlp_models/test'

    parser = argparse.ArgumentParser(description='Distillation process')
    parser.add_argument('--model_id', type=str, 
                        help='model_id', 
                        default='id-00000040')

    parser.add_argument('--round_number', type=int, 
                        help='Round', 
                        default=6)

    parser.add_argument('--path_to_folder', type=str, 
                        help='File path to the file where output result should be written.',
                        default=path_to_folder)

    parser.add_argument('--device', type=int,
                        help='Cuda number',
                        default=5)   

    parser.add_argument('--beta', type=float,
                        help='Beta',
                        default=1)

    layers_of_interest = ['LSTM', 'GRU', 'Linear']
    args = parser.parse_args()

    finetune_teacher = True
    random_reinitialization = True
    path_to_folder = args.path_to_folder
    model_id = args.model_id
    device = torch.device(f'cuda:{args.device}')
    augmentation = True
    round_number = args.round_number
    beta = args.beta
    p = 2
    max_iter = 30
    finetune_lr = 1e-4
    lr = 1e-3
    weight_decay = 5e-4
    batch_size = 32
    tol_rounds = 5
    step_size = 10
    neural_distillation = NeuralDistillation(model_id=model_id, train_size=train_size, val_size=val_size, attack_size=attack_size, path_to_save_folder=path_to_folder,
                                             layers_of_interest=layers_of_interest, beta=beta, p=p, round_number=round_number, device=device, batch_size=batch_size)

    print(f'{datetime.now():%Y/%m/%d-%H:%M:%S} - Round {round_number}, model {model_id}, device {device}, architecture {neural_distillation.model_architecture}')
    print(f'Train size: {train_size}, Test size: {val_size}, Attack size: {attack_size}, layers_of_interest: {layers_of_interest}, tol_rounds: {tol_rounds}')
    print(f'beta: {beta}, max_iter: {max_iter}, p: {p}, initial_lr: {lr}, lr_step_size: {step_size}, weight_decay: {weight_decay}')
    
    # neural_distillation.finetune(finetune_lr=finetune_lr)
    output = neural_distillation.start_distillation(max_iter=max_iter, lr=lr, weight_decay=weight_decay, random_reinitialization=random_reinitialization,
                                                    finetune_teacher=finetune_teacher, tol_rounds=tol_rounds, step_size=step_size)
