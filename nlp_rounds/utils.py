import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import json
import random

# PATH = '/scratch/data/round6-train-dataset'
path_to_text = '/scratch/data/sentiment-classification/'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu') # torch.device('cpu')
# METADATA = pd.read_csv(f'{PATH}/METADATA.csv', index_col=0)

PATH_TO_TROJAI= '/scratch/data/'
device = torch.device('cuda:2')

def get_path_to_round(round_number):
    return os.path.join(PATH_TO_TROJAI, f'round{round_number}-train-dataset')


def get_path_to_supl_data(round_number, model_id=None):
    model_id = '' if model_id is None else model_id
    return f'/scratch/data/TrojAI/round{round_number}/supl_data/{model_id}'

def get_metadata(round_number):
    path_to_round = get_path_to_round(round_number=round_number)
    return pd.read_csv(os.path.join(path_to_round, 'METADATA.csv'), index_col=0)


def pass_on_loader(model, loader, criterion, to_cls, optimizer=None, optimize=False, device=device):
    model.train() if optimize else model.eval()
    running_loss = 0.0
    good_preds = 0
    size = 0
    for i, data in enumerate(loader):
        inputs, labels = data
        if optimize:
            optimizer.zero_grad()
        inputs = to_cls(inputs)
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).view(-1).cpu()
        good_preds += (preds==labels).sum().item()
        size += preds.shape[0]
        loss = criterion(outputs, labels.to(device))
        if optimize:
            loss.backward()
            optimizer.step()
        running_loss += loss.cpu().item()
    accuracy = 100 * good_preds/size
    return running_loss/(i+1), accuracy

def get_cls_embedding(sequence, tokenizer, embedding, cls_token_is_first, device=device):
    with torch.no_grad():
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
        result = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length-2)
        tokens = result['input_ids'].to(device)
        attention_mask = result['attention_mask'].to(device)
        sentence_embedding = embedding(tokens, attention_mask=attention_mask)[0]
        if cls_token_is_first:
            # for BERT-like models use the first token as the text summary
            embedding_vector = sentence_embedding[:, 0, :]
        else:
            # for GPT-2 use the last token as the text summary
            # embedding_vector = embedding_vector[:, -1, :]  # if all sequences are the same length
            embedding_vector = sentence_embedding.cpu().detach().numpy()
            attn_mask = attention_mask.detach().cpu().detach().numpy()
            emb_list = list()
            for i in range(attn_mask.shape[0]):
                idx = int(np.argwhere(attn_mask[i, :] == 1)[-1])
                emb_list.append(embedding_vector[i, idx, :])
            embedding_vector = np.stack(emb_list, axis=0)
            embedding_vector = torch.from_numpy(embedding_vector).to(device)
    return embedding_vector

def evaluate_accuracy(classifier, tokenizer, embedding, cls_token_is_first, examples, device=device):
    accuracy = {}
    for label in examples:
        texts = examples[label]
        with torch.no_grad():
            embedded_cls = get_cls_embedding(texts, tokenizer, embedding, cls_token_is_first).unsqueeze(1)
            output = classifier(embedded_cls)
        pred_labels = output.argmax(dim=1).cpu().numpy()
        accuracy[label] = (pred_labels == label).mean()
    return accuracy

def evaluate_attack(classifier, tokenizer, embedding, cls_token_is_first, examples, device=device):
    accuracy = {}
    for (source_label, target_label) in examples:
        texts = examples[(source_label, target_label)]
        with torch.no_grad():
            embedded_cls = get_cls_embedding(texts, tokenizer, embedding, cls_token_is_first).unsqueeze(1)
            output = classifier(embedded_cls)
        pred_labels = output.argmax(dim=1).cpu().numpy()
        accuracy[(source_label, target_label) ] = (pred_labels == target_label).mean()
    return accuracy


def cosine_torch(x1, x2=None, eps=1e-8, dim=1):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=dim, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=dim, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def load_classifier(model_id, round_number, load_clean_examples=False, load_poisoned_examples=False, device=device):
    path = get_path_to_round(round_number=round_number)
    model_filepath = f'{path}/models/{model_id}/model.pt'
    classification_model = torch.load(model_filepath, map_location=torch.device(device))
    clean_examples = {}
    poisoned_examples = {}
    if load_clean_examples:
        examples_dirpath = f'{path}/models/{model_id}/clean_example_data'
        # Returns a dictionnary {class_id : ['example 1', 'example 2], ...}
        class_idx = -1
        while True:
            class_idx += 1
            fn = 'class_{}_example_{}.txt'.format(class_idx, 1)
            if not os.path.exists(os.path.join(examples_dirpath, fn)):
                break
            clean_examples[class_idx] = []
            example_idx = 0
            while True:
                example_idx += 1
                fn = 'class_{}_example_{}.txt'.format(class_idx, example_idx)
                if not os.path.exists(os.path.join(examples_dirpath, fn)):
                    break
                # load the example
                with open(os.path.join(examples_dirpath, fn), 'r') as fh:
                    text = fh.read()
                clean_examples[class_idx].append(text)

    if load_poisoned_examples:
        examples_dirpath = f'{path}/models/{model_id}/poisoned_example_data'
        # Returns a dictionnary {class_id : ['example 1', 'example 2], ...}
        if os.path.exists(examples_dirpath):
            source_class_idx = -1
            while True:
                source_class_idx += 1
                target_class_idx = -1
                while True:
                    target_class_idx += 1
                    fn = 'source_class_{}_target_class_{}_example_{}.txt'.format(source_class_idx, target_class_idx, 1)
                    if not os.path.exists(os.path.join(examples_dirpath, fn)):
                        if target_class_idx>10: # arbitrary for now
                            break
                        continue
                    poisoned_examples[(source_class_idx, target_class_idx)] = []
                    example_idx = 0
                    while True:
                        example_idx += 1
                        fn = 'source_class_{}_target_class_{}_example_{}.txt'.format(source_class_idx, target_class_idx, example_idx)
                        if not os.path.exists(os.path.join(examples_dirpath, fn)):
                            break
                        # load the example
                        with open(os.path.join(examples_dirpath, fn), 'r') as fh:
                            text = fh.read()
                        poisoned_examples[(source_class_idx, target_class_idx)].append(text)
                    
                if source_class_idx>10: # arbitrary for now
                    break
    if load_clean_examples or load_poisoned_examples:
        return classification_model, {'clean': clean_examples, 'poisoned': poisoned_examples}
    return classification_model


def load_language_model(round_number, language_model='DistilBERT', device=device):
    path = get_path_to_round(round_number=round_number)
    name = [s for s in os.listdir(f'{path}/tokenizers/') if s.startswith(language_model)][0]
    tokenizer_filepath = f'{path}/tokenizers/{name}'
    embedding_filepath = f'{path}/embeddings/{name}'

    tokenizer = torch.load(tokenizer_filepath)
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # load the specified embedding
    embedding = torch.load(embedding_filepath, map_location=torch.device(device))

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    return tokenizer, embedding


def load_text_examples(dataset_name, n_sample=None, random_sample=False, balanced=True):
    file = open(f'{path_to_text}/{dataset_name}/train.json')

    data = json.load(file)
    file.close()

    n_sample = len(data) if n_sample is None else n_sample
    train_data = [(val['data'], val['label']) for val in data.values() if val['data'] is not None and len(val['data']) > 10] # At least 10 characters
    if balanced:
        labels = set([x[1] for x in train_data])
        data_per_label = {label : [x[0] for x in train_data if x[1]==label] for label in labels}
        n_per_label = n_sample // len(labels)
        
        train_text = []
        train_label = []
        for label in data_per_label:
            if random_sample:
                random.shuffle(data_per_label[label])
            train_text = train_text + data_per_label[label][:n_per_label]
            train_label = train_label + [label] * n_per_label
        return train_text, train_label
    else:

        if random_sample:
            random.shuffle(train_data)
        
        train_text = [x[0] for x in train_data[:n_sample]]
        train_label = [x[1] for x in train_data[:n_sample]]
        return train_text, train_label


class TextDatasetSimple(Dataset):
    def __init__(self, data_per_label):
        self.x, self.y = [], []
        for label in data_per_label:
            self.x = self.x + data_per_label[label]
            self.y = self.y + [label] *len(data_per_label[label])
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)


def load_text_examples_per_label(dataset_name=None, n_sample=None, random_sample=False, load_from='train', path_to_folder=None):
    """
    load_from: 'train' or 'test'
    """
    path_to_folder = f'{path_to_text}/{dataset_name}' if path_to_folder is None else path_to_folder
    file = open(f'{path_to_folder}/{load_from}.json')

    data = json.load(file)
    file.close()

    n_sample = len(data) if n_sample is None else n_sample
    train_data = [(val['data'], val['label']) for val in data.values() if val['data'] is not None and len(val['data']) > 10] # At least 10 characters
    if random_sample:
        random.shuffle(train_data)
    labels = set([x[1] for x in train_data])
    data_per_label = {label : [x[0] for x in train_data if x[1]==label] for label in labels}

    n_per_label = n_sample // len(labels)

    return {label: x[:n_per_label] for label, x in data_per_label.items()}

# def load_text_examples_per_label(dataset_name, size_1, size_2=None, random_sample=True, load_from='train'):
#     """
#     load_from: 'train' or 'test'
#     """
#     file = open(f'{path_to_text}/{dataset_name}/{load_from}.json')

#     all_data = json.load(file)
#     file.close()
#     assert size_1 is not None
#     size_2 = size_2 if size_2 is not None else 0
#     n_sample = size_1 + size_2 
#     assert n_sample <= len(all_data)
#     data = [(val['data'], val['label']) for val in all_data.values() if val['data'] is not None and len(val['data']) > 10] # At least 10 characters
#     if random_sample:
#         random.shuffle(data)
#     labels = set([x[1] for x in data])
#     data_per_label = {label : [x[0] for x in data if x[1]==label] for label in labels}
#     if size_2:
#         n_per_label = [size_1 // len(labels), size_2 // len(labels)]
#         set_1 = {label: x[:n_per_label[0]] for label, x in data_per_label.items()}
#         set_2 = {label: x[n_per_label[0]:n_per_label[1]+n_per_label[0]] for label, x in data_per_label.items()}
#         return set_1, set_2
#     else:
#         n_per_label = n_sample // len(labels)
#         return {label: x[:n_per_label] for label, x in data_per_label.items()}
       

