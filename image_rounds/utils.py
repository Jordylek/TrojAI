import torch
from torch import optim
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


TRANSFORM_CENTERCROP = transforms.Compose([transforms.CenterCrop(size=224), transforms.ToTensor()])
TRANSFORM_RANDOM_AUGMENT = transforms.Compose([transforms.RandomCrop(size=224, padding=4),
                                               transforms.RandomHorizontalFlip(p=.5)])

PATH_TO_TROJAI= '/scratch/data/TrojAI/'
device = torch.device('cuda:2')

def get_path_to_round(round_number):
    return os.path.join(PATH_TO_TROJAI, f'round{round_number}')


def get_metadata(round_number):
    path_to_round = get_path_to_round(round_number=round_number)
    return pd.read_csv(os.path.join(path_to_round, 'METADATA.csv'), index_col=0)


def load_image_model(model_id, round_number=2, device=device):
    path_to_round = get_path_to_round(round_number=round_number)
    model = torch.load(os.path.join(path_to_round, 'models', model_id, 'model.pt'), map_location=device)
    return model


def string_list_to_array(s, dtype=np.uint8):
    return np.array(s.strip('][').split(' '), dtype=dtype)


def pass_on_loader(model, loader, criterion, optimizer=None, optimize=False, device=device, augment_transform=None):
    model.train() if optimize else model.eval()
    running_loss = 0.0
    good_preds = 0
    size = 0
    for i, data in enumerate(loader):
        inputs, labels = data
        if augment_transform is not None:
            inputs = augment_transform(inputs)
        if optimize:
            optimizer.zero_grad()
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


class ImageSet(Dataset):

    def __init__(self, model_id, round_number=2, original_dataset=False, size=None, size_per_class=None, poisoned=False, n_classes=None, classes=None, random_choice=False, target_class=None,
                 transform=TRANSFORM_CENTERCROP, path_to_data=None, skip_first_n=0):
        self.model_id = model_id
        self.poisoned = poisoned
        self.round= round_number
        self.original_dataset = original_dataset

        path_to_round = get_path_to_round(round_number=self.round)

        if path_to_data is None:
            if original_dataset:
                folder = 'poisoned_example_data' if self.poisoned else 'clean_example_data'
                self.path_to_data = os.path.join(path_to_round, 'models', self.model_id, folder)
            else:
                folder = 'poisoned_examples' if self.poisoned else 'clean_examples'
                self.path_to_data = os.path.join(path_to_round, 'supl_data', self.model_id, folder)
        else:
            self.path_to_data = path_to_data

        assert os.path.isdir(self.path_to_data)

        if classes is None:
            assert n_classes is not None
            self.n_classes = n_classes 
            self.classes = np.arange(self.n_classes)
        else:
            self.n_classes = len(classes)
            self.classes = classes

        self.size_per_class = size_per_class if size is None else int(np.ceil(size/self.n_classes))
        self.skip_per_class = int(np.ceil(skip_first_n/self.n_classes))
        self.size = self.size_per_class * self.n_classes
        self.random_choice = random_choice

        if self.poisoned:
            assert target_class is not None
            self.target_class = target_class
        self.transform = transform
        self.filenames = []
        self.load_images()
       

    def load_images(self):
        self.x = []
        all_imgs_filepath = sorted(os.listdir(self.path_to_data))
        self.y = [self.target_class] * self.size if self.poisoned else []
        for label in self.classes:
            imgs = [s for s in all_imgs_filepath if s.startswith(f'class_{label}')]

            assert len(imgs)>0

            if self.random_choice:
                imgs = np.random.choice(imgs, size=self.size_per_class, replace=False)
            else:
                imgs = imgs[self.skip_per_class:self.skip_per_class + self.size_per_class] 
            torch_imgs = []

            for filename in imgs:
                img = Image.open(os.path.join(self.path_to_data, filename))
                torch_imgs.append(self.transform(img))
            self.x.extend(torch_imgs)
            self.filenames.extend(imgs)
            if not self.poisoned:
                self.y.extend([label] * self.size_per_class)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size
