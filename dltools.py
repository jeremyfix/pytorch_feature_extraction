#!/usr/bin/env python3
# coding: utf-8
'''
Script used to extract the intermediate features of a deep neural network trained on CIFAR-10. It will download CIFAR-10 and extract the features of a pretrained model.

The model is evaluated on the CPU through it's not difficult to adapt it
to work on the GPU.

To make it work, ensure you clone git Pytorch_CIFAR10 repository :
    git clone git@github.com:huyvnphan/PyTorch_CIFAR10.git

From there, you need to download the pretrained models :
    cd Pytorch_CIFAR10 && python3 cifar10_download.py
'''


import itertools
import inspect
import tqdm
import logging
logging.basicConfig(level=logging.INFO)
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PyTorch_CIFAR10 import cifar10_models
# from PyTorch_CIFAR10 import cifar10_module

model_name = 'mobilenet_v2'
NUM_WORKERS = 6
BATCH_SIZE = 128
DATADIR = '/opt/Datasets/'

modules_idx = {'mobilenet_v2': [5, 35, 67, 139, 212], 'googlenet': [4, 5]}

def load_model(model_name):
    model = getattr(cifar10_models, model_name)()
    model.eval()
    return model

def is_module_base_layer(obj):
    modules = getattr(obj, '__module__', None).split('.')
    return len(modules) >= 3 and \
            modules[0] == 'torch' and \
            modules[1] == 'nn' and \
            'container' not in modules

class Activations(object):
    '''
    Object to extract the intermediate feature representations from 
    a pytorch model
    '''

    def __init__(self, model):
        self.model = model
        self.activations = {}

    def register_hooks(self, module_idx):
        modules = list(self.model.modules())
        # for idx in module_idx:
        self.hook = modules[module_idx].register_forward_hook(self.make_hook("module_{}".format(module_idx)))

    def clean_hooks(self):
        self.hook.remove()

    def print_modules(self):
        for i, m in enumerate(self.model.modules()):
            if is_module_base_layer(m):
                print("Module {} : {}".format(i, m))

    def make_hook(self, module_name):
        def hook(model, inputs, outputs):
            vec_outputs = outputs.reshape(outputs.shape[0], -1)
            if module_name not in self.activations:
                self.activations[module_name] = vec_outputs.detach()
            else:
                self.activations[module_name] = torch.cat((self.activations[module_name], vec_outputs.detach()), dim=0)
        return hook

    def __call__(self, data):
        self.activations = {'input': None}
        if isinstance(data, torch.tensor):
            logging.info("Got a tensor")
            
        for batch in tqdm.tqdm(data):
            inputs, _ = batch
            # Forward propagate, the activations are saved thanks to
            # the hook defined beforehand
            vect_inputs = inputs.reshape(inputs.shape[0], -1)
            if self.activations['input'] is None:
                self.activations['input'] = vect_inputs
            else:
                self.activations['input'] = torch.cat((self.activations['input'],
                                                       vect_inputs), dim=0)
            self.model(inputs)
        return self.activations


def train_val_loaders():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    train_dataset = CIFAR10(download=True,
                            root=DATADIR,
                            train=True,
                            transform=transform)
    trainloader = DataLoader(train_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)
    valid_dataset = CIFAR10(download=True,
                            root=DATADIR,
                            train=False,
                            transform=transform)
    validloader = DataLoader(valid_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)
    return trainloader, validloader

def check_accuracy(model, loader):
    nsamples = 0
    ncorrect = 0
    for batch in tqdm.tqdm(loader):
        inputs, labels = batch
        nsamples += inputs.shape[0]
        outputs = model(inputs)
        predicted_labels = outputs.argmax(axis=1)
        num_correct = (predicted_labels == labels).sum()
        ncorrect += num_correct
    return ncorrect / float(nsamples)

def main(idx=None):
    '''
    Main function for testing the module
    '''
    logging.info("Loading the CIFAR-10 data")
    trainloader, validloader = train_val_loaders()

    logging.info("Loading the pretrained model")
    model = load_model(model_name)

    logging.info("Checking the accuracies")
    # tacc = 100 * check_accuracy(model, trainloader)
    # vacc = 100 * check_accuracy(model, validloader)
    # logging.info("Accuracies : Training({:.2f} %), Test({:.2f} %)".format(tacc,
                                                                          # vacc))
    activations = Activations(model)

    logging.info("Listing the modules on which to possibly anchor a hook")
    activations.print_modules()

    logging.info("Registering the hooks")
    activations.register_hooks(modules_idx[model_name][0])

    activations.clean_hooks()
    if idx is None:
        logging.info("Forward propagate the validation data")
        valid_acts = activations(validloader)
        print(valid_acts['module_5'].shape)
    else:
        # Process a single image
        logging.info("Processing image {}".format(idx))
        img, label = validloader.dataset[0]
        img_activations = activations(img)
        print(label)

if __name__ == '__main__':
    # Process one image for illustration
    main(0)

    # Process the whole dataset
    # main()
