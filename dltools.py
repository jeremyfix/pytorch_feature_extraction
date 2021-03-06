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


# Standard modules
import argparse
import itertools
import logging
logging.basicConfig(level=logging.INFO)
import sys
# External modules
from PIL import Image
import numpy as np
import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PyTorch_CIFAR10 import cifar10_models
# from PyTorch_CIFAR10 import cifar10_module

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
CIFAR10_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

def load_model(model_name):
    model = getattr(cifar10_models, model_name)(pretrained=True)
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
        self.labels = None

    def register_hooks(self, module_idx):
        modules = list(self.model.modules())
        for idx in module_idx:
            self.hook = modules[idx].register_forward_hook(self.make_hook("module_{}".format(idx)))

    def clean_hooks(self):
        self.hook.remove()

    def print_modules(self):
        for i, m in enumerate(self.model.modules()):
            if is_module_base_layer(m):
                print("Module {} ({}): {}".format(i, type(m), m))

    def make_hook(self, module_name):
        def hook(model, inputs, outputs):
            # vec_outputs = outputs.reshape(outputs.shape[0], -1)
            if module_name not in self.activations:
                self.activations[module_name] = outputs.detach()
            else:
                self.activations[module_name] = torch.cat((self.activations[module_name], outputs.detach()), dim=0)
        return hook

    def __call__(self, data, size):
        self.activations = {'input': None}
        self.labels = None
        if isinstance(data, torch.Tensor):
            # We consider this is a single image
            vect_inputs = data.reshape(data.shape[0], -1)
            self.activations['input'] = vect_inputs
            self.model(data)
        elif isinstance(data, torch.utils.data.dataloader.DataLoader):
            # We consider this is a DataLoader
            n_samples = 0
            for batch in tqdm.tqdm(data):
                inputs, labels = batch
                n_samples += inputs.shape[0]
                # Forward propagate, the activations are saved thanks to
                # the hook defined beforehand
                # vect_inputs = inputs.reshape(inputs.shape[0], -1)
                if self.activations['input'] is None:
                    self.activations['input'] = inputs.detach()
                else:
                    self.activations['input'] = torch.cat((self.activations['input'],
                                                           inputs), dim=0)
                if self.labels is None:
                    self.labels = labels.detach()
                else:
                    self.labels = torch.cat((self.labels, labels.detach()), dim=0)
                self.model(inputs)
                if size is not None and n_samples > size:
                    break
        else:
            raise Exception("What should I do with a {}".format(type(data)))
        return self.activations, self.labels


def train_val_loaders(batch_size, num_workers, dataset_dir):
    train_dataset = CIFAR10(download=True,
                            root=dataset_dir,
                            train=True,
                            transform=CIFAR10_transform)
    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True)
    valid_dataset = CIFAR10(download=True,
                            root=dataset_dir,
                            train=False,
                            transform=CIFAR10_transform)
    validloader = DataLoader(valid_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
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

def main(args):
    '''
    Main function for testing the module
    '''

    logging.info("Loading the pretrained model")
    model = load_model(args.model_name)

    if args.check_accuracy or args.image is None:
        logging.info("Loading the CIFAR-10 data")
        trainloader, validloader = train_val_loaders(args.batch_size,
                                                     args.num_workers,
                                                     args.dataset_dir)

    if args.check_accuracy:
        logging.info("Checking the accuracies")
        tacc = 100 * check_accuracy(model, trainloader)
        vacc = 100 * check_accuracy(model, validloader)
        logging.info("Accuracies : Training({:.2f} %), Test({:.2f} %)".format(tacc,
                                                                          vacc))
        sys.exit(0)

    activations = Activations(model)

    logging.info("Listing the modules on which to possibly anchor a hook")
    activations.print_modules()

    hooks = args.modules_idx
    save_input = True

    while len(hooks) != 0:

        if args.sequential:
            logging.info("Registering the hook at module {}".format(hooks[0]))
            activations.register_hooks([hooks.pop(0)])
        else:
            logging.info("Registering the hook at modules {}".format(",".join(map(str, hooks))))
            activations.register_hooks(hooks)
            hooks.clear()

        if args.image is None:

            logging.info("Forward propagate the validation data")
            valid_acts, valid_labels = activations(validloader, args.size)

        else:
            # Process a single image
            logging.info("Processing image {}".format(args.image))
            input_image = Image.open(args.image)
            input_tensor = CIFAR10_transform(input_image)
            input_batch = input_tensor.unsqueeze(0)  # adds Batch dim
            valid_acts, valid_labels = activations(input_batch, args.size)
        # Valid_acts is a dictionnary with the inputs and the features
        # of one (--sequential) or several intermediate layers
        # The values of the dictionnary are torch.Tensor with one row 
        # if --sequential or 10000 rows (the number of validation data
        # in the CIFAR 10 dataset)
        datafile_prefix = 'image_' if args.image is not None else 'cifar10_'

        datafile_prefix += args.model_name + '_'
        # Save the labels once for all
        if valid_labels is not None:
            with open("{}labels.npy".format(datafile_prefix), 'wb') as f:
                np.save(f, valid_labels.numpy())
        # And save all the activations now
        for k, v in valid_acts.items():
            # We should save the input only once
            if v == 'input':
                if not save_input:
                    continue
                save_input = False
            #
            logging.info("Saving {} of size {}".format(k, v.shape))
            with open("{}{}.npy".format(datafile_prefix, k), 'wb') as f:
                np.save(f, v.numpy())
        activations.clean_hooks()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''
                                     Forward propagate an image or the whole
                                     CIFAR 10 datasets and saves some of the
                                     intermediate features
                                     ''')
    parser.add_argument('--size',
                        type=int,
                        help='The maximum number of elements to save',
                        default=None
                       )
    parser.add_argument('--batch_size',
                        type=int,
                        help='The batch size',
                        default=128
                       )
    parser.add_argument('--num_workers',
                        type=int,
                        help='The number of workers for the dataloaders',
                        default=6
                       )
    parser.add_argument('--dataset_dir',
                        type=str,
                        help='The path to store the CIFAR10 dataset',
                        default='/opt/Datasets'
                       )

    parser.add_argument('--check_accuracy',
                        action="store_true",
                        help="Whether or not to check the accuracy on the valid set")
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='''Which model to use. Must be one of the
                        PyTorch_CIFAR10 repository (see 
                        PyTorch_CIFAR10/cifar10_module.py:get_classifier 
                        to see all the available models)''')
    parser.add_argument('--modules_idx', nargs='+',
                        type=int,
                        required=True,
                        help='''Which modules idx to save. These are the 
                        indices shown in the console''')

    parser.add_argument('--sequential',
                        action="store_true",
                        help='''Whether or not to process sequentially
                        all the layers. For low memory
                        systems, this option should be on''')
    parser.add_argument('--image',
                        type=str,
                        default=None,
                        help='''The path to an image if a single image should be
                        be processed''')

    args = parser.parse_args()
   
    main(args)
