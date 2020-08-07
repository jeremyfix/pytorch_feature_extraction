# Pytorch intermediate layer feature extraction

This script allows to extract the features of the intermediate layer of pytorch networks. 

The selection of the layers to export is by providing modules idx of the module list of a pytorch model. We make use of
the PyTorch_CIFAR10 pretrained models. For the `dltools.py` script to work, you need to clone this repository
recursively :

    git clone --recurse-submodules git@github.com:jeremyfix/pytorch_feature_extraction.git

and you also need to download the pretrained networks from
[PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10).

Check the documentation with

    python3 dltools.py --help

Then you can process a single image with 

    python3 dltools.py --image path/to/an/image

Or the whole CIFAR-10 validation dataset

    python3 dltools.py 

By default, it is going to process `mobilenet_v2` (the one of PyTorch_CIFAR10) with a batch_size of 128. The features of
the modules 5, 35, 67, 139 and 212 are saved in numpy `npy` files.

If your CPU/GPU has not enough memory, you should also consider passing in the `--sequential` flag which is going to
perform one forward pass per intermediate layer preventing to store all the intermediate layers in memory.

# Example usage 

For example, to save the features of the modules 5, 35, 67, 139 of a `mobilenet_v2`, processing the image `coq.png`

    python3 dltools.py --model_name mobilenet_v2 --modules_idx 5 35 67 139 --image coq.png 

For example, to save the features of the last linear layer of a `googlenet`, processing the while CIFAR-10 validation set :

    python3 dltools.py --model_name googlenet --modules_idx 215  
