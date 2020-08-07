#!/usr/bin/env python3

# Standard modules
import logging
logging.basicConfig(level=logging.INFO)
import sys
# External modules
import matplotlib.pyplot as plt
import numpy as np


def main():
    '''
    Loads a npy feature tensor and plot it in a single image
    '''
    if len(sys.argv) != 2:
        logging.error("Usage : {} file.npy".format(sys.argv[0]))
        sys.exit(1)

    # Load the data
    logging.info("Loading {}".format(sys.argv[1]))
    with open(sys.argv[1], 'rb') as f:
        data = np.load(f).ravel()

    # We build a squared image
    size = data.size
    N = int(np.ceil(np.sqrt(data.size)))

    # Fill in the first elements with our data
    img = np.empty((N, N))
    img.reshape(-1)[:size] = data[::]

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    plt.tight_layout()

    filename = '{}.png'.format(sys.argv[1][:-4])
    plt.savefig(filename, bbox_inches='tight')
    logging.info("{} saved".format(filename))


if __name__ == '__main__':
    main()
