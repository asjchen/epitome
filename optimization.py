# Optimization

import argparse
import numpy as np
import pandas as pd
from PIL import Image

from lenet import LeNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', required=True,
        help='File path to a trained PyTorch model')
    parser.add_argument('--target-digit', required=True, type=int, choices=range(10),
        help='The digit that we want to find the \"epitome\" for')
    parser.add_argument('--num-samples', type=int, default=10,
        help='The number of epitome samples to generate')
    args = parser.parse_args()

    model = LeNet()
    model.load_saved_state(args.model_file)
    opt_inputs = model.optimize_for_digit(args.target_digit, num_samples=args.num_samples)
    for i in range(opt_inputs.shape[0]):
        img = Image.fromarray(opt_inputs[i], 'L')
        img.save(f'digit_{args.target_digit}_{i}.png')

    pos_x = model.preprocess_X(opt_inputs.reshape(-1, 784))
    print(model.forward(pos_x))

if __name__ == '__main__':
    main()
