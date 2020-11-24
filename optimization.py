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
    parser.add_argument('--target-digit', required=True, type=int, 
        choices=range(10),
        help='The digit that we want to find the \"epitome\" for')
    parser.add_argument('--num-samples', type=int, default=10,
        help=('The number of epitome samples to generate'))
    parser.add_argument('--use-training-file', default=None, help=(
        'Start constructions with entries from the given training set, '
        'rather than from uniform noise. The format of the file should match: '
        'https://www.kaggle.com/c/digit-recognizer/data?select=train.csv'))
    args = parser.parse_args()

    model = LeNet()
    model.load_saved_state(args.model_file)

    starter_X = None
    if args.use_training_file:
        train_df = pd.read_csv(args.use_training_file)
        starter_df = train_df[train_df['label'] == args.target_digit]
        starter_X = starter_df.drop(columns=['label']).values[: args.num_samples]

    opt_inputs = model.optimize_for_digit(
        args.target_digit, 
        num_samples=args.num_samples,
        starter_X=starter_X,
    )
    for i in range(opt_inputs.shape[0]):
        img = Image.fromarray(opt_inputs[i], 'L')
        img.save(f'digit_{args.target_digit}_{i}.png')


if __name__ == '__main__':
    main()
