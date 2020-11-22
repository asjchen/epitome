# Classification

import argparse
import pandas as pd

from lenet import LeNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', help=(
        'Training dataset, as per the format from Kaggle: '
        'https://www.kaggle.com/c/digit-recognizer/data?select=train.csv'))
    parser.add_argument('--test-file', help=(
        'Testing dataset, as per the format from Kaggle: '
        'https://www.kaggle.com/c/digit-recognizer/data?select=test.csv'))
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    # TODO: split into train and validation
    train_X = train_df.drop(columns=['label']).values
    train_y = train_df['label'].values
    test_X = test_df.values

    model = LeNet()
    print(model.model)
    model.train(train_X, train_y)


if __name__ == '__main__':
    main()
