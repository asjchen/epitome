# Classification

import argparse
import pandas as pd

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

    print(train_df.head())

if __name__ == '__main__':
    main()
