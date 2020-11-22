# Classification

import argparse
import numpy as np
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

    train_X = train_df.drop(columns=['label']).values
    train_y = train_df['label'].values
    test_X = test_df.values

    # model = LeNet()
    # print(model.model)
    # name = 'relu_batch_64_epochs_30_lr_1e-2'
    # model.train_on_dataset(train_X, train_y, save_path=name)

    # test_predictions = model.predict_on_dataset(test_X)
    # pred_df = pd.DataFrame({'ImageId': np.arange(1, test_X.shape[0] + 1), 'Label': test_predictions})
    # pred_df.to_csv(f'{name}_predictions.csv', index=False)


if __name__ == '__main__':
    main()
