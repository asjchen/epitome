# Epitome: Constructing the Perfect Handwritten Digits (or Not)
The idea is to train a simple CNN (say LeNet), then take another input image
and adjust it to maximize the softmax output for a given digit. For the digit 
1, I freeze the CNN weights, then perform SGD (or other optimization) on the 
input pixels so the output vector is as close to [0 1 0 0 0 0 0 0 0 0] as 
possible.

This LeNet implementation yields about a 97% accuracy on the MNIST dataset;
however, when starting with random noise, the input optimization simply yields 
more seemingly random noise. When starting with an actual image of a 1 from 
the training set, the optimization hardly modifies the original image.

!(The "Perfect" 1 from Random Noise)[examples/digit_1_from_uniform_random_noise.png]
!(The "Perfect" 1 from Training)[examples/digit_1_from_training.png]


## Training the CNN
I first download the MNIST datasets from [Kaggle](https://www.kaggle.com/c/digit-recognizer/data) and then run:
```
python classification.py --train-file train.csv --test-file test.csv
```
which produces both a pickled model and a CSV file (in the Kaggle submission 
format).

To then optimize images based on the classifier model, I then run either
```
python optimization.py --model-file <model_path> --target-digit 1 --num-samples 10
```
or 
```
python optimization.py --model-file <model_path> --target-digit 1 --num-samples 10 --use-training-file train.csv
```
depending on whether I want to transform uniformly random noise or training 
images of the target digit.
