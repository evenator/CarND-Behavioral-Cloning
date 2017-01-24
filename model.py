#! /usr/bin/env python

import argparse
from behavioral_cloning.model import Dave2, PreprocessImage
import csv
import functools
import itertools
import numpy as np
import os.path
import random
import scipy.ndimage

def normalize_path(directory, path):
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(directory.strip(), path.strip()))
    return path

def add_images(dataset, directory, row, angle_offset=0.25):
    '''
    Add all three images from the row to the dataset
    The dataset is a list of (image path, steering angle) pairs, where each
    one is a training example.
    The row is a row in a simulator recording CSV file.
    The angle_offset is the offset to add to the steering angle for the 
    left and right images.
    '''
    center, left, right, steering, _, _, _ = row
    try:
        steering = float(steering)
    except ValueError:
        # This must be the header row, or there's no steering for some reason
        return
    # Center
    center = normalize_path(directory, center)
    if os.path.isfile(center):
       dataset.append((center, steering))
    if steering == 0:    
        # Left
        left = normalize_path(directory, left)
        if os.path.isfile(left):
            dataset.append((left, steering + angle_offset))
        # Right
        right = normalize_path(directory, right)
        if os.path.isfile(right):
            dataset.append((right, steering - angle_offset))
    

# A generic generator
# TODO: Split into a data generators file
def data_generator(data, batch_size):
    inputs = list()
    targets = list()
    random.seed()  # Reseed the random generator
    # Shuffle the data just in case
    random.shuffle(data)
    for sample in itertools.cycle(data):
        path, steering = sample
        try:
            image = scipy.ndimage.imread(path, mode='RGB')
        except IOError:
            print("Couldn't open {}".format(path))
            continue
        # Randomly decide whether to flip the image to simulate
        # turning the other way
        if random.randrange(1):
            image = numpy.fliplr(image)
            steering = -steering
        inputs.append(PreprocessImage(image))
        targets.append(steering)
        if len(inputs) == batch_size:
            yield((np.stack(inputs), targets))
            inputs = list()
            targets = list()

def save_model(model):
    with open('model.json', 'w') as j:
        j.write(model.to_json())
    # Weights only
    model.save_weights('model.h5')
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', type=int, help='Number of samples in each batch', required=True)
    parser.add_argument('--epochs', '-e', type=int, help='number of training epochs', required=True)
    parser.add_argument('--optimizer', '-o', type=str, help='One of the Keras optimizers', required=True)
    parser.add_argument('--dropout', '-x', type=float, help='Dropout rate [0,1.0]', default=0.0)
    parser.add_argument('--loss', '-l', type=str, help='One of the Keras loss functions', required=True)
    parser.add_argument('--data-dir', '-d', type=str, help='Data directories to use', nargs='+')
    parser.add_argument('--verbose', '-v', action='count', help='verbosity level (up to 2)')
    parser.add_argument('--metrics', '-m', type=str, nargs='*', help='Metrics for Keras to evaluate')
    args = parser.parse_args()
    print(vars(args))

    samples = list()
    for directory in args.data_dir:
        with open(directory + '/driving_log.csv', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for r in reader:
                add_images(samples, directory, r)
    # Shuffle data and select training and validation set
    random.shuffle(samples)
    training_size = int(0.8 * len(samples))
    validation_size = len(samples) - training_size
    training_data = samples[:training_size]
    validation_data = samples[training_size:]
    print("After augmentation, there are {} training examples and {} validation examples".format(training_size, validation_size))
            
    # Create generator for training_data
    training_generator = data_generator(training_data, args.batch_size)
    # Create generator for validation data
    validation_generator = data_generator(validation_data, args.batch_size)
    # Create model
    model = Dave2(dropout=args.dropout)
    model.compile(args.optimizer, args.loss, args.metrics)
    # Train model
    history = model.fit_generator(training_generator, training_size, args.epochs, validation_data=validation_generator, nb_val_samples=validation_size)
    #Save model
    save_model(model)
    #Print history results
    print(history.history)

if __name__ == "__main__":
    main()
