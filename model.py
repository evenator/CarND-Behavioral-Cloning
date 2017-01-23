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

# A generic generator
# TODO: Split into a data generators file
# TODO: Use side images
# TODO: Add mirroring
# TODO: Move preprocessing into this generator
# TODO: YUV space
def data_generator(data, batch_size):
    inputs = list()
    targets = list()
    for sample in itertools.cycle(data):
        path = sample[0]
        try:
            image = scipy.ndimage.imread(path, mode='RGB')
        except IOError:
            print("Couldn't open {}".format(path))
            continue
        inputs.append(PreprocessImage(image))
        targets.append(sample[1])
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
                img_path = r[0]
                steering = r[3]
                if not os.path.isabs(img_path):
                    img_path = os.path.normpath(os.path.join(directory, img_path))
                samples.append((img_path, steering))
    # Shuffle data and select training and validation set
    random.shuffle(samples)
    training_size = int(0.8 * len(samples))
    validation_size = len(samples) - training_size
    training_data = samples[:training_size]
    validation_data = samples[training_size:]
            
    # Create generator for training_data
    training_generator = data_generator(training_data, args.batch_size)
    # Create generator for validation data
    validation_generator = data_generator(validation_data, args.batch_size)
    # Create model
    model = Dave2()
    model.compile(args.optimizer, args.loss, args.metrics)
    # Train model
    history = model.fit_generator(training_generator, training_size, args.epochs, validation_data=validation_generator, nb_val_samples=validation_size)
    #Save model
    save_model(model)
    #Print history results
    print(history.history)

if __name__ == "__main__":
    main()
