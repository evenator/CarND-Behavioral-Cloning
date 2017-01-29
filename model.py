#! /usr/bin/env python

import argparse
from behavioral_cloning.model import Dave2, PreprocessImage
from behavioral_cloning.generators import DataGenerator, MultiCameraGenerator, StochasticGeneratorPool, JitteredMultiCameraGenerator
import csv
import functools
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import random

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
    parser.add_argument('--data-dir', '-d', type=str, help='Data directory to use', nargs='+')
    parser.add_argument('--verbose', action='count', help='verbosity level (up to 2)')
    parser.add_argument('--validation-dir', '-v', type=str, help='Data directory to use for validation', required=True)
    parser.add_argument('--metrics', '-m', type=str, nargs='*', help='Metrics for Keras to evaluate')
    args = parser.parse_args()
    print(vars(args))
            
    # Create generator for training_data
    training_generator = StochasticGeneratorPool([JitteredMultiCameraGenerator(d, args.batch_size) for d in args.data_dir])
    # Create generator for validation data
    validation_generator = DataGenerator(args.validation_dir, args.batch_size)
    # Create model
    model = Dave2(dropout=args.dropout)
    model.compile(args.optimizer, args.loss, args.metrics)
    # Train model
    history = model.fit_generator(training_generator, len(training_generator), args.epochs, validation_data=validation_generator, nb_val_samples=len(validation_generator))
    #Save model
    save_model(model)
    #Print history results
    print(history.history)
    plt.plot(range(args.epochs), history.history['loss'], 'g')
    plt.plot(range(args.epochs), history.history['val_loss'], 'b')

if __name__ == "__main__":
    main()
