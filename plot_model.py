#! /usr/bin/env python

import argparse
from behavioral_cloning.generators import DrivingLogReader, DataGenerator
import behavioral_cloning.model
import datetime
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('data', type=str)
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # custom_objects is needed to load custom layer(s) used in the model
        model = model_from_json(jfile.read(), custom_objects=behavioral_cloning.model.keras_objects)
    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    times = list()
    steerings = list()
    for r in DrivingLogReader(args.data):
        t = r['timestamp']-datetime.datetime(1970,1,1)
        times.append(t.total_seconds() + 1.0e-6 * t.microseconds)
        steerings.append(r['steering'])
    model_steerings = model.predict_generator(DataGenerator(args.data, min(128, len(times))), len(times))
    ordering = np.argsort(times)
    times = np.array(times)[ordering]
    steerings = np.array(steerings)[ordering]
    model_steerings = np.array(model_steerings)[ordering]
    plt.figure()
    plt.plot(times, steerings, 'b')
    plt.plot(times, model_steerings, 'g')
    plt.title("Steering vs. Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (deg)")
    plt.show()
