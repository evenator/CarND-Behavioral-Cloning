#! /usr/bin/env python

import matplotlib.pyplot as plt
import csv
import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline
import sys

def smooth(x_series, kernel = np.array([1.0, 1.0, 1.0])):
    normalized_kernel = kernel/kernel.sum()
    result = np.convolve(x_series, normalized_kernel, 'valid')
    result = np.insert(result, 0, x_series[0])
    result = np.append(result, x_series[-1])
    return result

with open(sys.argv[1] + '/driving_log.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',')
    start_time = None
    times = list()
    steerings = list()
    speeds = list()
    dists = list()
    for r in reader:
        center, left, right, steering, throttle, brake, speed = r
        steering = float(steering)
        throttle = float(throttle)
        brake = float(brake)
        speed = float(speed)
        dateparts = center[:-4].split('/')[-1].split('_')[1:]
        dateparts = [int(x) for x in dateparts]
        t = datetime.datetime(dateparts[0], dateparts[1], dateparts[2], dateparts[3], dateparts[4], dateparts[5], 1000*dateparts[6])
        if start_time is None:
            start_time = t
            last_dist = 0.0
            last_time = 0.0
            last_speed = speed
        else:
            last_dist = dists[-1]
            last_time = times[-1]
            last_speed = speeds[-1]
        dt = t - start_time
        time_seconds = dt.seconds + dt.microseconds/1.0E6
        if len(steerings):
            d_steering = steering - steerings[-1]
            if abs(d_steering) > 0.5:
                print("Rapid change in steering ({}) at {}".format(d_steering, center))
        times.append(time_seconds)
        steerings.append(steering)
        speeds.append(speed)
        dists.append(last_dist + (speed + last_speed)/ 2.0 * (time_seconds - last_time))
plt.figure()
plt.plot(times, steerings, 'b')
spline = UnivariateSpline(times, steerings, s=25.0)
plt.plot(times, spline(times), 'g')
plt.title("Steering vs. Time")
plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (deg)")
'''
plt.figure()
plt.plot(dists, steerings)
plt.title("Steering vs. Distance")
plt.xlabel("Dist")
plt.ylabel("Steering Angle (deg)")
'''
plt.show()
