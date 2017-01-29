from .model import PreprocessImage
import csv
import datetime
import numpy as np
import os.path
import random
import scipy.ndimage

class GeneratorBase(object):
    '''Prototype for a functor generator'''
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()

def normalize_path(directory, path):
    path = path.strip()
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(directory, path))
    return path

class StochasticGeneratorPool(GeneratorBase):
    '''Join several generators and select randomly from them.'''
    def __init__(self, generators):
        self._generators = generators
        try:
            self._length = sum([len(g) for g in self._generators])
        except TypeError:
            self._length = None
    def next(self):
        random.shuffle(self._generators)
        for g in self._generators:
            try:
                return next(g)
            except StopIteration:
                pass
    def __len__(self):
        return sum([len(g) for g in self._generators])

def DrivingLogReader(directory):
    directory = directory.strip()
    log_name = os.path.join(directory, 'driving_log.csv')
    with open(log_name) as f:
        reader = csv.DictReader(f, delimiter=',', fieldnames=['center', 'left', 'right', 'steering', 'gas', 'brake', 'speed'])
        for row in reader:
            try:            
                row['center'] = normalize_path(directory, row['center'])
                row['left'] = normalize_path(directory, row['left'])
                row['right'] = normalize_path(directory, row['right'])
                row['steering'] = float(row['steering'])
                row['gas'] = float(row['gas'])
                row['brake'] = float(row['brake'])
                row['speed'] = float(row['speed'])
                dateparts = row['center'][:-4].split('/')[-1].split('_')[1:]
                dateparts = [int(x) for x in dateparts]
                row['timestamp'] = datetime.datetime(dateparts[0], dateparts[1], dateparts[2], dateparts[3], dateparts[4], dateparts[5], 1000*dateparts[6])
            except (ValueError, AttributeError):
                # This must be the header row, or there's no steering for some reason
                continue
            else:
                yield row

class DataGenerator(GeneratorBase):
    '''
        This generator reads the data from the center image and the steering
        directly, with no augmentation or processing. It's primarily useful
        for validation
    '''
    def __init__(self, directory, batch_size):
        self.data = list()
        self.batch_size = batch_size
        reader = DrivingLogReader(directory)
        for r in reader:
            if os.path.isfile(r['center']):
                self.data.append((r['center'], r['steering']))
        random.shuffle(self.data)
        self.index = 0
    def __len__(self):
        return len(self.data)
    def next(self):
        batch_end = min(self.index + self.batch_size, len(self.data))
        inputs = list()
        targets = list()
        for sample in self.data[self.index:batch_end]:
            try:
               image = scipy.ndimage.imread(sample[0], mode='RGB')
            except IOError:
                print("Couldn't open {}".format(sample[0]))
                continue
            inputs.append(PreprocessImage(image))
            targets.append(sample[1])
        self.index = batch_end
        if self.index >= len(self.data):
            self.index = 0
        return((np.stack(inputs), targets))

class MultiCameraGenerator(DataGenerator):
    ''' This generator reads the data from all three images.
        
        The steering angle for the side images is offset by
        the amount of the offset term.
    '''
    def __init__(self, directory, batch_size, offset=0.25):
        self.data = list()
        self.batch_size = batch_size
        for r in DrivingLogReader(directory):
            if os.path.isfile(r['center']):
                self.data.append((r['center'], r['steering']))
            if os.path.isfile(r['left']):
                self.data.append((r['left'], r['steering'] + offset))
            if os.path.isfile(r['right']):
                self.data.append((r['right'], r['steering'] - offset))
        random.shuffle(self.data)
        self.index = 0

