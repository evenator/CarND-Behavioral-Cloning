import keras
from keras.layers import Convolution2D, Dense, Flatten, Dropout
from keras.models import Sequential
import scipy.misc

#TODO: Add cropping
def PreprocessImage(img):
    return scipy.misc.imresize(img, (66, 200), 'bilinear')

class NormalizeImage(keras.engine.topology.Layer):
    '''Custom Keras layer that performs image normalization '''
    def __init__(self, **kwargs):
        super(NormalizeImage, self).__init__(**kwargs)
    def build(self, input_shape):
        super(NormalizeImage, self).build(input_shape)
    def call(self, x, mask=None):
        x = x - keras.backend.mean(x, (1,2), keepdims=True)
        x_maxabs  = keras.backend.max(keras.backend.abs(x), (1,2), keepdims=True)
        return x / x_maxabs
    def get_output_shape_for(self, input_shape):
        return input_shape

#TODO: Try ELU
def Dave2(dropout=0.0):
    '''
    A keras implementation of Nvidia's Dave2 architecture.
    See "End to End Learning for Self-Driving Cars" by Boiarski et al
    25 April 2016
    '''
    model = Sequential()
    model.add(NormalizeImage(input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model

# Define custom keras layers for loading from JSON
keras_objects = {'NormalizeImage': NormalizeImage}
