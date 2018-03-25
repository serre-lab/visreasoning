import tensorflow as tf
import numpy as np
from components import component
from operator import mul

# NOTE: input_size and output_size exclude batch dimension but the actual input and output must have batch dimension as its first dim.

class BaseLayer(component.UniversalComponent):
    """
    Contains variables and operations common to all layers. Sets attributes, checks for inconsistencies, builds variable list.
    """

    def __init__(self, name, input_size, batch_size=1, trainable=True):
        """
        Inputs:

        name : 			(str) layer name
        input_size : 	(list) dimensions of layer input [height, width, channels]
        batch_size : 	(int) batch size
        trainable : 	(bool) If true, layer weights can be updated duirng learning.
        """
        self.name = name  # must be a string
        self.input_size = input_size  # must be a list (height-width-inchannels)
        self.output_size = None
        self.batch_size = batch_size  # must be an int
        self.output = None
        self.trainable = trainable

        if not isinstance(self.name, str):
            raise TypeError("BaseLayer: name should be string.")
        if not isinstance(self.input_size, list):
            raise TypeError("BaseLayer: input_size should be a list of ints.")
        elif not len(self.input_size) == 3:
            raise ValueError("BaseLayer: input_size should be shaped like height-width-inchannels.")
        else:
            for idim in range(len(self.input_size)):
                    if not isinstance(self.input_size[idim], int):
                        raise TypeError("BaseLayer: input_size should be a list of ints.")
        if not (isinstance(self.batch_size, int) and self.batch_size > 0):
            raise TypeError("BaseLayer: batch_size should be a positive int.")


    def run(self, X):
        """
        Checks to see if current input is properly shaped.

        Inputs :
        X : (tensor) an input tensor of size [batch_size] + self.input_size
        """

        if '1.0' in tf.__version__:
            if not X.shape.as_list()[1:] == self.input_size:
                raise TypeError("BaseLayer: input (X)  has different shape (excluding batch) than input_size.")


def get_variables(self):
    """
    Builds variable list
    """
    var_list = []
    if hasattr(self, 'weights'):
        var_list = var_list + [self.weights]
    if hasattr(self, 'biases'):
        var_list = var_list + [self.biases]
    return var_list


class AttentionLayer(BaseLayer):
    """
    Abstract attentional layer. Super-object for spatial and feature attention layers. Sets mask size and output size.
    """

    def initialize_vars(self):
        """
        Sets output size and mask.
        """
        self.output_size = self.input_size
        self.mask = tf.ones([self.batch_size] + self.input_size, dtype=tf.float32) / reduce(mul, self.input_size)

    def run(self, X):
        """
        Checks input size and multiplies mask with input.

        Inputs :
        X : (tensor) input tensor of size [batch_size] + self.input_size
        """
        super(AttentionLayer, self).run(X)
        output = tf.multiply(X, self.mask, name=self.name + '_out')

        if '1.0' in tf.__version__:
            if not output.shape.as_list()[1:] == self.output_size:
                raise TypeError(
                    "AttentionLayer: output (Y) has different shape (excluding batch) than input_size. Could be a bug in implementation.")
        return output

    def set_batch_size(self, batch_size):
        """
        Sets batch size.

        Inputs :
        batch_size : (int) batch_size
        """
        super(AttentionLayer, self).set_batch_size(batch_size)
        self.initialize_vars()

