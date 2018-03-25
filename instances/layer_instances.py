import tensorflow as tf
import numpy as np
from components import component
from components import layers
from operator import mul
from operator import sub
from operator import add
from operator import truediv
import ipdb

# NOTE: input_size and output_size exclude batch dimension but the actual input and output must have batch dimension as its first dim.
# NOTE: relu is automatically applied to conv or fc layers

class FCLayer(layers.BaseLayer):
    """
    A fully connected layer. Sets layer attributes, passes data through layer.
    """

    def initialize_vars(self, output_channels, activation_type='relu', has_biases=True):
        """
        Inputs :

        output_channels : 	(int) number of output neurons
        activation_type : 	(str) the activation function. acceptable strings are 'raw' for identity, 'relu' for relu and 'sigmoid' for sigmoid
        has_biases : 		(bool) If true, the layer will have biases. Otherwise, the layer is just a linear transormation.
        """
        self.rf_size = self.input_size  # height-width-inchannels
        self.output_size = [1, 1, output_channels]  # 1-1-outchannels
        self.activation_type = activation_type
        self.has_biases = has_biases

        if not isinstance(self.output_size, list):
            raise TypeError("FCLayer: output_size should be a list of ints.")
        elif not len(self.output_size) == 3:
            raise ValueError("FCLayer: output_size should be shaped by 1-1-outchannels.")
        else:
            for idim in range(len(self.output_size)):
                if not isinstance(self.output_size[idim], int):
                    raise TypeError("FCLayer: output_size should be a list of ints.")
        if not self.output_size[0:2] == [1, 1]:
            raise ValueError("FCLayer: output_size should be shaped by 1-1-outchannels.")
        self.input_size_flattened = reduce(mul, self.input_size)

        # [inchannels, outchannels]
        self.weights = tf.get_variable(name=self.name + '/_W', shape=[self.input_size_flattened, output_channels],
                                       dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=self.trainable)

        # [1, outchannels]
        if self.has_biases == True:
            self.biases = tf.get_variable(name=self.name + '/_b', shape=[1, output_channels], dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(), trainable=self.trainable)


    def run(self, X):
        """
        Inputs:

        X: (tensor) the input tensor of size [self.batch_size] + self.input_size
        """
        super(FCLayer, self).run(X)
        flattened_input = tf.reshape(X, [self.batch_size, self.input_size_flattened])
        if self.has_biases == True:
            flattened_output = tf.add(tf.matmul(flattened_input, self.weights, name=self.name + '_IP'), self.biases,
                                      name=self.name + '_BIAS')
        else:
            flattened_output = tf.matmul(flattened_input, self.weights, name=self.name + '_IP')
        if self.activation_type is 'relu':
            flattened_output = tf.nn.relu(flattened_output)
        elif self.activation_type is 'sigmoid':
            flattened_output = tf.sigmoid(flattened_output)
        elif self.activation_type is not 'raw':
            raise ValueError('activation_type should be either relu or raw')
        output = tf.reshape(flattened_output, [self.batch_size] + self.output_size, name=self.name + '_out')  # batch-1-1-outchannels

        if '1.0' in tf.__version__:
            if not output.shape.as_list()[1:] == self.output_size:
                raise TypeError(
                    "FCLayer: output (Y) has different shape (excluding batch) than input_size. Could be a bug in implementation.")
        return output


class Conv2dLayer(layers.BaseLayer):
    """
    2D convolutional layer. Sets layer attributes, passes data through layer.
    """

    def initialize_vars(self, rf_size, output_channels, stride, activation_type='relu'):
        """
        Inputs :

        rf_size : 			(list) The receptive field size of the kernel,  [height, width]
        output_channels : 	(int) the number of features in the output
        stride : 			(int) the stride of the convolution
        activation_type : 	(str) activation function: 'raw' (identity), 'relu': (rectification), 'sigmoid' (sigmoid)

        """
        self.rf_size = rf_size  # height-width
        self.stride = stride
        self.activation_type = activation_type

        if not isinstance(self.rf_size, list):
            raise TypeError("Conv2dLayer: rf_size should be a list of ints.")
        elif not len(self.rf_size) == len(self.input_size) - 1:
            raise ValueError("Conv2dLayer: rf_size should be shaped by height-width")
        else:
            for idim in range(len(self.rf_size)):
                if not isinstance(self.rf_size[idim], int):
                    raise TypeError("Conv2dLayer: rf_size should be a list of ints.")
            for idim in range(len(self.rf_size)):
                if not self.rf_size[idim] <= self.input_size[idim]:
                    raise ValueError("Conv2dLayer: rf_size should not be larger than the spatial dims of input_size.")
        if not isinstance(output_channels, int):
            raise TypeError("Conv2dLayer: output_channels should be an int.")
        if not isinstance(self.stride, list):
            raise TypeError("Conv2dLayer: stride should be a list of ints.")
        else:
            for idim in range(len(self.stride)):
                if not isinstance(self.stride[idim], int):
                    raise TypeError("Conv2dLayer: stride should be a list of ints.")
        if not len(stride) == 2:
            raise ValueError("Conv2dLayer: stride should be shaped by height-width.")

        # [height, width, out_channels]
        # side length (height and width) is side length = floor((input_size - rf_size)/stride)+1
        self.output_size = map(add, map(lambda x: int(np.floor(x)),
                                        map(truediv, map(sub, self.input_size[:2], self.rf_size), stride)),
                               [1] * len(rf_size))
        self.output_size = map(lambda x: int(x), self.output_size) + [output_channels]

        # rf_size + [inchannels, outchannels]
        self.weights = tf.get_variable(name=self.name + '/_w',
                                       shape=self.rf_size + [self.input_size[2]] + [output_channels], dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       trainable=self.trainable)
        self.biases = tf.get_variable(name=self.name + '/_b', shape=[1, 1, output_channels], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      trainable=self.trainable)

    def run(self, X):
        """
        Passes data trough the convolution.

        Inputs :

        X : (tensor) [batch, height, width, channels]
        """
        super(Conv2dLayer, self).run(X)

        output = tf.add(
            tf.nn.conv2d(X, filter=self.weights, strides=[1] + self.stride + [1], padding="VALID", data_format="NHWC",
                         name=self.name + '_CONV'), self.biases, name=self.name + '_BIAS')

        if self.activation_type is 'relu':
            output = tf.nn.relu(output, name=self.name + '_out')
        elif self.activation_type is 'sigmoid':
            output = tf.sigmoid(output, name=self.name + '_out')
        elif self.activation_type is 'halfsquare':
            output = tf.square(tf.nn.relu(output), name=self.name + '_out')
        elif self.activation_type is not 'raw':
            raise ValueError('activation_type should be either relu or raw')

        if '1.0' in tf.__version__:
            if not output.shape.as_list()[1:] == self.output_size:
                raise TypeError(
                    "Conv2dLayer: output (Y) has different shape (excluding batch) than input_size. Could be a bug in implementation.")
        return output


class Maxpool2dLayer(layers.BaseLayer):
    """
    A layer that takes a max pool over spatially discrete blocks
    """

    def initialize_vars(self, rf_size, stride):
        """
        Inputs :

        rf_size :	(list) [height, width]
        stride : 	(list) [row stride, col stride]
        """
        self.rf_size = rf_size
        self.stride = stride

        if not isinstance(self.rf_size, list):
            raise TypeError("Maxpool2dLayer: rf_size should be a list of ints.")
        elif not len(self.rf_size) == len(self.input_size) - 1:
            raise ValueError("Maxpool2dLayer: rf_size should be shaped by height-width")
        else:
            for idim in range(len(self.rf_size)):
                if not isinstance(self.rf_size[idim], int):
                    raise TypeError("Maxpool2dLayer: rf_size should be a list of ints.")
            for idim in range(len(self.rf_size)):
                if not self.rf_size[idim] <= self.input_size[idim]:
                    raise ValueError(
                        "Maxpool2dLayer: rf_size should not be larger than the spatial dims of input_size.")
        if not isinstance(self.stride, list):
            raise TypeError("Maxpool2dLayer: stride should be a list of ints.")
        else:
            for idim in range(len(self.stride)):
                if not isinstance(self.stride[idim], int):
                    raise TypeError("Maxpool2dLayer: stride should be a list of ints.")
        if not len(stride) == 2:
            raise ValueError("Maxpool2dLayer: stride should be shaped by height-width.")

        # [height, width, out_channels]
        # side length (height and width) is side length = floor((input_size - rf_size)/stride)+1
        self.output_size = map(add, map(lambda x: int(np.floor(x)),
                                        map(truediv, map(sub, self.input_size[:2], self.rf_size), stride)),
                               [1] * len(rf_size))
        self.output_size = map(lambda x: int(x), self.output_size) + [self.input_size[2]]

    def run(self, X):
        """
        Passes data through the pooling filter.

        Inputs:

        X: (tensor) [batch, height, width, channels]
        """
        super(Maxpool2dLayer, self).run(X)

        output = tf.nn.pool(X, window_shape=self.rf_size, pooling_type="MAX", padding="VALID", dilation_rate=None,
                            strides=self.stride, data_format="NHWC",name=self.name + '_out')

        if '1.0' in tf.__version__:
            if not output.shape.as_list()[1:] == self.output_size:
                raise TypeError(
                    "Maxpool2dLayer: output (Y) has different shape (excluding batch) than input_size. Could be a bug in implementation.")
        return output


class Itemavg2dLayer(layers.BaseLayer):
    """
    A layer that takes an average over spatially discrete blocks. Like an average pooling.
    """

    def initialize_vars(self, item_size):
        """
        Inputs:

        item_size: (list) [height, width] The size of the blocks over which the average is taken.
        """
        self.item_size = item_size  # height-width

        if not isinstance(self.item_size, list):
            raise TypeError("Itemavg2dLayer: item_size should be a list of ints.")
        elif not len(self.item_size) == len(self.input_size) - 1:
            raise ValueError("Itemavg2dLayer: item_size should be shaped by height-width")
        else:
            for idim in range(len(self.item_size)):
                if not isinstance(self.item_size[idim], int):
                    raise TypeError("Itemavg2dLayer: item_size should be a list of ints.")
            for idim in range(len(self.item_size)):
                if not self.item_size[idim] <= self.input_size[idim]:
                    raise ValueError(
                        "Itemavg2dLayer: item_size should not be larger than the spatial dims of input_size.")
        if self.input_size[0] % item_size[0] + self.input_size[1] % item_size[1] > 0:
            raise ValueError(
                "Itemavg2dLayer: edge elements in input are not included for combination. Adjust item_size")

        # output size (spatial) is floor((input_size - rf_size)/stride)+1
        # height-width-outchannels

        self.output_size = item_size + [self.input_size[2]]

    def run(self, X):
        """
        Passes data through average pooling.

        Inputs:

        X: (tensor) the input to the average pooling layer [batch, height, width, channels].
        """
        super(Itemavg2dLayer, self).run(X)

        output = tf.zeros([self.batch_size] + self.output_size)
        for xpos in range(np.int(self.input_size[1] / self.item_size[1])):
            for ypos in range(np.int(self.input_size[0] / self.item_size[0])):
                begin = [ypos * self.item_size[0], xpos * self.item_size[1]]
                item = tf.slice(X, [0] + begin + [0], [self.batch_size] + self.item_size + [self.input_size[2]])
                output = output + item
        output = tf.identity(output, name=self.name + '_out')

        if '1.0' in tf.__version__:
            if not output.shape.as_list()[1:] == self.output_size:
                raise TypeError(
                    "Itemavg2dLayer: output (Y) has different shape (excluding batch) than input_size. Could be a bug in implementation.")
        return output

class DropoutLayer(layers.BaseLayer):
    """
    A fully connected layer. Sets layer attributes, passes data through layer.
    """

    def initialize_vars(self, dropout_multiplier=1.):
        """
        Inputs :
        dropout_multiplier : 	multiplier (from zero to one) used to scale keep_prob
        """
        self.dropout_multiplier = dropout_multiplier
        if (self.dropout_multiplier>1) | (self.dropout_multiplier<=0):
            raise ValueError('dropout multipler should range between 0 and 1.')
        self.output_size = self.input_size

    def run(self, X, dropout_keep_prob=1.):
        """
        Inputs:

        X: (tensor) the input tensor of size [self.batch_size] + self.input_size
        """
        super(DropoutLayer, self).run(X)

        output = tf.nn.dropout(X,keep_prob=dropout_keep_prob*self.dropout_multiplier,name=self.name + '_out')
        return output


class SAttentionLayer(layers.AttentionLayer):
    """
    Implements spatial attention. Inherits ``AttentionLayer'' object.
    """

    def set_mask(self, mask):
        """
        Creates a mask the size of the input to the attentional layer. Tiles over space.

        Inputs:

        mask: (tensor) A tensor which is tiled to be the same shape as the input data. Multiplied with input in
              the AttentionLayer super object method 'run'
        """

        if not isinstance(mask, tf.Tensor):
            raise TypeError("SAttentionLayer: mask must be a tf.Tensor.")
        if '1.0' in tf.__version__:
            if not mask.shape.as_list()[1:3] == self.input_size[:2]:
                raise TypeError("SAttentionLayer: mask has different shape (excluding batch and channels) than input_size.")

        self.mask = tf.tile(mask, tf.stack([1, 1, 1, self.input_size[2]]))  # batch-height-width-inchannels


class FAttentionLayer(layers.AttentionLayer):
    """
    Implements feature-based attention. Inherits ``AttentionLayer'' object.
    """

    def set_mask(self, mask):
        """
        Creates a mask the size of the input to the attentional layer. Tiles over channels.

        Inputs:

        mask: (tensor)  A tensor which is tiled to be the same shape as the input data. Multiplied with input in
              the AttentionLayer super object method 'run'
        """
        if not isinstance(mask, tf.Tensor):
            raise TypeError("FAttentionLayer: mask must be a tf.Tensor.")
        if '1.0' in tf.__version__:
            if not mask.shape.as_list()[3] == self.input_size[2]:
                raise TypeError("FAttentionLayer: mask has different shape (excluding batch and spatial) than input_size.")

        self.mask = tf.tile(mask,
                            tf.stack([1, self.input_size[0], self.input_size[1], 1]))  # batch-height-width-inchannels
