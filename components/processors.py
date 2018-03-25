import tensorflow as tf
import numpy as np
from components import layers
from instances import layer_instances
from components import component


class BaseFeedforwardProcessor(component.FeedforwardComponent):
    '''
    Abstract feedforward object. Sets attributes (dimensions, number of layers, etc), constructs layer list,
    passes data through layers, sets batch size, displays variables in this processor.
    '''

    def __init__(self, name, input_size, batch_size=1, gpu_addresses=[0]):
        """
        Inputs:

        name : 			(str) Name of processor
        input_size : 	(list) [height, width, channels] size of input to processor
        batch_size : 	(int) batch size
        """
        self.name = name
        self.input_size = list(input_size)
        self.output_size = list(input_size)
        self.batch_size = batch_size
        self.layer_list = []
        self.num_layers = 0
        self.spatial_attention_layer = None
        self.feature_attention_layer = None
        self.gpu_addresses=gpu_addresses

        if not isinstance(self.input_size, list):
            raise TypeError("BaseFeedforwardProcessor: input_size should be a list of ints.")
        elif not len(self.input_size) == 3:
            raise ValueError("BaseFeedforwardProcessor: input_size should be shaped by height-width-inchannels.")
        else:
            for idim in range(len(self.input_size)):
                if not isinstance(self.input_size[idim], int):
                    raise TypeError("BaseFeedforwardProcessor: input_size should be a list of ints.")
        if not isinstance(self.batch_size, int):
            raise TypeError("BaseFeedforwardProcessor: batch_size should be an int.")

    def add_layer(self, layer, force=False):
        """
        Adds a layer object to the processors layer_list attribute.

        Inputs:

        layer: (layer) the layer to be added to the layer list
        """
        if (not isinstance(layer, layers.BaseLayer)) | isinstance(layer, layers.AttentionLayer):
            raise TypeError("BaseFeedforwardProcessor: layer should be an instance of BaseLayer.")
        if layer.get_output_size() is None:
            raise ValueError("BaseFeedforwardProcessor: layer should be initialized using initialize_vars.")

        # shape check
        if not force:
            if self.output_size != layer.get_input_size():
                raise ValueError(
                    "BaseFeedforwardProcessor: layer should have the same input size as the processor output size.")
            if self.batch_size != layer.get_batch_size():
                raise Warning(
                    "BaseFeedforwardProcessor: layer has different batch size than the processor. Fixing it automatically...")
        layer.set_batch_size(self.batch_size)

        self.layer_list.append(layer)
        self.num_layers += 1
        self.output_size = layer.get_output_size()

    def run(self, X, dropout_keep_prob=1.):
        """
        Passes data through each layer in the layer list in the opposite order in which they were added.

        Input:

        X: (tensor) data to be passed through the processor. [batch, height, width, channels]
        """
        intermediate = X
        layer_ind = 0
        for current_layer in self.layer_list:
            kth_device = np.mod(layer_ind,len(self.gpu_addresses))
            layer_ind += 1
            with tf.device('/gpu:' + str(self.gpu_addresses[kth_device])):
                if isinstance(current_layer,layer_instances.DropoutLayer):
                    intermediate = current_layer.run(intermediate, dropout_keep_prob=dropout_keep_prob)
                else:
                    intermediate = current_layer.run(intermediate)

        return intermediate

    def run_list(self, X, layer_names, dropout_keep_prob=1.):
        intermediate = X
        layer_ind = 0
        output_list = []
        num_layers_checked = 0
        for current_layer in self.layer_list:
            layer_ind += 1
            if isinstance(current_layer,layer_instances.DropoutLayer):
                intermediate = current_layer.run(intermediate, dropout_keep_prob=dropout_keep_prob)
            else:
                intermediate = current_layer.run(intermediate)
            for layer_name in layer_names:
                if current_layer.name.endswith(layer_name):
                    output_list.append(intermediate)
                    num_layers_checked += 1
            if len(layer_names) == 0:
                break
        if num_layers_checked < len(layer_names):
            raise ValueError('layer_names contains layers not in the network')
        return output_list

    def set_batch_size(self, batch_size):
        """
        Sets the batch size:

        Inputs:

        batch_size: (int)
        """
        super(BaseFeedforwardProcessor, self).set_batch_size(batch_size)

        for current_layer in self.layer_list:
            current_layer.set_batch_size(self.batch_size)

    def get_variables(self):
        var_list = []
        for layer in self.layer_list:
            var_list = var_list + layer.get_variables()

        return var_list


class AttendedProcessor(BaseFeedforwardProcessor):
    '''
    A feedforward processor with attention. Inherits the `BasaeFeedforwardProcessor` object
    '''

    def add_attention_spatial(self, layer):
        """
        Adds a spatial attention layer to layer list.

        Inputs:

        layer: (layer) a spatial attention layer
        """
        if self.spatial_attention_layer is not None:
            raise ValueError("AttendedProcessor: spatial attention already exists.")
        if not isinstance(layer, layer_instances.SAttentionLayer):
            raise TypeError("AttendedProcessor: layer should be an instance of SAttentionLayer.")
        if layer.get_output_size() is None:
            raise ValueError("AttendedProcessor: layer should be initialized using initialize_vars.")
        if self.output_size != layer.get_input_size():
            raise ValueError("AttendedProcessor: layer should have the same input size as the processor output size.")
        if self.batch_size != layer.get_batch_size():
            raise Warning(
                "AttendedProcessor: layer has different batch size than the processor. Fixing it automatically...")

        layer.set_batch_size(self.batch_size)
        layer.name = self.name + '/' + layer.name
        self.layer_list.append(layer)
        self.num_layers += 1
        self.spatial_attention_layer = self.num_layers - 1
        self.output_size = layer.get_output_size()
        self.mask_size = layer.input_size[0:2] + [1]  # [height,width,1]

    def add_attention_feature(self, layer):
        """
        Adds a feature attention layer to layer list.

        Inputs:

        layer: (layer) a feature attention layer
        """
        if self.spatial_attention_layer is not None:
            raise ValueError("AttendedProcessor: spatial attention already exists.")
        if not isinstance(layer, layer_instances.FAttentionLayer):
            raise TypeError("AttendedProcessor: layer should be an instance of FAttentionLayer.")
        if layer.get_output_size() is None:
            raise ValueError("AttendedProcessor: layer should be initialized using initialize_vars.")
        if self.output_size != layer.get_input_size():
            raise ValueError("AttendedProcessor: layer should have the same input size as the processor output size.")
        if self.batch_size != layer.get_batch_size():
            layer.set_batch_size(self.batch_size)
            raise Warning(
                "AttendedProcessor: layer has different batch size than the processor. Fixing it automatically...")

        layer.name = self.name + '/' + layer.name
        self.layer_list.append(layer)
        self.num_layers += 1
        self.feature_attention_layer = self.num_layers - 1
        self.output_size = layer.get_output_size()
        self.mask_size = [1, 1] + layer.input_size[2]  # [1,1,numfeats]

    def reset_attention(self):
        """
        Sets attention layers to their initial positions by calling their initialize_vars methods.
        """
        if self.feature_attention_layer is None and self.spatial_attention_layer is None:
            raise Warning("AttendedProcessor: no attention layer exists. This function will do nothing.")
        else:
            if self.feature_attention_layer is not None:
                self.layer_list[self.feature_attention_layer].initialize_vars()
            if self.spatial_attention_layer is not None:
                self.layer_list[self.spatial_attention_layer].initialize_vars()

    def apply_attention_spatial(self, mask):
        """
        Sets the spatial attention mask.

        Inputs:

        mask: (tensor) the mask to be set.
        """
        if self.spatial_attention_layer is None:
            raise Warning("AttendedProcessor: no attention layer exists. This function will do nothing.")
        else:
            self.layer_list[self.spatial_attention_layer].set_mask(mask)

    def get_mask_spatial(self):
        """
        Retrieves the spatial attention mask.
        """
        if self.spatial_attention_layer is None:
            raise Warning("AttendedProcessor: no attention layer exists. This function will do nothing.")
        else:
            return tf.expand_dims(self.layer_list[self.spatial_attention_layer].mask[:, :, :, 0], -1)

    def apply_attention_feature(self, mask):
        """
        Sets the feature attention mask.

        Inputs:

        mask: (tensor) the mask to be set.
        """
        if self.feature_attention_layer is None:
            raise Warning("AttendedProcessor: no attention layer exists. This function will do nothing.")
        else:
            self.layer_list[self.feature_attention_layer].set_mask(mask)

    def get_mask_feature(self):
        """
        Retreives the feature attention map. 
        """
        if self.spatial_attention_layer is None:
            raise Warning("AttendedProcessor: no attention layer exists. This function will do nothing.")
        else:
            return self.layer_list[self.feature_attention_layer].mask[:, 0, 0, :]
            ### Needs expand_dim??



