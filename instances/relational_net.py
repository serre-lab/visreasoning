import tensorflow as tf
from tensorflow.contrib import rnn
from components import processors
from instances import layer_instances
from operator import mul
import numpy as np
import ipdb


class RN_no_lstm(processors.BaseFeedforwardProcessor):
    """
    A multilayer convolutional neural net (conv/pool layers -> FC Layers)
    """

    def initialize_vars(self, num_categories,
                        num_CP_layers, num_CP_features, convnet_type,
                        num_RN_layers, num_RN_features, num_RN_question_dims,
                        num_MLP_layers, num_MLP_features, MLP_use_dropout,
                        initial_conv_rf_size, interm_conv_rf_size, pool_rf_size, pool_stride_size, conv_stride_size,
                        activation_type='relu', trainable=True, hamstring_factor=1.0):
        
        if convnet_type=='theirs':
            self.convnet = RN_convnet_theirs(name=self.name+'_convnet',
                                             input_size=self.input_size,
                                             batch_size=self.batch_size,
                                             gpu_addresses=self.gpu_addresses)
            self.convnet.initialize_vars(num_CP_layers=num_CP_layers,
                                         num_CP_features=num_CP_features,
                                         initial_conv_rf_size=initial_conv_rf_size,
                                         interm_conv_rf_size=interm_conv_rf_size,
                                         pool_rf_size=pool_rf_size,
                                         pool_stride_size=pool_stride_size,
                                         conv_stride_size = conv_stride_size,
                                         activation_type=activation_type,
                                         trainable=trainable,
                                         hamstring_factor=hamstring_factor)
        elif convnet_type=='ours':
            self.convnet = RN_convnet_ours(name=self.name+'_convnet',
                                           input_size=self.input_size,
                                           batch_size=self.batch_size,
                                           gpu_addresses=self.gpu_addresses)
            self.convnet.initialize_vars(num_CP_layers=num_CP_layers,
                                         num_CP_features=num_CP_features,
                                         initial_conv_rf_size=initial_conv_rf_size,
                                         interm_conv_rf_size=interm_conv_rf_size,
                                         pool_rf_size=pool_rf_size,
                                         pool_stride_size=pool_stride_size,
                                         conv_stride_size=conv_stride_size,
                                         activation_type=activation_type,
                                         trainable=trainable,
                                         hamstring_factor=hamstring_factor)
        else:
            raise ValueError('convnet_type should be theirs or ours')
        conv_out_size = self.convnet.get_output_size()
        paired_featurevector_size = [conv_out_size[0],
                                     conv_out_size[1],
                                     2*conv_out_size[2]+num_RN_question_dims]
        self.persistent_lstm_state = tf.get_variable(name=self.name + '/_q',
                                                     shape=[1, 1, num_RN_question_dims],
                                                     dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer(),
                                                     trainable=trainable)

        self.gtheta_conv = RN_gtheta_conv(name=self.name+'_gtheta',
                                         input_size=paired_featurevector_size,
                                         batch_size=self.batch_size,
                                         gpu_addresses=self.gpu_addresses)
        self.gtheta_conv.initialize_vars(num_gtheta_layers=num_RN_layers,
                                        num_gtheta_features=num_RN_features,
                                        activation_type=activation_type,
                                        trainable=trainable,
                                        hamstring_factor=hamstring_factor)
        global_mlp_input_size = [1,1,self.gtheta_conv.get_output_size()[2]]

        self.global_mlp = RN_mlp(name=self.name+'_global_mlp',
                                 input_size=global_mlp_input_size,
                                 batch_size=self.batch_size,
                                 gpu_addresses=self.gpu_addresses)
        self.global_mlp.initialize_vars(num_MLP_layers=num_MLP_layers,
                                        num_MLP_features=num_MLP_features,
                                        num_categories=num_categories,
                                        dropout=MLP_use_dropout,
                                        activation_type=activation_type,
                                        trainable=trainable,
                                        hamstring_factor=hamstring_factor)
        self.output_size = self.global_mlp.get_output_size()

    def run(self, X, dropout_keep_prob=1.):

        conv_output = self.convnet.run(X)
        conv_output_size = self.convnet.get_output_size()
        persistent_lstm_state_blown = tf.tile(tf.expand_dims(self.persistent_lstm_state, axis=0),[self.batch_size,
                                                                                                  conv_output_size[0],
                                                                                                  conv_output_size[1],
                                                                                                  1])
        # generate an input array with paired feature vectors
        first_pair = True
        for y1 in range(conv_output_size[0]):
            for x1 in range(conv_output_size[1]):
                feat1 = tf.expand_dims(tf.expand_dims(conv_output[:, y1, x1, :], axis=1), axis=1)
                feat1_blown = tf.tile(feat1,[1,
                                             conv_output_size[0],
                                             conv_output_size[1],
                                             1])
                print('next')
                paired = tf.concat([conv_output,
                                    feat1_blown,
                                    persistent_lstm_state_blown], axis=3)
                if first_pair:
                    relational_output = self.gtheta_conv.run(paired)
                    first_pair = False
                else:
                    relational_output += self.gtheta_conv.run(paired)
        relational_output_folded = tf.reduce_sum(relational_output, axis=[1,2], keep_dims=True)

        final_out = self.global_mlp.run(relational_output_folded,dropout_keep_prob=dropout_keep_prob)
        return final_out

    def set_batch_size(self, batch_size):
        """
        Sets the batch size:

        Inputs:

        batch_size: (int)
        """
        self.batch_size = batch_size
        self.convnet.set_batch_size(batch_size)
        self.gtheta_conv.set_batch_size(batch_size)
        self.global_mlp.set_batch_size(batch_size)


class RN_convnet_theirs(processors.BaseFeedforwardProcessor):
    def initialize_vars(self, num_CP_layers, num_CP_features,
                        initial_conv_rf_size, interm_conv_rf_size, pool_rf_size=[3, 3], pool_stride_size=[2, 2], conv_stride_size = [1,1],
                        activation_type='relu', trainable=True, hamstring_factor=1.0):

        layer_list = []
        intermediate_output_size = self.get_output_size()
        num_features = int(num_CP_features * hamstring_factor)
        layer_ind = 0

        for ii in range(num_CP_layers):
            kth_device = np.mod(layer_ind,len(self.gpu_addresses))
            layer_ind += 1
            with tf.device('/gpu:'+str(self.gpu_addresses[kth_device])):
                conv_rf_size = initial_conv_rf_size if (ii == 0) else interm_conv_rf_size


                # construct conv layer
                layer_list.append(
                    layer_instances.Conv2dLayer(name=self.name + '/conv_' + str(ii + 1),
                                                input_size=intermediate_output_size,
                                                batch_size=self.batch_size,
                                                trainable=trainable))
                layer_list[-1].initialize_vars(rf_size=conv_rf_size,
                                               output_channels=num_features,
                                               stride=conv_stride_size,
                                               activation_type=activation_type)
                intermediate_output_size = layer_list[-1].get_output_size()
                self.add_layer(layer_list[-1])

                # construct pool layer
                layer_list.append(
                    layer_instances.Maxpool2dLayer(name=self.name + '/pool_' + str(ii + 1),
                                                   input_size=intermediate_output_size,
                                                   batch_size=self.batch_size))
                layer_list[-1].initialize_vars(rf_size=pool_rf_size,
                                               stride=pool_stride_size)
                intermediate_output_size = layer_list[-1].get_output_size()
                self.add_layer(layer_list[-1])

        self.output_size = layer_list[-1].get_output_size()

class RN_convnet_ours(processors.BaseFeedforwardProcessor):
    def initialize_vars(self, num_CP_layers, num_CP_features,
                        initial_conv_rf_size, interm_conv_rf_size, pool_rf_size=[3, 3], pool_stride_size=[2, 2], conv_stride_size=[1,1],
                        activation_type='relu', trainable=True, hamstring_factor=1.0):
        
        layer_list = []
        intermediate_output_size = self.get_output_size()
        layer_ind = 0
        for ii in range(num_CP_layers):
            kth_device = np.mod(layer_ind,len(self.gpu_addresses))
            layer_ind += 1
            with tf.device('/gpu:'+str(self.gpu_addresses[kth_device])):
                conv_rf_size = initial_conv_rf_size if (ii == 0) else interm_conv_rf_size
                num_features = int(num_CP_features*hamstring_factor) if (ii==0) else int(num_features*interm_conv_rf_size[0])

                # construct conv layer
                layer_list.append(
                    layer_instances.Conv2dLayer(name=self.name + '/conv_' + str(ii + 1),
                                                input_size=intermediate_output_size,
                                                batch_size=self.batch_size,
                                                trainable=trainable))
                layer_list[-1].initialize_vars(rf_size=conv_rf_size,
                                               output_channels=num_features,
                                               stride=conv_stride_size,
                                               activation_type=activation_type)
                intermediate_output_size = layer_list[-1].get_output_size()
                self.add_layer(layer_list[-1])

                # construct pool layer
                layer_list.append(
                    layer_instances.Maxpool2dLayer(name=self.name + '/pool_' + str(ii + 1),
                                                   input_size=intermediate_output_size,
                                                   batch_size=self.batch_size))
                layer_list[-1].initialize_vars(rf_size=pool_rf_size,
                                               stride=pool_stride_size)
                intermediate_output_size = layer_list[-1].get_output_size()
                self.add_layer(layer_list[-1])

        self.output_size = layer_list[-1].get_output_size()

class RN_gtheta_conv(processors.BaseFeedforwardProcessor):
    def initialize_vars(self, num_gtheta_layers, num_gtheta_features,
                        activation_type='relu', trainable=True, hamstring_factor=1.0):

        layer_list = []
        intermediate_output_size = self.get_output_size()
        layer_ind = 0

        # construct final mlp (Phi)
        for jj in range(num_gtheta_layers):
            kth_device = np.mod(layer_ind, len(self.gpu_addresses))
            layer_ind += 1
            with tf.device('/gpu:' + str(self.gpu_addresses[kth_device])):
                num_features = int(num_gtheta_features*hamstring_factor)

                layer_list.append(
                    layer_instances.Conv2dLayer(name=self.name + '/gtheta_' + str(jj + 1),
                                                input_size=intermediate_output_size,
                                                batch_size=self.batch_size,
                                                trainable=trainable))
                layer_list[-1].initialize_vars(rf_size=[1,1],
                                               output_channels=num_features,
                                               stride=[1, 1],
                                               activation_type=activation_type)
                intermediate_output_size = layer_list[-1].get_output_size()

                self.add_layer(layer_list[-1])

        self.output_size = layer_list[-1].get_output_size()



class RN_mlp(processors.BaseFeedforwardProcessor):
    def initialize_vars(self, num_MLP_layers, num_MLP_features, num_categories, dropout,
                        activation_type='relu', trainable=True, hamstring_factor=1.0):

        layer_list = []
        intermediate_output_size = self.get_output_size()
        layer_ind = 0

        # construct final mlp (Phi)
        for jj in range(num_MLP_layers):
            kth_device = np.mod(layer_ind, len(self.gpu_addresses))
            layer_ind += 1
            with tf.device('/gpu:' + str(self.gpu_addresses[kth_device])):
                num_features = int(num_MLP_features*hamstring_factor) if (jj<num_MLP_layers-1) else num_categories
                activation = activation_type if (jj<num_MLP_layers-1) else 'raw'

                layer_list.append(layer_instances.FCLayer(name=self.name + '/MLP_' + str(jj + 1),
                                                          input_size=intermediate_output_size,
                                                          batch_size=self.batch_size,
                                                          trainable=trainable))
                layer_list[-1].initialize_vars(output_channels=num_features,
                                               activation_type=activation)
                intermediate_output_size = layer_list[-1].get_output_size()

                self.add_layer(layer_list[-1])
            if (jj == 0) & (dropout==True):  # add dropout layer
                layer_list.append(layer_instances.DropoutLayer(name=self.name + '/dropout_' + str(jj + 1),
                                                               input_size=intermediate_output_size,
                                                               batch_size=self.batch_size))
                layer_list[-1].initialize_vars(dropout_multiplier=1.)
                intermediate_output_size = layer_list[-1].get_output_size()

                self.add_layer(layer_list[-1])

        self.output_size = layer_list[-1].get_output_size()


