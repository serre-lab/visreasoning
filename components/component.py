import tensorflow as tf
import numpy as np

class UniversalComponent(object):
    """
    An abstract component with methods corresponding to all processors.
    """

    def get_output_size(self):
        """
        Retrieves output size of processor.
        """
        return list(self.output_size)

    def get_input_size(self):
        """
        Retrieves output size of processor.
        """
        return list(self.input_size)

    def get_batch_size(self):
        """
        Retrieves batch size of processor.
        """
        return self.batch_size

    def set_batch_size(self, batch_size):
        """
        Sets batch size of processor.
        Inputs:
        batch_size: (int)
        """
        if not (isinstance(batch_size, int) and batch_size > 0):
            raise TypeError("UniversalComponent: batch_size should be a positive int.")
        self.batch_size = batch_size

    def get_variables(self):
        """
        Retrieves variables in processor.
        """
        raise NotImplementedError


class FeedforwardComponent(UniversalComponent):
    '''
    An abstract feedforward component inheriting the UniversalComponent object.
    '''

    def run(self, X):
        """
        Not implemented.
        """
        raise NotImplementedError


class RecurrentComponent(UniversalComponent):
    '''
    An abstract recurrent component inhereiting the UniversalComponent object.
    '''

    def run(self, X, recurrent_vars):
        """
        Not implemented.
        """
        raise NotImplementedError
