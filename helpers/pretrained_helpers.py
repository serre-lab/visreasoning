import os
import tensorflow as tf
from helpers import utility

def find_best_model():
    # TODO: When working the 'new' result files, use this script to select the ckpt for the best model.
    return

def save_model(session, save_path, ckpt_filename):

    utility.llprint('Storing snapshot')
    saver = tf.train.Saver()
    if not '.ckpt' in ckpt_filename:
        ValueError('.ckpt should be in ckpt_filename.')
    saver.save(session, os.path.join(save_path, ckpt_filename))
    utility.llprint("Done!\n")


def load_model_with_graph(session, meta_file_path, ckpt_file_path, tensor_name_list):
    "Load a pretrained model along with the original graph."
    saver = tf.train.import_meta_graph(meta_file_path)
    # graph_def = tf.get_default_graph().as_graph_def()\
    tensor_list = []
    for i, tn in enumerate(tensor_name_list):
        tensor_list.append(tf.get_default_graph().get_tensor_by_name(tn))
    saver.restore(session, ckpt_file_path)

    return tensor_list


def load_model_without_graph(session, ckpt_file_path):
    "Load a pretrained model when you have a predefined graph with matching variable names."
    saver = tf.train.Saver()
    saver.restore(session, ckpt_file_path)

