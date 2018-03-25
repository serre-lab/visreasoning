import os
import sys
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

sys.path.append(os.path.abspath(os.path.join('..','..')))
from helpers import train_helpers
from experiments.regular_n_siamese import params

warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':

    t = time.time()

    num_machines = int(sys.argv[1])
    i_machine = int(sys.argv[2])
    n_gpus = int(sys.argv[3])

    params = params.get_params()
    model_gpu_addresses = range(n_gpus)
    gradient_gpu_address = n_gpus

    num_repeats = 20
    box_extent_list = [[60, 60]] # [[30, 30],[90, 90],[120, 120],[150, 150],[180, 180]]
    num_items_list = [2] # [3,4,5,6]
    item_size_list = [[4, 4]]  # [[3, 3],[5, 5],[6, 6],[7, 7]]

    params['train_data_init_args']['problem_type'] = 'SD' # 'SR'
    params['train_data_init_args']['organization'] = 'raw' # 'obj'
    params['model_init_args']['organization'] = params['train_data_init_args']['organization']

    results_root = '/home/jk/PSVRT_test_result/'
    summary_dir = os.path.join(results_root, params['train_data_init_args']['organization'], params['train_data_init_args']['problem_type'])

    kth_job = 0

    for be, box_extent in enumerate(box_extent_list):
        for ps, item_size in enumerate(item_size_list):
            for ni, n in enumerate(num_items_list):
                for rep in range(num_repeats):
                    params['train_data_init_args']['item_size'] = list(item_size)
                    params['train_data_init_args']['box_extent'] = list(box_extent)
                    params['train_data_init_args']['num_items'] = n
                    params['model_init_args']['num_items'] = n
                    params['val_data_init_args'] = params['train_data_init_args'].copy()

                    params['save_learningcurve_as'] = os.path.join(summary_dir,
                                                                   str(box_extent),
                                                                   str(n),
                                                                   str(item_size),
                                                                   'lc' + str(rep) + '.npy')
                    params['save_textsummary_as'] = os.path.join(summary_dir,
                                                                 str(box_extent),
                                                                 str(n),
                                                                 str(item_size),
                                                                 'summary' + str(rep) + '.txt')

                    graph = tf.Graph()
                    with graph.as_default():
                        with tf.Session(graph=graph,
                                        config=tf.ConfigProto(allow_soft_placement=True,
                                                              log_device_placement=True)) as session:
                            kth_job += 1
                            if np.mod(kth_job, num_machines) != i_machine and num_machines != i_machine:
                                continue
                            elif np.mod(kth_job, num_machines) != 0 and num_machines == i_machine:
                                continue

                            _, _, _, _, _, _, imgs_to_acquisition = train_helpers.fftrain(session=session,
                                                                                          model_gpu_addresses=model_gpu_addresses,
                                                                                          gradient_gpu_address=gradient_gpu_address,
                                                                                          **params)

    elapsed = time.time() - t

    print(imgs_to_acquisition)
    print('ELAPSED TIME : ', str(elapsed))
