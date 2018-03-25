import os, sys
sys.path.append(os.path.abspath(os.path.join('..','..')))

def get_params():
    params = {}
    params['raw_input_size'] = [180, 180, 1]
    params['batch_size'] = 50
    params['num_categories'] = 2
    params['num_min_train_imgs'] = 10000000
    params['num_max_train_imgs'] = 10000000
    params['num_val_period_imgs'] = 100000
    params['num_val_imgs'] = 50000
    params['threshold_loss'] = 1.1

    params['learning_rate'] = 1e-4
    params['clip_gradient'] = True
    params['dropout_keep_prob'] = 0.8

    from instances import processor_instances

    params['model_obj'] = processor_instances.PSVRT_siamesenet
    params['model_name'] = 'model'
    params['model_init_args'] = {'num_categories': 2,
                                 'num_items': 2,
                                 'organization':'obj',
                                 'num_CP_layers': 4,
                                 'num_CP_features': 16, # 4
                                 'num_FC_layers': 4,
                                 'num_FC_features': 256,
                                 'initial_conv_rf_size': [4, 4],
                                 'interm_conv_rf_size': [2, 2],
                                 'pool_rf_size': [2, 2],
                                 'stride_size': [2, 2],
                                 'activation_type': 'relu',
                                 'global_pool': False,
                                 'trainable': True,
                                 'hamstring_factor': 0.5}
    from instances import psvrt_sdsr
    params['train_data_obj'] = psvrt_sdsr.psvrt_sdsr
    params['train_data_init_args'] = {'problem_type':'SDSR',
                                      'item_size': [4,4],
                                      'box_extent': [90,90],
                                      'num_items': params['model_init_args']['num_items'],
                                      'num_item_pixel_values': 1,
                                      'organization': params['model_init_args']['organization'],
                                      'display': False}

    params['val_data_obj'] = psvrt_sdsr.psvrt_sdsr
    params['val_data_init_args'] = params['train_data_init_args'].copy()

    params['save_learningcurve_as'] = '/home/jk/PSVRT_test_result'
    params['learningcurve_type'] = 'array'
    params['save_textsummary_as'] = '/home/jk/PSVRT_test_result'
    params['tb_logs_dir'] = None #'/home/jk/PSVRT_test_result_tb'

    return params
