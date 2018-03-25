import os
import sys
import time
import warnings

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

sys.path.append(os.path.abspath(os.path.join('..')))
from helpers import utility, pretrained_helpers
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)


def int_to_onehot(array, batch_size, num_categories):
    if len(array.shape) > 1:
        ValueError('int_to_onehot: array should be a (batch-sized) scalar')
    elif array.shape[0] != batch_size:
        ValueError('int_to_onehot: array batch size does not match model batch size')
    elif np.max(array) > num_categories:
        ValueError('int_to_onehot: array contains value exceeding num_categories')

    onehot = np.zeros((batch_size, 1, 1, num_categories))

    for i in range(batch_size):
        for val_zerobased in range(num_categories):
            if array[i] == val_zerobased:
                onehot[i, 0, 0, val_zerobased] = 1

    return onehot


def get_uninitialized_variables(session, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()

    is_not_initialized  = session.run([tf.is_variable_initialized(var) for var in list_of_variables])
    uninitialized_vars = [v for (v, f) in zip(list_of_variables, is_not_initialized) if not f]

    return uninitialized_vars


def make_parallel(model, model_gpu_addresses, keep_prob_placeholder, name, **in_tensors):
    in_splits = {}
    original_batch_size = model.get_batch_size()
    mini_batch_size = original_batch_size/len(model_gpu_addresses)
    for k, v in in_tensors.items():
        in_splits[k] = tf.split(v, len(model_gpu_addresses), axis=0)
    out_split = []
    for i in range(len(model_gpu_addresses)):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=model_gpu_addresses[i])):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                model.set_batch_size(mini_batch_size)
                out_split.append(model.run(dropout_keep_prob=keep_prob_placeholder, **{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0, name=name)


def fftrain(**kwargs):

    session = kwargs['session']
    model_gpu_addresses = kwargs['model_gpu_addresses']

    raw_input_size = kwargs['raw_input_size']
    batch_size = kwargs['batch_size']
    num_categories = kwargs['num_categories']
    learning_rate = kwargs['learning_rate']
    clip_gradient = kwargs['clip_gradient']
    dropout_keep_prob = kwargs['dropout_keep_prob']
    threshold_loss = kwargs['threshold_loss']
    num_min_train_imgs = kwargs['num_min_train_imgs']
    num_max_train_imgs = kwargs['num_max_train_imgs']
    num_val_period_imgs = kwargs['num_val_period_imgs']
    num_val_imgs = kwargs['num_val_imgs']

    model_obj = kwargs['model_obj']
    model_init_args = kwargs['model_init_args']
    model_name = kwargs['model_name']
    train_data_obj = kwargs['train_data_obj']
    train_data_init_args = kwargs['train_data_init_args']
    val_data_obj = kwargs['val_data_obj']
    val_data_init_args = kwargs['val_data_init_args']

    tb_logs_dir = kwargs['tb_logs_dir'] if 'tb_logs_dir' in kwargs else None
    extra_tensors_to_grab = kwargs['extra_tensors_to_grab'] if 'extra_tensors_to_grab' in kwargs else None
    save_ckpt_as = kwargs['save_ckpt_as'] if 'save_ckpt_as' in kwargs else None
    save_learningcurve_as = kwargs['save_learningcurve_as'] if 'save_learningcurve_as' in kwargs else None
    learningcurve_type = kwargs['learningcurve_type'] if 'learningcurve_type' in kwargs else None
    save_textsummary_as = kwargs['save_textsummary_as'] if 'save_textsummary_as' in kwargs else None

    ############### SET UP GRAPH ###############
    utility.llprint("Building Computational Graph... ")

    ##### SETUP INPUT & OUTPUT PLACEHOLDERS
    if (train_data_init_args['organization'] == 'obj') | (train_data_init_args['organization'] == 'full'):
        input_placeholder = tf.placeholder(tf.float32, [batch_size] +
                                                        raw_input_size[0:2] +
                                                        [raw_input_size[2]*train_data_init_args['num_items']], name='input_placeholder')
    else:
        input_placeholder = tf.placeholder(tf.float32, [batch_size] + raw_input_size,
                                           name='input_placeholder')
    keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
    target_output_placeholder = tf.placeholder(tf.float32, [batch_size] + [1, 1, num_categories],
                                               name='target_output_placeholder')

    ##### SETUP DATA FETCHER
    train_data_generator = train_data_obj(raw_input_size, batch_size)
    train_data_generator.initialize_vars(**train_data_init_args)
    val_data_generator = val_data_obj(raw_input_size, batch_size)
    val_data_generator.initialize_vars(**val_data_init_args)

    ##### SETUP MODEL (THIS PART IS MODEL-SPECIFIC)
    model = model_obj(name=model_name, input_size=raw_input_size, batch_size=batch_size,
                      gpu_addresses=model_gpu_addresses)
    model.initialize_vars(**model_init_args)

    ##### COMPUTE OUTPUT, LOSS and GRADIENTS
    model_output = make_parallel(model, model_gpu_addresses, keep_prob_placeholder, name='model_output', X=input_placeholder)
    model_output_argmax = tf.cast(tf.argmax(model_output, axis=-1), tf.float32)

    def ace(output, target, name=None):
        output_softmax = tf.nn.softmax(output,dim=-1)
        target_softmax = tf.nn.softmax(target,dim=-1)
        return -tf.reduce_mean(target_softmax * tf.log(output_softmax) +
                              (1 - target_softmax) * tf.log(1 - output_softmax), name=name)

    def mse(output, target, name=None):
        return tf.reduce_mean(tf.square(target - output), name=name)

    def acc(output, target, name=None):
        return tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(output, axis=3), tf.argmax(target, axis=3)), tf.float32), name=name)

    loss = ace(model_output, target_output_placeholder)
    accuracy = acc(model_output, target_output_placeholder, name='accuracy')
    average = tf.reduce_mean(model_output_argmax)
    variance = tf.reduce_mean(tf.square(model_output_argmax-average))
    if clip_gradient:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 1e+3), var)
        apply_gradients = optimizer.apply_gradients(gradients)
    else:
        apply_gradients = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    utility.llprint("Done!\n")

    ############### SET UP TENSORBOARD AND CHRONICLER ###############
    if tb_logs_dir is not None:
        if not os.path.exists(tb_logs_dir):
            os.makedirs(tb_logs_dir)
        utility.tb_flush(tb_logs_dir)
        summarize_op, summarizer, summaries = utility.tb_setup_BasicFFClassifier(tb_logs_dir,
                                                                                 tf.reduce_sum(input_placeholder,axis=3,keep_dims=True),
                                                                                 model_output,
                                                                                 target_output_placeholder,
                                                                                 gradients, loss, accuracy, average, variance,
                                                                                 session)
    # TODO: Not implemented yet
    # if history_obj is not None:
    #     history_obj.initialize_vars(session, graph, **history_init_args)
    #     history_op = history_obj.get_history_op()
    no_op = tf.no_op()

    ############### TRAIN ###############
    utility.llprint("Initializing Variables ... ")
    session.run(tf.initialize_all_variables())
    if save_ckpt_as is not None:
        saver = tf.train.Saver()
    utility.llprint("Done!\n")

    learned = False
    max_train_iters = int(num_max_train_imgs / batch_size)
    min_train_iters = int(num_min_train_imgs / batch_size)
    val_iters = int(num_val_imgs / batch_size)
    val_period_iters = int(num_val_period_imgs / batch_size)
    titer = 0
    imgs_to_acquisition = 0
    learning_curve = []
    time_per_iter = []
    while (titer < min_train_iters) | ((titer < max_train_iters) & (not learned)):
        titer += 1
        utility.llprint("\r(T) Iteration %d/%d" % (titer, max_train_iters))
        validation_period_reached = ((titer + 1) % val_period_iters is 0) and (titer > 0)

        # READ DATA FROM TRAINING DATASET
        data_obj_out = train_data_generator.single_batch()
        input_data_fetched = data_obj_out[0]
        target_output_onehot_fetched = data_obj_out[1]

        # RUN NETWORK
        t = time.time()
        session_results = session.run([apply_gradients,
                                       summarize_op if (validation_period_reached and (tb_logs_dir is not None)) else no_op],
                                      feed_dict={input_placeholder: input_data_fetched,
                                                 keep_prob_placeholder: dropout_keep_prob,
                                                 target_output_placeholder: target_output_onehot_fetched})
        time_per_iter.append(time.time() - t)
        summarize_op_fetched = session_results[-1]

        ############### ACCUMULATE LEARNING CURVE & DISPLAY PROGRESS ###############
        if validation_period_reached:
            accuracies = []
            utility.llprint("\n")
            print('Average per-iteration runtime: '+str(np.mean(time_per_iter)))
            utility.llprint("\n")
            if tb_logs_dir is not None:
                summarizer.add_summary(summarize_op_fetched, titer)
            for viter in range(val_iters):
                utility.llprint("\r(V) Iteration %d/%d" % (viter, val_iters))

                # READ DATA FROM TRAINING DATASET
                data_obj_out = val_data_generator.single_batch()
                input_data_fetched = data_obj_out[0]
                target_output_onehot_fetched = data_obj_out[1]

                # RUN NETWORK
                session_results = session.run([accuracy],
                                              feed_dict={input_placeholder: input_data_fetched,
                                                         keep_prob_placeholder: 1.,
                                                         target_output_placeholder: target_output_onehot_fetched})
                accuracies.append(session_results[0])
            mean_accuracy = np.mean(accuracies)
            utility.llprint("\n\tValidation Avg. Accuracy: %.4f\n" % (mean_accuracy))
            learning_curve.append(mean_accuracy)
            utility.llprint("\n")
            if mean_accuracy > threshold_loss:
                learned = True
                imgs_to_acquisition = titer*batch_size
    utility.llprint("\nTraining Complete.\n")

    ############### GET TRAIN DATA DISTRIBUTION ###############
    train_data_tracker = train_data_generator.get_tracker()

    ############### DETERMINE IMGS TO ACQUISITION ###############
    if imgs_to_acquisition == 0:
        imgs_to_acquisition = np.inf

    ############### GRAB EXTRA TENSORS AS REQUESTED ###############
    utility.llprint("\nGrabbing extra tensors.\n")
    tensors_grabbed_extra = []
    if extra_tensors_to_grab is not None:
        for i, tn in enumerate(extra_tensors_to_grab):
            tensors_grabbed_extra.append(tf.get_default_graph().get_tensor_by_name(tn))

    ############### DRAW AND SAVE LEARNING CURVE (VALIDATION ACC) ###############

    if save_learningcurve_as is not None:
        # SAVE HISTORY
        if not os.path.exists(os.path.split(save_learningcurve_as)[0]):
            os.makedirs(os.path.split(save_learningcurve_as)[0])
        if (learningcurve_type is None) | (learningcurve_type == 'figure'):
            utility.llprint("\nRendering learning curve.\n")
            tick_value_multiplier = 1. / 1000000
            num_ticks = 6
            plt.figure(figsize=(8, 6))  # roughly in inches
            plt.plot([a for a in learning_curve], color='blue', alpha=0.6)
            plt.xlabel('(x' + str(1. / tick_value_multiplier) + ') Images', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.xticks(np.linspace(0, max_train_iters / val_period_iters, num_ticks),
                       ((10 * np.linspace(0, max_train_iters / val_period_iters, num_ticks)).astype(int)).astype(
                           float) / 10)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.xlim((0, max_train_iters / val_period_iters))
            plt.ylim((0.45, 1.05))
            plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            plt.savefig(save_learningcurve_as)
            plt.clf()
        elif learningcurve_type == 'array':
            np.save(save_learningcurve_as, learning_curve)
        else:
            raise ValueError('learningcurve_type should be None, figure, or array.')



    ############### SAVE TEXT SUMMARY ###############
    if save_textsummary_as is not None:
        if not os.path.exists(os.path.split(save_textsummary_as)[0]):
            os.makedirs(os.path.split(save_textsummary_as)[0])
        text_summary = open(save_textsummary_as, 'w')
        text_summary.write('Final accuracy: ' + str(mean_accuracy) + '\n')
        text_summary.write('Total iterations: ' + str(titer) + '\n')
        text_summary.write('Total images: ' + str(imgs_to_acquisition))
        text_summary.close()

    ############### SAVE TRAINED NET ###############
    utility.llprint("\nSaving checkpoint.\n")
    if save_ckpt_as is not None:
        # SAVE HISTORY
        if not os.path.exists(os.path.split(save_ckpt_as)[0]):
            os.makedirs(os.path.split(save_ckpt_as)[0])
        saver.save(session, save_ckpt_as + '.ckpt')

    return input_placeholder, target_output_placeholder, keep_prob_placeholder, \
           accuracy, tensors_grabbed_extra, \
           train_data_tracker, imgs_to_acquisition






def fftrain_pre_initialized(**kwargs):

    session = kwargs['session']
    model_gpu_addresses = kwargs['model_gpu_addresses']

    batch_size = kwargs['batch_size']
    learning_rate = kwargs['learning_rate']
    clip_gradient = kwargs['clip_gradient']
    dropout_keep_prob = kwargs['dropout_keep_prob']
    threshold_loss = kwargs['threshold_loss']
    num_min_train_imgs = kwargs['num_min_train_imgs']
    num_max_train_imgs = kwargs['num_max_train_imgs']
    num_val_period_imgs = kwargs['num_val_period_imgs']
    num_val_imgs = kwargs['num_val_imgs']

    model_obj = kwargs['model_obj']
    input_placeholder = kwargs['input_placeholder']
    target_output_placeholder = kwargs['target_output_placeholder']
    keep_prob_placeholder = kwargs['keep_prob_placeholder']
    train_data_obj = kwargs['train_data_obj']
    val_data_obj = kwargs['val_data_obj']

    tb_logs_dir = kwargs['tb_logs_dir'] if 'tb_logs_dir' in kwargs else None
    extra_tensors_to_grab = kwargs['extra_tensors_to_grab'] if 'extra_tensors_to_grab' in kwargs else None
    save_ckpt_as = kwargs['save_ckpt_as'] if 'save_ckpt_as' in kwargs else None
    save_learningcurve_as = kwargs['save_learningcurve_as'] if 'save_learningcurve_as' in kwargs else None
    learningcurve_type = kwargs['learningcurve_type'] if 'learningcurve_type' in kwargs else None
    save_textsummary_as = kwargs['save_textsummary_as'] if 'save_textsummary_as' in kwargs else None

    ############### SET UP GRAPH ###############
    utility.llprint("Building Computational Graph... ")


    ##### COMPUTE OUTPUT, LOSS and GRADIENTS
    model_output = make_parallel(model_obj, model_gpu_addresses, keep_prob_placeholder, name='model_output', X=input_placeholder)

    def mse(output, target, name=None):
        return tf.reduce_mean(tf.square(target - output), name=name)

    def acc(output, target, name=None):
        return tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(output, axis=3), tf.argmax(target, axis=3)), tf.float32), name=name)

    def bias(output, name=None):
        return tf.reduce_mean(tf.cast(tf.argmax(output, axis=3), tf.float32), name=name)

    loss = mse(model_output, target_output_placeholder)
    accuracy = acc(model_output, target_output_placeholder, name='accuracy')
    positive_bias = bias(model_output)
    if clip_gradient:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 1e+3), var)
        apply_gradients = optimizer.apply_gradients(gradients)
    else:
        apply_gradients = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    utility.llprint("Done!\n")

    ############### SET UP TENSORBOARD AND CHRONICLER ###############
    if tb_logs_dir is not None:
        if not os.path.exists(tb_logs_dir):
            os.makedirs(tb_logs_dir)
        utility.tb_flush(tb_logs_dir)
        summarize_op, summarizer, summaries = utility.tb_setup_BasicFFClassifier(tb_logs_dir,
                                                                                 input_placeholder,
                                                                                 model_output,
                                                                                 target_output_placeholder,
                                                                                 gradients, loss, accuracy,
                                                                                 positive_bias, session)
    # TODO: Not implemented yet
    # if history_obj is not None:
    #     history_obj.initialize_vars(session, graph, **history_init_args)
    #     history_op = history_obj.get_history_op()
    no_op = tf.no_op()


    ############### INITIALIZE AND TRAIN ###############
    if save_ckpt_as is not None:
        saver = tf.train.Saver()
    utility.llprint("Initializing Variables ... ")
    uninitialized_variables = get_uninitialized_variables(session=session, list_of_variables=tf.all_variables())
    session.run(tf.initialize_variables(uninitialized_variables))
    utility.llprint("Done!\n")

    learned = False
    max_train_iters = int(num_max_train_imgs / batch_size)
    min_train_iters = int(num_min_train_imgs / batch_size)
    val_iters = int(num_val_imgs / batch_size)
    val_period_iters = int(num_val_period_imgs / batch_size)
    titer = 0
    imgs_to_acquisition = 0
    learning_curve = []
    time_per_iter = []
    while (titer < min_train_iters) | ((titer < max_train_iters) & (not learned)):
        titer += 1
        utility.llprint("\r(T) Iteration %d/%d" % (titer, max_train_iters))
        validation_period_reached = ((titer + 1) % val_period_iters is 0) and (titer > 0)

        # READ DATA FROM TRAINING DATASET
        data_obj_out = train_data_obj.single_batch()
        input_data_fetched = data_obj_out[0]
        target_output_onehot_fetched = data_obj_out[1]

        # RUN NETWORK
        t = time.time()
        session_results = session.run([apply_gradients,
                                       summarize_op if (validation_period_reached and (tb_logs_dir is not None)) else no_op],
                                      feed_dict={input_placeholder: input_data_fetched,
                                                 keep_prob_placeholder: dropout_keep_prob,
                                                 target_output_placeholder: target_output_onehot_fetched})
        time_per_iter.append(time.time() - t)
        summarize_op_fetched = session_results[-1]

        ############### ACCUMULATE LEARNING CURVE & DISPLAY PROGRESS ###############
        if validation_period_reached:
            accuracies = []
            utility.llprint("\n")
            print('Average per-iteration runtime: '+str(np.mean(time_per_iter)))
            utility.llprint("\n")
            if tb_logs_dir is not None:
                summarizer.add_summary(summarize_op_fetched, titer)
            for viter in range(val_iters):
                utility.llprint("\r(V) Iteration %d/%d" % (viter, val_iters))

                # READ DATA FROM TRAINING DATASET
                data_obj_out = val_data_obj.single_batch()
                input_data_fetched = data_obj_out[0]
                target_output_onehot_fetched = data_obj_out[1]

                # RUN NETWORK
                session_results = session.run([accuracy],
                                              feed_dict={input_placeholder: input_data_fetched,
                                                         keep_prob_placeholder: 1.,
                                                         target_output_placeholder: target_output_onehot_fetched})
                accuracies.append(session_results[0])
            mean_accuracy = np.mean(accuracies)
            utility.llprint("\n\tValidation Avg. Accuracy: %.4f\n" % (mean_accuracy))
            learning_curve.append(mean_accuracy)
            utility.llprint("\n")
            if mean_accuracy > threshold_loss:
                learned = True
                imgs_to_acquisition = titer*batch_size
    utility.llprint("\nTraining Complete.\n")

    ############### GET TRAIN DATA DISTRIBUTION ###############
    train_data_tracker = train_data_obj.get_tracker()

    ############### DETERMINE IMGS TO ACQUISITION ###############
    if imgs_to_acquisition == 0:
        imgs_to_acquisition = np.inf

    ############### GRAB EXTRA TENSORS AS REQUESTED ###############
    utility.llprint("\nGrabbing extra tensors.\n")
    tensors_grabbed_extra = []
    if extra_tensors_to_grab is not None:
        for i, tn in enumerate(extra_tensors_to_grab):
            tensors_grabbed_extra.append(tf.get_default_graph().get_tensor_by_name(tn))

    ############### DRAW AND SAVE LEARNING CURVE (VALIDATION ACC) ###############

    if save_learningcurve_as is not None:
        # SAVE HISTORY
        if not os.path.exists(os.path.split(save_learningcurve_as)[0]):
            os.makedirs(os.path.split(save_learningcurve_as)[0])
        if (learningcurve_type is None) | (learningcurve_type == 'figure'):
            utility.llprint("\nRendering learning curve.\n")
            tick_value_multiplier = 1. / 1000000
            num_ticks = 6
            plt.figure(figsize=(8, 6))  # roughly in inches
            plt.plot([a for a in learning_curve], color='blue', alpha=0.6)
            plt.xlabel('(x' + str(1. / tick_value_multiplier) + ') Images', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.xticks(np.linspace(0, max_train_iters / val_period_iters, num_ticks),
                       ((10 * np.linspace(0, max_train_iters / val_period_iters, num_ticks)).astype(int)).astype(
                           float) / 10)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.xlim((0, max_train_iters / val_period_iters))
            plt.ylim((0.45, 1.05))
            plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            plt.savefig(save_learningcurve_as)
            plt.clf()
        elif learningcurve_type == 'array':
            np.save(save_learningcurve_as, learning_curve)
        else:
            raise ValueError('learningcurve_type should be None, figure, or array.')



    ############### SAVE TEXT SUMMARY ###############
    if save_textsummary_as is not None:
        text_summary = open(save_textsummary_as, 'w')
        text_summary.write('Final accuracy: ' + str(mean_accuracy) + '\n')
        text_summary.write('Total iterations: ' + str(titer) + '\n')
        text_summary.write('Total images: ' + str(imgs_to_acquisition))
        text_summary.close()

    ############### SAVE TRAINED NET ###############
    utility.llprint("\nSaving checkpoint.\n")
    if save_ckpt_as is not None:
        # SAVE HISTORY
        if not os.path.exists(os.path.split(save_ckpt_as)[0]):
            os.makedirs(os.path.split(save_ckpt_as)[0])
        saver.save(session, save_ckpt_as + '.ckpt')

    return input_placeholder, target_output_placeholder, keep_prob_placeholder, \
           accuracy, tensors_grabbed_extra, \
           train_data_tracker, imgs_to_acquisition


