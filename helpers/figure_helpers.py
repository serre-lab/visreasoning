import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')

def learningcurve_from_list(learning_curve, save_as, max_train_iters, val_period_iters,
                            tick_value_multiplier=1./1000000, num_ticks=6):

    print("Rendering learning curve.")
    plt.figure(figsize=(8, 6))  # roughly in inches
    plt.plot([a for a in learning_curve], color='blue', alpha=0.6)
    plt.xlabel('(x' + str(1. / tick_value_multiplier) + ') Images', fontsize=16)
    plt.ylabel('Training accuracy', fontsize=16)
    plt.xticks(np.linspace(0, max_train_iters / val_period_iters, num_ticks),
               ((np.linspace(0, max_train_iters / val_period_iters, num_ticks)).astype(int)).astype(
                   float) / 10)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xlim((0, max_train_iters / val_period_iters))
    plt.ylim((0.45, 1.05))
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.savefig(save_as)
    plt.clf()
    print("Done.")