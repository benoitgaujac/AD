import os

import numpy as np
from math import pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors

import utils

import pdb


def plot_train(opts, trloss, teloss, scores, heatmap, Phi, D, exp_dir, filename):

    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3

        img1    -   obj/training curves
        img2    -   heatmap of score fct
        img3    -   Phi
        img4    -   nominal/anomalus scores
        img5    -   Diag values

    """


    ### Creating a pyplot fig
    dpi = 100
    height_pic = 1000
    width_pic = 1000
    fig_width = height_pic / float(dpi)
    fig_height = 5 * width_pic / float(dpi)
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(fig_height, fig_width))

    ### The loss curves
    # test
    array_loss = np.array(teloss).reshape((-1,4))
    obj = array_loss[:,0]
    # scr = np.abs(array_loss[:,1])
    scr = array_loss[:,1]
    dreg = opts['lmbda']*array_loss[:,2]
    wreg = opts['gamma']*array_loss[:,3]
    for y, (color, label) in zip([obj, scr, dreg, wreg],
                                [('black', 'teObj'),
                                # ('red', '|teScore|'),
                                ('red', 'teScore'),
                                ('yellow', r'$\lambda$teDreg'),
                                ('cyan', r'$\gamma$teWreg')]):
        total_num = len(y)
        # x_step = max(int(total_num / 200), 1)
        # x = np.arange(1, len(y) + 1, x_step)
        # y = np.log(y[::x_step])
        x = np.arange(1, total_num + 1)
        # y = np.log(y)
        axes[0].plot(x, y, linewidth=2, color=color, label=label)
    # train
    array_loss = np.array(trloss).reshape((-1,4))
    obj = array_loss[:,0]
    # scr = np.abs(array_loss[:,1])
    scr = array_loss[:,1]
    dreg = opts['lmbda']*array_loss[:,2]
    wreg = opts['gamma']*array_loss[:,3]
    for y, (color, label) in zip([obj, scr, dreg, wreg],
                                [('black', 'trObj'),
                                # ('red', '|trScore|'),
                                ('red', 'trScore'),
                                ('yellow', r'$\lambda$trDreg'),
                                ('cyan', r'$\gamma$trWreg')]):
        total_num = len(y)
        # x_step = max(int(total_num / 200), 1)
        # x = np.arange(1, len(y) + 1, x_step)
        # y = np.log(y[::x_step])
        x = np.arange(1, total_num + 1)
        # y = np.log(y)
        axes[0].plot(x, y, linewidth=2, color=color, linestyle='--', label=label)
    axes[0].grid(axis='y')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order, nlabels = [], len(labels)
    # for i in range(nlabels):
    #     if i%2==0:
    #         order.append(int((nlabels+i)/2))
    #     else:
    #         order.append(int(i/2))
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right')
    axes[0].legend(loc='best', ncol=2)
    axes[0].set_title('Log Losses')

    ### The score heatmap
    axes[1].imshow(heatmap, cmap='hot_r', interpolation='nearest')
    axes[1].set_title('Score heatmap')
    fig.colorbar(axes[1].imshow(heatmap, cmap='hot_r', interpolation='nearest'), ax=axes[1], shrink=0.75, fraction=0.08) #, format=format[dataset])

    ### Phi
    total_num = len(Phi)
    x = np.arange(1, total_num + 1)
    axes[2].plot(x, Phi, linewidth=2, color='red', label=r'$\phi$')
    axes[2].plot(x, opts['theta']*np.ones(total_num), linewidth=2, linestyle='--',
                            color='blue', label=r'$\theta_x$')
    axes[2].grid(axis='y')
    axes[2].set_yticks(np.linspace(0.,pi,7))
    axes[2].set_yticklabels(['0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$',
                            r'$\frac{\pi}{2}$', r'$\frac{2\pi}{3}$',
                            r'$\frac{5\pi}{6}$', r'$\pi$'])
    axes[2].legend(loc='best')
    axes[2].set_title(r'$\Phi$')

    ### The anomalous losses
    array_loss = np.abs(np.array(scores).reshape((-1,2)))
    for y, (color, label) in zip([np.abs(array_loss[:,0]),
                                np.abs(array_loss[:,1])],
                                [('red', 'nominal score'),
                                ('blue', '|anomalous score|')]):
        total_num = len(y)
        # x_step = max(int(total_num / 200), 1)
        # x = np.arange(1, len(y) + 1, x_step)
        # y = np.log(y)
        x = np.arange(1, total_num + 1)
        axes[3].plot(x, y, linewidth=1, color=color, label=label)
    axes[3].legend(loc='upper right')
    axes[3].set_title('Log Scores')

    ### D
    array_D = np.array(D).reshape((-1,2))
    axes[4].plot(x, array_D[:,0], linewidth=2, color='red', label=r'$\alpha$')
    axes[4].plot(x, array_D[:,1], linewidth=2, color='blue', label=r'$\beta$')
    axes[4].grid(axis='y')
    axes[4].legend(loc='best')
    axes[4].set_title('Model params')

    ### Saving plots
    # Plot
    plots_dir = 'train_plots'
    save_path = os.path.join(exp_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)
