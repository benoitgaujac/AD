import os

import numpy as np
from math import pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FormatStrFormatter

import utils

import pdb


def plot_train(opts, trloss, teloss, scores, heatmap, inputs, transformed, Phi, D, exp_dir, filename):

    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3

        img1    -   obj/training curves
        img2    -   nominal/anomalus scores
        img3    -   Diag values
        img4    -   Phi
        img5    -   heatmap of score fct
        img6    -   transformed inputs if non affine

    """


    ### Creating a pyplot fig
    dpi = 100
    height_pic = 1000
    width_pic = 1000
    fig_width = 3 *height_pic / float(dpi)
    fig_height = 2 * width_pic / float(dpi)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(fig_height, fig_width))

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
        axes[0,0].plot(x, y, linewidth=2, color=color, label=label)
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
        axes[0,0].plot(x, y, linewidth=2, color=color, linestyle='--', label=label)
    axes[0,0].grid(axis='y')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order, nlabels = [], len(labels)
    # for i in range(nlabels):
    #     if i%2==0:
    #         order.append(int((nlabels+i)/2))
    #     else:
    #         order.append(int(i/2))
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right')
    axes[0,0].legend(loc='best', ncol=2)
    axes[0,0].set_title('Log Losses')

    ### The anomalous losses
    array_loss = np.abs(np.array(scores).reshape((-1,2)))
    for y, (color, label) in zip([array_loss[:,0],
                                array_loss[:,1]],
                                [('red', 'nominal score'),
                                ('blue', '|anomalous score|')]):
        total_num = len(y)
        # x_step = max(int(total_num / 200), 1)
        # x = np.arange(1, len(y) + 1, x_step)
        # y = np.log(y)
        x = np.arange(1, total_num + 1)
        axes[0,1].plot(x, y, linewidth=1, color=color, label=label)
    axes[0,1].legend(loc='upper right')
    axes[0,1].set_title('Scores')

    ### Phi
    total_num = len(Phi)
    x = np.arange(1, total_num + 1)
    axes[1,0].plot(x, Phi, linewidth=2, color='red', label=r'$\phi$')
    axes[1,0].plot(x, opts['theta']*np.ones(total_num), linewidth=2, linestyle='--',
                            color='blue', label=r'$\theta_x$')
    axes[1,0].plot(x, np.abs(opts['theta']*np.ones(total_num)-Phi), linewidth=2, linestyle=':',
                            color='green', label=r'$|\phi-\theta_x|$')
    axes[1,0].grid(axis='y')
    axes[1,0].set_yticks(np.linspace(0.,pi,7))
    axes[1,0].set_yticklabels(['0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{3}$',
                            r'$\frac{\pi}{2}$', r'$\frac{2\pi}{3}$',
                            r'$\frac{5\pi}{6}$', r'$\pi$'])
    axes[1,0].legend(loc='best')
    axes[1,0].set_title(r'$\Phi$')

    ### D
    array_D = np.array(D).reshape((-1,2))
    axes[1,1].plot(x, array_D[:,0], linewidth=2, color='red', label=r'$\alpha$')
    axes[1,1].plot(x, array_D[:,1], linewidth=2, color='blue', label=r'$\beta$')
    axes[1,1].grid(axis='y')
    axes[1,1].legend(loc='best')
    axes[1,1].set_title('Model params')

    ### The transformed inputs if needed
    m, M = 100, -100
    if opts['flow']!='identity':
        img = zip([inputs, transformed],
                    [('red', 10, 'inputs', .8),
                    (('blue', 12, 'transformed', 1.))])
        for xy, style in img:
            x = xy[:,0]
            y = xy[:,1]
            axes[2,1].scatter(x, y, c=style[0], s=style[1], label=style[2], alpha=style[3])
            # m = min(m, np.amin(x), np.amin(y))
            # M = max(M, np.amax(x), np.amax(y))
            mx, Mx = min(m, np.amin(x)), max(M, np.amax(x))
            my, My = min(m, np.amin(y)), max(M, np.amax(y))
    else:
        axes[2,1].scatter(inputs[:,0], inputs[:,1], c='red', s=10, label='inputs')
        # m = min(m, np.amin(inputs[:,0]), np.amin(inputs[:,1]))
        # M = max(M, np.amax(inputs[:,0]), np.amax(inputs[:,1]))
        mx, Mx = min(m, np.amin(x)), max(M, np.amax(x))
        my, My = min(m, np.amin(y)), max(M, np.amax(y))
    lim = max(abs(m),abs(M))
    yticks = np.linspace(my,My,5)
    axes[2,1].set_yticks(yticks)
    axes[2,1].set_yticklabels(yticks)
    axes[2,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    xticks = np.linspace(-mx,Mx,5)
    axes[2,1].set_xticks(xticks)
    axes[2,1].set_xticklabels(xticks)
    # axes[2,1].set_xticklabels(np.linspace(-lim,lim,5))
    axes[2,1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[2,1].legend(loc='best')
    axes[2,1].set_title('Flow transformation')

    ### The score heatmap
    axes[2,0].imshow(np.log(heatmap), cmap='hot_r', interpolation='nearest')
    ticks = np.linspace(0,heatmap.shape[0]-1,5)
    # pdb.set_trace()
    # axes[2,0].set_yticks(ticks)
    # axes[2,0].set_yticklabels(np.linspace(-1,1,5)[::-1])
    # axes[2,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axes[2,0].set_xticks(ticks)
    # axes[2,0].set_xticklabels(np.linspace(-1,1,5))
    # axes[2,0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[2,0].set_title('logScore heatmap')
    fig.colorbar(axes[2,0].imshow(heatmap, cmap='hot_r', interpolation='nearest'), ax=axes[2,0], shrink=0.75, fraction=0.08) #, format=format[dataset])

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
