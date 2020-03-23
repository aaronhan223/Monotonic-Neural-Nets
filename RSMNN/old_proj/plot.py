import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import numpy as np
matplotlib.use('Agg')

keys = ['linear', 'w/o smooth', 'smooth', 'gt', 'nn']
colors = {
    'linear': 'r',
    'w/o smooth': 'b',
    'smooth': 'g',
    'gt': 'y',
    'nn': 'k'
}

legends = {
    'linear': 'Linear Offset',
    'w/o smooth': 'Without Smoothing',
    'smooth': 'After Smoothing',
    'gt': 'Ground Truth',
    'nn': 'Neural Networks'
}
markers = {
    'linear': 'x',
    'w/o smooth': '+',
    'smooth': '^',
    'gt': '+',
    'nn': '+'
}

linestyles = {
    'linear': '-',
    'w/o smooth': '--',
    'smooth': ':',
    'gt': '-',
    'nn': '-'
}


def plot_fig(X, Y):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6), facecolor='blue')
    plt.rc("axes.spines", top=True, right=True)

    for (idx, label) in enumerate(keys):
        plt.plot(X, Y[label], label=legends[label],
                 color=colors[label], lw=6,
                 ls=linestyles[label],
                 markersize=2,
                 marker=markers[label])

    # plt.xlim([-1, 21])
    # plt.xticks(np.arange(0, 20, 4), fontsize=25)
    # plt.ylim([-1, 10])
    # plt.yticks(np.arange(0, 10, 2), fontsize=25)
    plt.xlabel('X', fontsize=25)
    plt.legend(('linear', 'w/o smooth', 'smooth', 'ground truth', 'NN'), loc='upper left', fontsize='small', fancybox=True)
    plt.savefig('./smooth_plot.pdf', bbox_inches='tight')

    plt.clf()

