"""Visualization tools"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb

def plot_metrics(*metrics,
                 linestyle = '-',
                 batch_linestyle = 'dotted',
                 labels:list = None,
                 figsize:tuple = (16,6),
                 show = False):
    """
    Plots the metrics over epochs and batches.

    Args:
        *metrics: Variable-length argument list of dictionary containing the metrics per epochs and per batches.
        linestyle (str, optional): The style of the epoch metric line. Defaults to '-'.
        batch_linestyle (str, optional): The style of the batch metric line. Defaults to 'dotted'.
        labels (list[str], optional): Labels for the plotted lines. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size (width, height) in inches. Defaults to (16, 6).
        show (bool, optional): If True, displays the plot. Defaults to False.

    Returns:
        fig (matplotlib.figure.Figure): plot with metrics.

    Example:
        train_loss = {'epoch_loss' : [...], 'batch_loss': [...]}
        val_loss = {'epoch_loss' : [...], 'batch_loss': [...]}
        plot_metrics(train_loss,val_loss, labels=['Train','Valid.'])
    """
    if labels == None:
        labels = [None]*len(metrics)
    elif len(metrics) != len(labels):
        raise ValueError("The number of metrics doesn't match the number of labels.")

    # plotting parameters
    snb.set_style('darkgrid')
    snb.set(font_scale=1.5)
    plt.rcParams['lines.linewidth'] = 2

    fig = plt.figure(figsize = figsize)

    palette = snb.color_palette(None,len(metrics))
    
    for i,metric in enumerate(metrics):
        metric_name = list(metric.keys())[0]
        epoch_metric = metric[list(metric.keys())[0]]
        batch_metric = metric[list(metric.keys())[1]]
        
        x_axis = np.linspace(0,len(epoch_metric)-1,len(batch_metric))
        
        plt.plot(epoch_metric,linestyle = linestyle, color = palette[i], label = labels[i])
        plt.plot(x_axis, batch_metric, linestyle = batch_linestyle, color = palette[i], alpha = 0.4)
    
    if labels[0] != None:
        plt.legend()

    plt.ylabel(metric_name)
    plt.xlabel('epoch')

    # reset plotting parameters
    snb.set_style('white')

    if show:
        plt.show()
    plt.close()
    
    return fig