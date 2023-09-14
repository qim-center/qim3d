"""Visualization tools"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb

def plot_metrics(metric, color = 'blue', linestyle = '-', batch_linestyle = 'dotted', label = None):
    """
    Plots the metrics over epochs and batches.

    Args:
        metric (dict): A dictionary containing the metrics per epochs and per batches.
        color (str, optional): The color of the plotted lines. Defaults to 'blue'.
        linestyle (str, optional): The style of the epoch metric line. Defaults to '-'.
        batch_linestyle (str, optional): The style of the batch metric line. Defaults to 'dotted'.
        label (str, optional): The label for the epoch metric line. Defaults to None.

    Returns:
        None

    Example:
        train_loss = {'epoch_loss' : [...], 'batch_loss': [...]}
        plot_metrics(train_loss, color = 'red', label='Train')
    
    """
    # plotting parameters
    snb.set_style('darkgrid')
    snb.set(font_scale=1.5)
    plt.rcParams['lines.linewidth'] = 2

    metric_name = list(metric.keys())[0]
    epoch_metric = metric[list(metric.keys())[0]]
    batch_metric = metric[list(metric.keys())[1]]
    
    x_axis = np.linspace(0,len(epoch_metric)-1,len(batch_metric))
    
    plt.plot(epoch_metric,linestyle = linestyle, color = color,label = label)
    plt.plot(x_axis, batch_metric, linestyle = batch_linestyle, color = color, alpha = 0.4)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend()

    # reset plotting parameters
    snb.set_style('white')