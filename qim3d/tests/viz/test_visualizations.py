import pytest

import qim3d


# unit test for plot_metrics()
def test_plot_metrics():
    metrics = {'epoch_loss': [0.3, 0.2, 0.1], 'batch_loss': [0.3, 0.2, 0.1]}

    fig = qim3d.viz.plot_metrics(metrics, figsize=(10, 10))

    assert (fig.get_figwidth(), fig.get_figheight()) == (10, 10)


def test_plot_metrics_labels():
    metrics = {'epoch_loss': [0.3, 0.2, 0.1], 'batch_loss': [0.3, 0.2, 0.1]}

    with pytest.raises(
        ValueError, match="The number of metrics doesn't match the number of labels."
    ):
        qim3d.viz.plot_metrics(metrics, labels=['a', 'b'])
