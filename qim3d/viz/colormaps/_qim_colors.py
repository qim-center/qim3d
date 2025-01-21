from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


qim = LinearSegmentedColormap.from_list(
    "qim",
    [
        (0.6, 0.0, 0.0),  # 990000
        (1.0, 0.6, 0.0),  # ff9900
    ],
)
"""
Defines colormap in QIM logo colors. Can be accessed as module attribute or easily by ```cmap = 'qim'```

Example:
    ```python

    import qim3d

    display(qim3d.viz.colormaps.qim)
    ```
    ![colormap objects](../../assets/screenshots/viz-colormaps-qim.png)
"""
colormaps.register(qim)
