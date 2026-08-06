import os

import seaborn as sns
from matplotlib import pyplot as plt


def set_rc_params(notebook_dpi=None):
    """Apply the project's standard Matplotlib/Seaborn style settings.

    Sets the seaborn context to ``'paper'``, applies the ``'ticks'`` style,
    and loads the bundled ``paper.mplstyle`` stylesheet.

    Args:
        notebook_dpi: Optional figure DPI to set for interactive notebooks.
            If ``None``, the DPI is left unchanged.
    """
    sns.set_context('paper')
    sns.set_style('ticks')
    plt.style.use(os.path.join(os.path.dirname(__file__), 'paper.mplstyle'))
    if notebook_dpi is not None:
        plt.rcParams['figure.dpi'] = notebook_dpi
