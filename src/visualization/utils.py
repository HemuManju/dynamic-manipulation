import pickle
import matplotlib.pyplot as plt


def read_dataframe(path):
    """Save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save


    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def read_model_log(read_path):
    """Read the model log.

    Parameters
    ----------
    read_path : str
        Path to read data from.

    Returns
    -------
    dict
        model log.

    """
    with open(read_path, 'rb') as handle:
        data = pickle.load(handle)

    return data


def figure_asthetics(ax, subplot):
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """

    ax.set_axisbelow(True)
    # Hide the top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Increase the ticks width
    ax.xaxis.set_tick_params(direction='out', width=1.5)
    ax.yaxis.set_tick_params(direction='out', width=1.5)

    # Increase the left and bottom spines width to match with ticks
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Increase the x and y ticks
    if not subplot:
        xtickslocs = ax.get_xticks().tolist()
        ax.set_xticks(xtickslocs)
        ytickslocs = ax.get_yticks().tolist()
        ax.set_yticks(ytickslocs)

    # Grid
    ax.grid(True)

    return None


def plot_settings():
    """Change the asthetics of the given figure (operators in place).

    Parameters
    ----------
    ax : matplotlib ax object

    """

    # plt.rcParams.update({'font.family'}: "Arial")
    plt.rcParams.update({'font.size': 12})

    return None


def annotate_significance(x1, x2, y, p):
    """Add significance annotations over a plot.

    Parameters
    ----------
    x1 : float
        x position of factor 1.
    x2 : float
        x position of factor 2.
    y : float
        Outcome variable.

    Returns
    -------
    None

    """
    h = 0.01
    star = []
    if p < 0.001:
        star = "***"
    elif p < 0.01:
        star = "**"
    if star:
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
        plt.text((x1 + x2) * .5,
                 y - h / 2,
                 star,
                 ha='center',
                 va='bottom',
                 color='k',
                 size=20)

    return None


def add_hatches(ax):
    """Add hatches to the axes.

    Parameters
    ----------
    ax : matplotlib axes

    Returns
    -------
    None
        Description of returned object.

    """

    hatches = ['\\', '\\', 'x', 'x', '+', '+']
    for i, thisbar in enumerate(ax.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(3 * hatches[i])

    return None


def fix_microseconds(timestamp):
    parts = timestamp.split(':')

    return ':'.join(parts[:-1] + ['{:06d}'.format(int(parts[-1]))])
