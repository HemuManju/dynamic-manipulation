import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .utils import *


def plot_optimal_trajectories(config):
    """Plots the optimal trajectories obtained from optimization

    Parameters
    ----------
    config : yaml
        The yaml configuration rate.

    Returns
    -------
    None

    """

    read_path = Path(__file__).parents[2] / config['save_path']
    fname = [str(f) for f in read_path.iterdir() if f.suffix == '.pkl']
    fname.sort(reverse=False)
    trajectories = ['time', 'bd', 'bv', 'ba', 'hd', 'hv', 'md', 'mv']
    aggregate_df = pd.DataFrame(columns=trajectories)
    for i, item in enumerate(fname):
        data = read_model_log(item)
        temp = data['optimal_values'][trajectories].copy()
        temp['stiffness'] = config['stiffness'][i]
        aggregate_df = pd.concat([aggregate_df, temp],
                                 ignore_index=True,
                                 sort=False)

    aggregate_df.dropna(how='any', inplace=True)

    labels = {
        'bd': 'base displacement',
        'bv': 'base velocity',
        'ba': 'base acceleration',
        'hd': 'hammer displacement',
        'hv': 'hammer velocity',
        'md': 'magnet displacement',
        'mv': 'magnet velocity'
    }
    for key, value in labels.items():

        fig, ax = plt.subplots()
        df_low = aggregate_df[aggregate_df['stiffness'] == 'low_stiffness']
        ax.plot(df_low['time'].values,
                df_low[key].values,
                label='low stiffness')

        df_high = aggregate_df[aggregate_df['stiffness'] == 'high_stiffness']
        ax.plot(df_high['time'].values,
                df_high[key].values,
                label='high stiffness')

        df_variable = aggregate_df[aggregate_df['stiffness'] ==
                                   'variable_stiffness']
        ax.plot(df_variable['time'].values,
                df_variable[key].values,
                label='variable stiffness')
        # Asthetics
        figure_asthetics(ax)
        ax.set_xlabel('Time')
        ax.set_ylabel(value)
        plt.legend()
        plt.show()

    return None