import os
import pickle

import pandas as pd
from pathlib import Path


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


def export_trajectory_data(config, stiffness, trajectories):
    """Plots the optimal trajectories obtained from optimization

    Parameters
    ----------
    config : yaml
        The yaml configuration rate.
    save_plot : boolean
        To save the plot or not

    Returns
    -------
    None

    """

    read_path = Path(__file__).parents[2] / config['save_path']
    fname = [str(f) for f in read_path.iterdir() if f.suffix == '.pkl']
    fname.sort()
    columns = ['time', 'bd', 'bv', 'hd', 'hv', 'md', 'mv']
    aggregate_df = pd.DataFrame(columns=trajectories)
    for i, item in enumerate(fname):
        data = read_model_log(item)
        temp = data['optimal_values'][columns].copy()
        temp['stiffness'] = config['stiffness'][i]
        aggregate_df = pd.concat([aggregate_df, temp],
                                 ignore_index=True,
                                 sort=False)

    aggregate_df.dropna(how='any', inplace=True)
    df = aggregate_df[aggregate_df['stiffness'] == stiffness]
    df = df[trajectories]

    save_path = Path(__file__).parents[2] / config['trajectory_save_path']
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    df.to_csv(str(save_path) + '/' + stiffness + '.csv', index=False)

    return None
