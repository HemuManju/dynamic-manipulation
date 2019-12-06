import os
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from .utils import (read_model_log, figure_asthetics, plot_settings,
                    fix_microseconds)


def get_plot_data(df, feature):
    """Extracts the features from the dataframe

    Parameters
    ----------
    df : dataframe
        A pandas dataframe with all the basic features as columns
    feature : list
        A list constainig the expresssion we want to extract
        e.g. ['feature_1 + feature_2']

    Returns
    -------
    array
        An array of the feature constructedf from the dataframe
    """

    df.eval('result=' + feature, inplace=True)
    data = df['result'].values

    return data


def get_dataframe_dict(config):
    """Get the dataframe with all the columns.

    Parameters
    ----------
    config : yaml
        The configuration file.

    Returns
    -------
    dataframe
        The dataframe with the optimized trajectories.
    """

    # Prepare the data
    read_path = Path(__file__).parents[2] / config['save_path']
    fname = [str(f) for f in read_path.iterdir() if f.suffix == '.pkl']
    fname.sort()
    trajectories = ['time', 'bd', 'ba', 'bv', 'hd', 'hv', 'md', 'mv']
    aggregate_df = pd.DataFrame(columns=trajectories)
    for i, item in enumerate(fname):
        data = read_model_log(item)
        temp = data['optimal_values'][trajectories].copy()
        temp['stiffness'] = config['stiffness'][i]
        aggregate_df = pd.concat([aggregate_df, temp],
                                 ignore_index=True,
                                 sort=False)

    aggregate_df.dropna(how='any', inplace=True)

    return aggregate_df


def plot_optimal_trajectories(config, features, save_plot):
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

    aggregate_df = get_dataframe_dict(config)

    # Global plot settings
    plot_settings()
    for key, value in features.items():

        fig, ax = plt.subplots()
        df_low = aggregate_df[aggregate_df['stiffness'] == 'low_stiffness']
        ax.plot(df_low['time'].values,
                get_plot_data(df_low, key),
                label='low stiffness',
                color='#469B55')

        df_high = aggregate_df[aggregate_df['stiffness'] == 'high_stiffness']
        ax.plot(df_high['time'].values,
                get_plot_data(df_high, key),
                label='high stiffness',
                color='#3C5CA0')

        df_variable = aggregate_df[aggregate_df['stiffness'] ==
                                   'variable_stiffness']
        ax.plot(df_variable['time'].values,
                get_plot_data(df_variable, key),
                label='variable stiffness',
                color='#B53941')
        # Asthetics

        figure_asthetics(ax, subplot=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(value)
        plt.legend()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if save_plot:
            name = value.lower().replace(" ", "-")
            name = name.replace("/", " ")
            path = str(Path(__file__).parents[2] /
                       config['figure_save_path']) + '/' + name + '.pdf'
            # If the folder does not exist create it
            if not os.path.isdir(config['figure_save_path']):
                os.mkdir(config['figure_save_path'])
            plt.savefig(path, bbox_inches='tight')

    return None


def plot_magnet_hammer_path(config, save_plot):

    # Get the data
    aggregate_df = get_dataframe_dict(config)

    stiffness = ['low_stiffness', 'high_stiffness', 'variable_stiffness']
    colors = ['#469B55', '#3C5CA0', '#B53941']
    features = ['md', 'hd', '-md']

    # Global plot settings
    plot_settings()
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))

    for i, item in enumerate(stiffness):

        df = aggregate_df[aggregate_df['stiffness'] == item]
        for feature in features:
            if feature == 'hd':
                linestyle = '-'
            else:
                linestyle = '--'
            ax[i].plot(df['time'].values,
                       get_plot_data(df, feature),
                       color=colors[i],
                       linestyle=linestyle)

        # Asthetics
        figure_asthetics(ax[i], subplot=True)
        ax[i].set_xlim(0, 0.8)
        ax[i].set_ylim(-0.07, 0.07)
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('Displacement (m)')
        ax[i].set_title(item.lower().replace("_", " "))
        ax[i].legend(['magnet position', 'hammer position'], loc="upper left")
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if save_plot:
            name = item.lower().replace(" ",
                                        "-") + "_hammer_magnet_displacement"
            path = str(Path(__file__).parents[2] /
                       config['figure_save_path']) + '/' + name + '.pdf'
            # If the folder does not exist create it
            if not os.path.isdir(config['figure_save_path']):
                os.mkdir(config['figure_save_path'])
            plt.savefig(path, bbox_inches='tight')

    return None


def plot_magnet_hammer(config, save_plot):

    stiffness = ['high_stiffness', 'low_stiffness', 'variable_stiffness']
    features = ['Magnet_position', 'handle_disp', '-Magnet_position']

    # Global plot settings
    plot_settings()
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))

    for i, item in enumerate(stiffness):
        for feature in features:
            if feature == 'handle_disp':
                linestyle = '-'
                color = 'r'
                path = config['output_data_path']
            else:
                linestyle = '--'
                color = 'b'
                path = config['input_data_path']
            df = pd.read_csv(path + item + '.csv')
            # Covert time to seconds
            if feature == 'handle_disp':
                time_data = df['Time'].values.tolist()
                time_list = [
                    datetime.strptime(fix_microseconds(item), '%H:%M:%S:%f')
                    for item in time_data
                ]
                time = []  # convert to numpy
                dt = np.array(time_list)
                dt = dt - dt[0]
                for t in dt:
                    time.append(t.total_seconds())
            else:
                time = df['Time'].values
            ax[i].plot(time,
                       get_plot_data(df, feature),
                       color=color,
                       linestyle=linestyle)

        # Asthetics
        # figure_asthetics(ax[i], subplot=True)
        # ax[i].set_xlim(0, 0.8)
        ax[i].set_ylim(-0.065, 0.065)
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('Displacement (m)')
        ax[i].set_title(item.lower().replace("_", " "))
        plt.legend(['magnet position', 'hammer position'],
                   ncol=2,
                   bbox_to_anchor=(-0.65, 1.25))
        ax[i].grid()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if save_plot:
            name = item.lower().replace(" ",
                                        "-") + "_hammer_magnet_displacement"
            path = str(Path(__file__).parents[2] /
                       config['figure_save_path']) + '/' + name + '.pdf'
            # If the folder does not exist create it
            if not os.path.isdir(config['figure_save_path']):
                os.mkdir(config['figure_save_path'])
            plt.savefig(path, bbox_inches='tight')

    return None


def plot_simulation_trajectories(config, features, ax, save_plot):

    stiffness = ['variable_stiffness']

    # Global plot settings
    path = config['data_path']
    color = ['k', 'b', 'r']
    style = ['-', '--', '-.']

    for i, item in enumerate(stiffness):
        for j, feature in enumerate(features):
            df = pd.read_csv(path + item + '.csv')
            ax.plot(df['time'].values,
                    get_plot_data(df, feature),
                    color=color[j],
                    linestyle=style[j])
            ax.set_xlabel('Time (s)')
            ax.set_xlim([0, 1.35])
            ax.set_ylabel(features[feature])
            ax.grid('on')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.legend(['Hammer', 'Hammer + Base'])
        if save_plot:
            name = item.lower().replace(" ",
                                        "-") + "_hammer_magnet_displacement"
            path = str(Path(__file__).parents[2] /
                       config['figure_save_path']) + '/' + name + '.pdf'
            # If the folder does not exist create it
            if not os.path.isdir(config['figure_save_path']):
                os.mkdir(config['figure_save_path'])
            plt.savefig(path, bbox_inches='tight')

    return None


def plot_experiment_trajectories(config, save_plot):

    stiffness = ['displacement', 'velocity']
    label = ['Displacement (m)', 'Velocity (m/s)']
    features = ['HS', 'LS', 'VS']
    plot_label = ['High stiffness', 'Low stiffness', 'Variable stiffness']

    # Global plot settings
    path = config['data_path']
    color = ['k', 'b', 'r']
    style = ['-', '--', '-.']

    plot_settings()
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6))
    for i, item in enumerate(stiffness):
        for j, feature in enumerate(features):
            if item == 'displacement':
                ax[i].axhline(y=0.05,
                              color='k',
                              linestyle='--',
                              linewidth=0.75)

            df = pd.read_csv(path + item + '.csv')
            ax[i].axvline(x=0.27, color='k', linestyle='--', linewidth=0.75)
            ax[i].axvline(x=0.25, color='k', linestyle='--', linewidth=0.75)
            ax[i].plot(df['time'].values,
                       get_plot_data(df, feature),
                       color=color[j],
                       linestyle=style[j],
                       label=plot_label[i])
            ax[i].set_xlim([0, 0.4])
            ax[i].set_xlabel('Time (s)')
            ax[i].grid(alpha=0.3)
            ax[i].set_ylabel(label[i])
        plt.legend()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if save_plot:
            name = item.lower().replace(" ",
                                        "-") + "_hammer_magnet_displacement"
            path = str(Path(__file__).parents[2] /
                       config['figure_save_path']) + '/' + name + '.pdf'
            # If the folder does not exist create it
            if not os.path.isdir(config['figure_save_path']):
                os.mkdir(config['figure_save_path'])
            plt.savefig(path, bbox_inches='tight')

    return None
