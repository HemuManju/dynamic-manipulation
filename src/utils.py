import os
import sys
import pickle

import deepdish as dd
from contextlib import contextmanager


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """

    @contextmanager
    def check_active():
        deactivated = ['skip']
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip('{:>12}  {:>2}  {:>12}'.format(
                'Skipping the block', '|', f))
            raise SkipWith()
        else:
            p.print_run('{:>12}  {:>3}  {:>12}'.format('Running the block',
                                                       '|', f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end='\n'):
        sys.stderr.write('\x1b[88m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_run(message, end='\n'):
        sys.stdout.write('\x1b[1;32m' + message.strip() + '\x1b[0m' + end)

    @staticmethod
    def print_warn(message, end='\n'):
        sys.stderr.write('\x1b[1;33m' + message.strip() + '\x1b[0m' + end)


def save_dataset(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        hdf5 dataset.
    save : Bool

    """
    if save:
        dd.io.save(path, dataset)

    return None


def save_dataframe(path, dataframe, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save

    save : Bool

    """
    if save:
        with open(path, 'wb') as f:
            pickle.dump(dataframe, f, pickle.HIGHEST_PROTOCOL)

    return None


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


def save_model_log(info, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    with open(save_path + '/' + info['model_name'] + '.pkl', 'wb') as f:
        pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

    return None
