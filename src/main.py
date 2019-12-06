from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from models import car_maneuver, hammering
from models import optimize
from models.utils import export_trajectory_data
from visualization.visualize import (plot_optimal_trajectories,
                                     plot_magnet_hammer_path,
                                     plot_magnet_hammer,
                                     plot_simulation_trajectories,
                                     plot_experiment_trajectories)
from visualization.utils import plot_settings
from utils import (skip_run, save_model_log)

# The configuration file
config_path = Path(__file__).parents[1] / 'src/config.yml'
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run('skip', 'car_maneuver_model') as check, check():
    tf = 50.0
    m = car_maneuver.motion_model(tf)

with skip_run('skip', 'dynamic_model_binary_search') as check, check():
    tf_min = 0.7
    tf_max = 2.0
    v = [0]
    t = []
    while (tf_max - tf_min) >= 10e-3:
        print(tf_max)
        m = hammering.dynamic_motion_model(tf_max, 'variable_stiffness',
                                           config)
        m, optimal_values, solution = optimize.run_optimization(m, 200)
        temp = optimal_values['hv'].values
        v.append(temp[-1])
        t.append(tf_max)

        # Conditions
        optimal_condition = pyo.TerminationCondition.optimal
        infeasible_condition = pyo.TerminationCondition.infeasible

        if solution.solver.termination_condition == optimal_condition:
            # if (v[-1] - v[-2]) <= 0:
            tf_max = (tf_min + tf_max) / 2
            # else:
            # tf_min = (tf_min + tf_max) / 2
        elif solution.solver.termination_condition == infeasible_condition:
            tf_min = (tf_min + tf_max) / 2

        print(v)
        print(tf_max, tf_min)

    # plt.plot(v)
    # plt.show()
    print(v)
    print(t)
    print(tf_max)

with skip_run('skip', 'dynamic_model_optimize') as check, check():
    tf = 2.0  # tf_max (optimal) 1.505859375 0.78125
    for item in config['stiffness']:
        output = {}
        print(item)
        m = hammering.dynamic_motion_model(tf, item, config)
        m, optimal_values, solution = optimize.run_optimization(m, 500)

        # Variables
        # ['bd', 'bv', 'ba', 'hd', 'hv', 'md', 'mv']
        print(m.obj())

        # Append model and other information
        output['obj_values'] = m.obj()
        output['optimal_values'] = optimal_values
        output['solver_status'] = solution.solver.termination_condition
        output['model_name'] = item

        save_path = str(Path(__file__).parents[1] / config['save_path'])
        save_model_log(output, save_path)

with skip_run('skip', 'differential_flat_model') as check, check():
    tf = 5.0
    m = hammering.differential_flat_model(tf, 'variable', config)
    print(m.display())
    m, optimal_values, solution = optimize.run_optimization(m, 300)
    # m = hammering.dynamic_motion_model(tf, 'low_stiffness', config)
    print(m.obj())

with skip_run('skip', 'export_optimal_trajectories') as check, check():
    stiffness = 'low_stiffness'
    trajectories = ['time', 'bd', 'bv', 'hd', 'hv', 'md', 'mv']
    export_trajectory_data(config, stiffness, trajectories)

with skip_run('skip', 'plot_trajectories') as check, check():
    features = {
        'bd': 'end-effector displacement (m)',
        'bd + hd': 'Hammer displacement (m)',
        'bv + hv': 'Hammer velocity (m/s)',
        'md': 'Magnet separation (m)'
    }
    plot_optimal_trajectories(config, features, save_plot=True)
    plt.show()

with skip_run('skip', 'plot_hammer_magnet_trajectory') as check, check():
    plot_magnet_hammer_path(config, save_plot=False)
    plt.show()

with skip_run('skip', 'plot_hammer_magnet_external') as check, check():
    plot_magnet_hammer(config, save_plot=False)
    plt.show()

with skip_run('skip', 'plot_simulation_trajectories') as check, check():
    features = {
        'bd': 'Displacement (m)',
        'bd + hd': 'Displacement (m)',
    }
    plot_settings()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    plot_simulation_trajectories(config, features, ax[0], save_plot=False)

    features = {
        'bv': 'Velocity (m/s)',
        'bv + hv': 'Velocity (m/s)',
    }
    plot_simulation_trajectories(config, features, ax[1], save_plot=False)
    plt.show()

with skip_run('run', 'plot_experiment_trajectories') as check, check():

    plot_settings()
    plot_experiment_trajectories(config, save_plot=False)
    plt.show()