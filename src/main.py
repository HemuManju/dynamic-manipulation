import yaml
import pyomo.environ as pyo
import numpy as np
from models import car_maneuver, hammering
from models import optimize
import seaborn as sb
import matplotlib.pyplot as plt
from utils import *

# config = yaml.load(open('config.yml'))

with skip_run('skip', 'car_maneuver_model') as check, check():
    tf = 50.0
    m = car_maneuver.motion_model(tf)

with skip_run('skip', 'hammer_model_binary_search') as check, check():
    tf_min = 1.0
    tf_max = 5.0
    while (tf_max - tf_min) >= 10e-2:
        m = hammering.motion_model(tf_max)
        m, optimal_values, solution = optimize.run_optimization(m, 500)
        if solution.solver.termination_condition == pyo.TerminationCondition.optimal:
            tf_max = (tf_min + tf_max) / 2
        elif solution.solver.termination_condition == pyo.TerminationCondition.infeasible:
            tf_min = (tf_min + tf_max) / 2
    print(tf_max)

with skip_run('run', 'hammering_model') as check, check():
    tf = 2.0
    m = hammering.motion_model(tf)

with skip_run('run', 'optimize_model') as check, check():
    m, optimal_values, solution = optimize.run_optimization(m, 500)

    sb.set()
    print(m.obj())
    optimal_values.plot(x='time',
                        y=['bv', 'hd', 'hv', 'md', 'ba'],
                        subplots=True)
    plt.show()
