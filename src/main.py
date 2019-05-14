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

with skip_run('run', 'hammer_model_binary_search') as check, check():
    tf_min = 0.0
    tf_max = 3.0
    v = [0]
    t = []
    while (tf_max - tf_min) >= 10e-3:
        m = hammering.motion_model(tf_max)
        m, optimal_values, solution = optimize.run_optimization(m, 500)
        temp = optimal_values['hv'].values
        v.append(temp[-1])
        t.append(tf_max)

        if solution.solver.termination_condition == pyo.TerminationCondition.optimal:
            if (v[-1] - v[-2]) <= 0:
                tf_max = (tf_min + tf_max) / 2
            else:
                tf_min = (tf_min + tf_max) / 2
        elif solution.solver.termination_condition == pyo.TerminationCondition.infeasible:
            tf_min = (tf_min + tf_max) / 2

    plt.plot(v)
    plt.show()
    print(v)
    print(t)
    print(tf_max)

with skip_run('run', 'hammering_model') as check, check():
    tf = tf_max  # 2.00390625
    m = hammering.motion_model(tf)

with skip_run('run', 'optimize_model') as check, check():
    m, optimal_values, solution = optimize.run_optimization(m, 500)

    sb.set()
    print(m.obj())
    optimal_values.plot(x='time',
                        y=['ba', 'bv', 'bd', 'hd', 'hv', 'md'],
                        subplots=True)
    plt.show()
