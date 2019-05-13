import yaml
import numpy as np
from models import car_maneuver
from models import optimize
import matplotlib.pyplot as plt
from utils import *

# config = yaml.load(open('config.yml'))

with skip_run_code('run', 'create_model') as check, check():
    tf = 50.0
    m = car_maneuver.motion_model(tf)

with skip_run_code('run', 'optimize_model') as check, check():
    m, optimal_values = optimize.run_optimization(m, 50)
    #
    # plt.plot(optimal_values['x'].values, optimal_values['y'].values)
    # plt.axis('square')
    # plt.show()
