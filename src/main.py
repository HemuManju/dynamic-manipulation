import yaml
import numpy as np
from models import car_maneuver, hammering
from models import optimize
import seaborn as sb
import matplotlib.pyplot as plt
from utils import *

# config = yaml.load(open('config.yml'))

with skip_run_code('skip', 'car_maneuver_model') as check, check():
    tf = 50.0
    m = car_maneuver.motion_model(tf)

with skip_run_code('run', 'hammering_model') as check, check():
    tf = 2.0
    m = hammering.motion_model(tf)

with skip_run_code('run', 'optimize_model') as check, check():
    m, optimal_values = optimize.run_optimization(m, 200)

    sb.set()
    optimal_values.plot(x='time', y=['bv', 'hd', 'hv', 'md'], subplots=True)
    plt.show()
