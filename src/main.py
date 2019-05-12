import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils import *

config = yaml.load(open('config.yml'))

with skip_run_code('skip', 'create_model') as check, check():
