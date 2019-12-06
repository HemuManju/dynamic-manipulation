import numpy as np
from scipy.special import erf
from pathlib import Path
import yaml

import pyomo.environ as pyo
import pyomo.dae as pyod


def skew_normal_trajectory(m, t):
    """This function generates a predefined trajectory
    for robot end effector.

    Parameters
    ----------
    m : pyomo model
        A pyomo model containing dynamic model and constraints.
    t : pyomo time
        A pyomo time model.
    """
    # The configuration file
    config_path = Path(__file__).parents[2] / 'src/config.yml'
    config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)
    tf = config['tf']

    # Constraint
    tp = config['t1'] + (config['t2'] - config['t1']) * t / tf
    phi = np.exp(-(tp**2) / 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + erf(config['alpha'] * tp / np.sqrt(2)))

    return m.bd[t] == (config['A'] * phi * Phi)


def dynamic_motion_model(tf, stiffness, config):
    """Motion model for hammer task from given intial conditions
    to final conditions.

    Parameters
    ----------
    tf : float
        Final time of the maneuvering.
    stiffness : str
        Stiffness of springs used for simulation
    config : yaml
        The configuration file for the simulation

    Returns
    -------
    m
        A pyomo model with all the variables and constraints described.

    """

    m = pyo.ConcreteModel()
    m.time = pyod.ContinuousSet(bounds=(0, tf))  # idependent variable (time)

    # Parameters
    w = 0.03
    h_mass = config['h_mass']  # mass of the hammer

    if stiffness == 'low_stiffness':
        w_min, w_max = config['w_max'], config['w_max']
    elif stiffness == 'high_stiffness':
        w_min, w_max = config['w_min'], config['w_min']
    else:
        w_min, w_max = config['w_min'], config['w_max']

    print(w_min, w_max)  # just to check once
    # Append dependent variables
    independent_variables = {
        'bd': (config['bd_min'], config['bd_max']),
        'hd': (None, None),
        'md': (w_min, w_max),
    }
    for key, value in independent_variables.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    if stiffness == 'variable_stiffness':
        mv_bounds = (-0.15, 0.15)
    else:
        mv_bounds = (0.0, 0.0)

    # Append first derivatives
    derivatives = {'bv': m.bd, 'hv': m.hd, 'mv': m.md}
    bounds = {
        'bv': (config['bv_min'], config['bv_max']),
        'hv': (None, None),
        'mv': mv_bounds
    }
    for key in derivatives.keys() & bounds.keys():
        m.add_component(
            key,
            pyod.DerivativeVar(derivatives[key],
                               wrt=m.time,
                               bounds=bounds[key]))

    # Append second derivatives
    derivatives = {'ba': m.bv, 'ha': m.hv}
    bounds = {
        'ba': (config['ba_min'], config['ba_max']),
        'ha': (None, None),
    }
    for key in derivatives.keys() & bounds.keys():
        m.add_component(
            key,
            pyod.DerivativeVar(derivatives[key],
                               wrt=m.time,
                               bounds=bounds[key]))

    # Hammer movement dynamics
    def hammer_acceleration(m, t):
        c1, c2 = 28.41, 206.35
        temp = +m.ba[t] * h_mass + 2 * c1 * pyo.exp(
            -c2 * (m.md[t] - w)) * pyo.sinh(c2 * (m.hd[t])) + 1 * m.hv[t]
        return m.ha[t] == -temp / h_mass

    m.ode_hv = pyo.Constraint(m.time, rule=hammer_acceleration)

    # Displacement constraints (end position and hammer displacement)
    m.disp_1 = pyo.Constraint(m.time,
                              rule=lambda m, time: m.hd[time] <=
                              (m.md[time] - w))
    m.disp_2 = pyo.Constraint(
        m.time, rule=lambda m, time: m.hd[time] >= -(m.md[time] - w))

    # Add initial values of independent variables
    m.ic = pyo.ConstraintList()
    intial_condition = {
        'bd': 0.0,
        'bv': 0.0,
        'ba': 0.0,
        'hd': 0.0,
        'hv': 0.0,
        'ha': 0.0,
        'md': w_max,
        'mv': 0.0,
    }
    for i, var in enumerate(m.component_objects(pyo.Var)):
        m.ic.add(var[0] == intial_condition[str(var)])

    for i, var in enumerate(m.component_objects(pyod.DerivativeVar)):
        m.ic.add(var[0] == intial_condition[str(var)])

    # End effector zero velocity constraint at the end
    m.ic.add(m.bv[tf] == 0)
    m.ic.add(m.bd[tf] == config['path_length'])

    # Objective function
    m.obj = pyo.Objective(expr=m.hv[tf], sense=pyo.maximize)

    return m


def dynamic_motion_model_with_trajectory(tf, stiffness, config):
    """Motion model for hammer task from given intial conditions
    to final conditions.

    Parameters
    ----------
    tf : float
        Final time of the maneuvering.
    stiffness : str
        Stiffness of springs used for simulation
    config : yaml
        The configuration file for the simulation

    Returns
    -------
    m
        A pyomo model with all the variables and constraints described.

    """

    m = pyo.ConcreteModel()
    m.time = pyod.ContinuousSet(bounds=(0, tf))  # idependent variable (time)

    # Parameters
    # m.d = pyo.Param(tf, config)
    w = 0.03
    h_mass = config['h_mass']  # mass of the hammer
    if stiffness == 'low_stiffness':
        w_min, w_max = config['w_max'], config['w_max']
    elif stiffness == 'high_stiffness':
        w_min, w_max = config['w_min'], config['w_min']
    else:
        w_min, w_max = config['w_min'], config['w_max']

    print(w_min, w_max)  # just to check once
    # Append dependent variables
    dependent_variables = {
        'bd': (config['bd_min'], config['bd_max']),
        'bv': (config['bv_min'], config['bv_max']),
        'hd': (None, None),
        'hv': (None, None),
    }
    for key, value in dependent_variables.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Append derivatives
    derivative = {
        'bd': 'dbddt',
        'bv': 'dbvdt',
        'hd': 'dhddt',
        'hv': 'dhvdt',
    }
    for var in m.component_objects(pyo.Var, active=True):
        m.add_component(derivative[str(var)],
                        pyod.DerivativeVar(var, wrt=m.time))

    # Append control variables
    control_inputs = {
        'ba': (config['ba_min'], config['ba_max']),
        'md': (w_min, w_max)
    }
    for key, value in control_inputs.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Add constraint on the magnet velocity
    if stiffness == 'variable_stiffness':
        mv_bounds = (-0.08, 0.08)
    else:
        mv_bounds = (0, 0)
    m.add_component('mv', pyod.DerivativeVar(m.md,
                                             wrt=m.time,
                                             bounds=mv_bounds))

    # Append differential equations as constraints
    m.ode_bd = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dbddt[time] == m.bv[time])
    m.ode_bv = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dbvdt[time] == m.ba[time])
    m.ode_hd = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dhddt[time] == m.hv[time])

    # Hammer movement dynamics
    def hammer_acceleration(m, t):
        c1, c2 = 28.41, 206.35
        temp = -m.ba[t] * h_mass - 2 * c1 * pyo.exp(
            -c2 * (m.md[t] - w)) * pyo.sinh(c2 * (m.hd[t])) - 0.5 * m.hv[t]
        return m.dhvdt[t] == temp / h_mass

    m.ode_hv = pyo.Constraint(m.time, rule=hammer_acceleration)
    m.disp_2 = pyo.Constraint(m.time,
                              rule=lambda m, time: m.hd[time] <=
                              (m.md[time] - w))
    m.disp_3 = pyo.Constraint(
        m.time, rule=lambda m, time: m.hd[time] >= -(m.md[time] - w))

    # End effector zero velocity constraint at the end
    m.base_velocity = pyo.Constraint(expr=m.bv[m.time.last()] == 0)

    # Trajectory constraint
    m.base_trajectory = pyo.Constraint(m.time, rule=skew_normal_trajectory)

    # Add initial values
    m.ic = pyo.ConstraintList()
    intial_condition = {
        # 'bd': 0.0,
        'bv': 0.0,
        'ba': 0.0,
        'hd': 0.0,
        'hv': 0.0,
        'md': (w_min + w_max) / 2,
    }
    for i, var in enumerate(m.component_objects(pyo.Var, active=True)):
        if str(var) != 'bd':
            m.ic.add(var[0] == intial_condition[str(var)])

    # Objective function
    m.obj = pyo.Objective(expr=m.hv[m.time.last()], sense=pyo.maximize)

    return m


def differential_flat_model(tf, stiffness, config):
    """Differentially flat model for hammering task.

    Parameters
    ----------
    tf : float
        Final time of the maneuvering.
    stiffness : str
        Stiffness of springs used for simulation
    config : yaml
        The configuration file for the simulation

    Returns
    -------
    m
        A pyomo model with all the variables and constraints described.

    """

    m = pyo.ConcreteModel()
    m.time = pyod.ContinuousSet(bounds=(0, tf))  # idependent variable (time)

    # Parameters
    w = 0.03
    h_mass = config['h_mass']  # mass of the hammer
    if stiffness == 'low_stiffness':
        w_min, w_max = config['w_max'], config['w_max']
    elif stiffness == 'high_stiffness':
        w_min, w_max = config['w_min'], config['w_min']
    else:
        w_min, w_max = config['w_min'], config['w_max']

    # Append dependent variables
    dependent_variables = {
        'bd': (config['bd_min'], config['bd_max']),
        'bv': (config['bv_min'], config['bv_max']),
        'hd': (None, None),
        'hv': (None, None),
    }
    for key, value in dependent_variables.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Append the polynomial terms
    m.a = pyo.Var(range(3), bounds=(None, None))
    m.b = pyo.Var(range(4), bounds=(None, None))

    # Append control variables
    control_inputs = {
        'ba': (config['ba_min'], config['ba_max']),
        'md': (w_min, w_max)
    }
    for key, value in control_inputs.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    def hammer_acceleration(m, t):
        c1, c2 = 28.41, 206.35
        temp = w - 1 / c2 * pyo.log((m.md[t] + m.b[1] + 2 * m.b[2] * t) /
                                    (-2 * c1 * pyo.sinh(c2 * (m.hd[t]))))
        return m.ba[t] == temp / h_mass

    # Constraints
    m.eq_u1 = pyo.Constraint(m.time,
                             rule=lambda m, time: m.md[time] ==
                             (m.a[1] + 2 * m.a[2] * time))

    m.eq_u2 = pyo.Constraint(m.time, rule=hammer_acceleration)

    m.eq_bv = pyo.Constraint(m.time,
                             rule=lambda m, time: m.bv[time] ==
                             (m.a[0] + m.a[1] * time + m.a[2] * time**2))

    m.eq_hv = pyo.Constraint(m.time,
                             rule=lambda m, time: m.hv[time] ==
                             (m.b[0] + m.b[1] * time + 0.5 * m.b[2] * time**2 +
                              m.b[3] * time**3 / 3))

    # # Add constraint on the magnet velocity
    # if stiffness == 'variable_stiffness':
    #     mv_bounds = (-0.08, 0.08)
    # else:
    #     mv_bounds = (0, 0)
    # m.add_component('mv', pyod.DerivativeVar(m.md,
    #                                          wrt=m.time,
    #                                          bounds=mv_bounds))

    # Displacement constraints (end position and hammer displacement)
    m.disp_1 = pyo.Constraint(m.time,
                              rule=lambda m, time: m.bd[m.time.last()] + m.hd[
                                  m.time.last()] == config['path_length'])
    m.disp_2 = pyo.Constraint(m.time,
                              rule=lambda m, time: m.hd[time] <=
                              (m.md[time] - w))
    m.disp_3 = pyo.Constraint(
        m.time, rule=lambda m, time: m.hd[time] >= -(m.md[time] - w))

    # Add initial values
    m.ic = pyo.ConstraintList()
    intial_condition = {
        'bd': 0.0,
        'bv': 0.0,
        'ba': 0.0,
        'hd': 0.0,
        'hv': 0.0,
        'md': (w_min + w_max) / 2,
    }
    for i, var in enumerate(m.component_objects(pyo.Var, active=True)):
        if str(var) == 'a':
            pass
        elif str(var) == 'b':
            pass
        else:
            m.ic.add(var[0] == intial_condition[str(var)])

    # # Initialise the variables
    # for var in m.component_objects(pyo.Var, active=True):
    #     if str(var) == 'a':
    #         for i in range(3):
    #             m.a[i] = np.random.rand(1).tolist()[0]
    #     elif str(var) == 'b':
    #         for i in range(4):
    #             m.b[i] = np.random.rand(1).tolist()[0]

    # Objective function
    m.obj = pyo.Objective(expr=m.hv[m.time.last()] + m.bv[m.time.last()],
                          sense=pyo.maximize)

    return m
