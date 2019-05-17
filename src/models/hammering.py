import pyomo.environ as pyo
import pyomo.dae as pyod


def motion_model(tf, stiffness, config):
    """Motion model for hammer task from given intial conditions to final conditions.

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
    m.time = pyod.ContinuousSet(bounds=(0, tf))
    # Parameters
    h_mass = config['h_mass']  # mass of the hammer
    if stiffness == 'low_stiffness':
        w_min, w_max = config['w_max'], config['w_max']
    elif stiffness == 'high_stiffness':
        w_min, w_max = config['w_min'], config['w_min']
    else:
        w_min, w_max = config['w_min'], config['w_max']

    print(w_min, w_max)
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
    m.add_component('mv',
                    pyod.DerivativeVar(m.md, wrt=m.time, bounds=(-0.08, 0.08)))

    # Append differential equations as constraints
    m.ode_bd = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dbddt[time] == m.bv[time])
    m.ode_bv = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dbvdt[time] == m.ba[time])
    m.ode_hd = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dhddt[time] == m.hv[time])

    # A special constraint
    def hammer_acceleration(m, t):
        c1, c2 = 28.41, 206.35
        temp = -m.ba[t] * h_mass - 2 * c1 * pyo.exp(
            -c2 * (m.md[t] - w_min)) * pyo.sinh(c2 * (m.hd[t])) - 1 * m.hv[t]
        return m.dhvdt[t] == temp / h_mass

    m.ode_hv = pyo.Constraint(m.time, rule=hammer_acceleration)

    m.disp_1 = pyo.Constraint(
        m.time,
        rule=lambda m, time: m.bd[m.time.last()] == config['path_length'])
    m.disp_2 = pyo.Constraint(
        m.time, rule=lambda m, time: m.hd[time] <= (m.md[time] - w_min))
    m.disp_3 = pyo.Constraint(
        m.time, rule=lambda m, time: m.hd[time] >= -(m.md[time] - w_min))

    # Add final and initial values
    m.ic = pyo.ConstraintList()
    intial_condition = {
        'bd': 0.0,
        'bv': 0.0,
        'ba': 0.0,
        'hd': 0.0,
        'hv': 0.0,
        'md': 0.07,
    }
    for i, var in enumerate(m.component_objects(pyo.Var, active=True)):
        m.ic.add(var[0] == intial_condition[str(var)])

    # Objective function
    m.integral = pyod.Integral(m.time,
                               wrt=m.time,
                               rule=lambda m, time: m.ba[time] + m.dhvdt[time])
    m.obj = pyo.Objective(expr=m.integral, sense=pyo.maximize)
    # m.obj = pyo.Objective(expr=m.hv[m.time.last()] + m.bv[m.time.last()],
    #                       sense=pyo.maximize)

    return m
