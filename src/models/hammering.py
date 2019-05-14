import pyomo.environ as pyo
import pyomo.dae as pyod


def motion_model(tf):
    """Motion model for hammer task from given intial conditions to final conditions.

    Parameters
    ----------
    tf : float
        Final time of the maneuvering.

    Returns
    -------
    m
        A pyomo model with all the variables and constraints described.

    """

    m = pyo.ConcreteModel()
    m.time = pyod.ContinuousSet(bounds=(0, tf))
    # Parameters
    h_mass = 0.5  # mass of the hammer
    w_min = 0.03
    w_max = 0.07

    # Append dependent variables
    dependent_variables = {
        'bd': (-0.3, 0.3),
        'bv': (-0.5, 0.5),
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
    control_inputs = {'ba': (-4, 4), 'md': (w_min, w_max)}
    for key, value in control_inputs.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Add constraint on the magnet velocity
    m.add_component('mv', pyod.DerivativeVar(m.md, wrt=m.time))

    # Append differential equations as constraints
    m.ode_bd = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dbddt[time] == m.bv[time])
    m.ode_bv = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dbvdt[time] == m.ba[time])
    m.ode_hd = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dhddt[time] == m.hv[time])

    m.ode_md = pyo.Constraint(m.time, rule=lambda m, time: m.mv[time] <= 0.08)

    # A special constraint
    def hammer_acceleration(m, t):
        c1, c2 = 28.41, 206.35
        temp = -m.ba[t] * h_mass + 2 * c1 * pyo.exp(
            -c2 * (m.md[t] - w_min)) * pyo.sinh(c2 * (m.hd[t])) + 1 * m.hv[t]
        return m.dhvdt[t] == temp / h_mass

    m.ode_hv = pyo.Constraint(m.time, rule=hammer_acceleration)

    m.disp_1 = pyo.Constraint(
        m.time,
        rule=lambda m, time: m.bd[m.time.last()] + m.hd[m.time.last()] == 0.3)
    m.disp_2 = pyo.Constraint(
        m.time, rule=lambda m, time: m.hd[time] <= (m.md[time] - w_min))
    m.disp_3 = pyo.Constraint(
        m.time, rule=lambda m, time: m.hd[time] >= -(m.md[time] - w_min))

    # Add final and initial values
    m.ic = pyo.ConstraintList()
    intial_condition = {
        'bd': 0.0,
        'bv': 0.0,
        'hd': 0.0,
        'hv': 0.0,
        'ba': 0.0,
        'md': 0.07
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
