import pyomo.environ as pyo
import pyomo.dae as pyod


def motion_model(tf):
    """Motion model for car maneuvering task from given intial
        conditions to final conditions.

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

    # Append dependent variables
    dependent_variables = {
        'x': (0, None),
        'y': (0, None),
        't': (None, None),
        'u': (None, None),
        'p': (-0.5, 0.5)
    }
    for key, value in dependent_variables.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Append derivatives
    derivative = {
        'x': 'dxdt',
        'y': 'dydt',
        't': 'dtdt',
        'u': 'dudt',
        'p': 'dpdt'
    }
    for var in m.component_objects(pyo.Var, active=True):
        m.add_component(derivative[str(var)],
                        pyod.DerivativeVar(var, wrt=m.time))

    # Append control variables
    control_inputs = {'a': (None, None), 'v': (-0.1, 0.1)}
    for key, value in control_inputs.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Append differential equations as constraints
    L = 2
    m.ode_x = pyo.Constraint(
        m.time,
        rule=lambda m, time: m.dxdt[time] == m.u[time] * pyo.cos(m.t[time]))
    m.ode_y = pyo.Constraint(
        m.time,
        rule=lambda m, time: m.dydt[time] == m.u[time] * pyo.sin(m.t[time]))
    m.ode_t = pyo.Constraint(m.time,
                             rule=lambda m, time: m.dtdt[time] == m.u[time] *
                             pyo.tan(m.p[time]) / L)
    m.ode_u = pyo.Constraint(m.time,
                             rule=lambda m, time: m.dudt[time] == m.a[time])
    m.ode_p = pyo.Constraint(m.time,
                             rule=lambda m, time: m.dpdt[time] == m.v[time])

    # Add final and initial values
    m.ic = pyo.ConstraintList()
    intial_condition = {'x': 0, 'y': 0, 't': 0, 'u': 0, 'p': 0, 'a': 0, 'v': 0}
    for i, var in enumerate(m.component_objects(pyo.Var, active=True)):
        m.ic.add(var[0] == intial_condition[str(var)])

    m.fc = pyo.ConstraintList()
    final_condition = {'x': 0, 'y': 20, 't': 0, 'u': 0, 'p': 0, 'a': 0, 'v': 0}
    for i, var in enumerate(m.component_objects(pyo.Var, active=True)):
        m.fc.add(var[tf] == final_condition[str(var)])

    # Objective function
    m.integral = pyod.Integral(
        m.time,
        wrt=m.time,
        rule=lambda m, time: 0.2 * m.p[time]**2 + m.a[time]**2 + m.v[time]**2)
    m.obj = pyo.Objective(expr=m.integral)

    return m
