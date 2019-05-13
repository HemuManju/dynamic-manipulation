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
        'bu': (-0.5, 0.5),
        'hd': (w_min, w_max),
        'hu': (0, None),
    }
    for key, value in dependent_variables.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Append derivatives
    derivative = {
        'bu': 'dbudt',
        'hd': 'dhddt',
        'hu': 'dhudt',
    }
    for var in m.component_objects(pyo.Var, active=True):
        m.add_component(derivative[str(var)],
                        pyod.DerivativeVar(var, wrt=m.time))

    # Append control variables
    control_inputs = {'ba': (None, None), 'md': (None, None)}
    for key, value in control_inputs.items():
        m.add_component(key, pyo.Var(m.time, bounds=value))

    # Append differential equations as constraints
    m.ode_bu = pyo.Constraint(m.time,
                              rule=lambda m, time: m.dbudt[time] == m.ba[time])
    m.ode_hd = pyo.Constraint(
        m.time, rule=lambda molde, time: m.dhddt[time] == m.hu[time])

    # A special constraint
    def hammer_acceleration(m, t):
        c1, c2 = 28.41, 206.35
        temp = m.ba[t] - 2 * c1 * pyo.exp(-c2 * (m.md[t] - w_min)) * pyo.sinh(
            c2 * (m.hd[t] - w_min))
        return temp == m.dhudt[t]

    m.ode_hu = pyo.Constraint(m.time, rule=hammer_acceleration)

    # Add final and initial values
    m.ic = pyo.ConstraintList()
    intial_condition = {'bu': 0, 'hd': 0, 'hu': 0, 'ba': 0, 'md': 0}
    for i, var in enumerate(m.component_objects(pyo.Var, active=True)):
        m.ic.add(var[0] == intial_condition[str(var)])

    # Objective function
    m.obj = pyo.Objective(expr=m.hu[m.time.last()], sense=pyo.maximize)

    return m
