import pyomo.environ as pyo
import pyomo.dae as pyod
from .pyomoio import get_profiles


def run_optimization(model, n_time_steps):
    """Short summary.

    Parameters
    ----------
    model : pyomo model
        A pyomo model with all the states, control and constraints described.
    n_time_steps : int
        Number of time steps to use in the simulation.

    Returns
    -------
    m, optimal_values
        An instance of optimised model and dataframe of all the optimal values.

    """

    # Create a model instance
    m = model
    # Transform and solve
    pyo.TransformationFactory('dae.finite_difference').apply_to(
        m, nfe=n_time_steps, wrt=m.time, scheme='FORWARD')
    opt = pyo.SolverFactory('ipopt')
    solution = opt.solve(m)
    solution.write()

    # Get the dataframe of all the states and control
    optimal_values = get_profiles(m)

    return m, optimal_values, solution
