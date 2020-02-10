from pybamm import exp, standard_parameters_lithium_ion


def lico2_electrolyte_reaction_rate_Dualfoil1998(T):
    """
    Reaction rate for Butler-Volmer reactions between lico2 and LiPF6 in EC:DMC.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Dimensional reaction rate [(A.m-2)(m3.mol-1)^1.5]
    """
    param = standard_parameters_lithium_ion
    m_ref = 6 * 10 ** (-7)
    arrhenius = exp(param.E_r_p / param.R * (1 / param.T_ref - 1 / T))

    return m_ref * arrhenius
