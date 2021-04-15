'''
This script was created by Chrishon Nilanthan during his master's thesis 
titled "1D Radiative Transfer for Reionization Simulations"

Script is modified later by Sambit K. Giri.
'''

import scipy.integrate as integrate
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interpolate
from tqdm import tqdm
from astropy import units as u
from astropy.cosmology import WMAP7 as pl
from scipy.optimize import fsolve
from sys import exit
import dill
from glob import glob
import pickle


# Initialize the units and conversion factors
facr = 1 * u.Mpc
facr = (facr.to(u.cm)).value
facE = (1 * u.eV).to(u.erg).value
cm_3 = u.cm ** -3
s1 = u.s ** -1
u1 = u.erg * u.cm ** 3 * u.s ** -1
u2 = u.cm ** 3 * u.s ** -1
u3 = u.erg * u.s ** -1

# H, He density and mass
n_H_0 = 1.9 * 10 ** -7 * cm_3
n_He_0 = 1.5 * 10 ** -8 * cm_3
m_H = 1.6 * 10 ** - 24 * u.g
m_He = 6.6464731 * 10 ** - 24 * u.g

# Energy limits and ionization energies
E_0 = 10.4 * facE
E_HI = 13.6 * facE
E_HeI = 24.59 * facE
E_HeII = 54.42 * facE
E_upp = 10 ** 4 * facE
E_cut = 200. * facE

# Constants
h = 1.0546 * 10 ** -27 * u.erg * u.s
c = 2.99792 * 10 ** 10 * u.cm / u.s
kb = 1.380649 * 10 ** -23 * u.J / u.K
kb = kb.to(u.erg / u.K)
m_e = 9.10938 * 10 ** -28 * u.g
sigma_s = 6.6524 * 10 ** -25 * u.cm ** 2


# Photoionization and Recombination coefficients
def alpha_HII(T):
    # return(6.28 * 10 ** -11 * T ** -0.5 * (T/10**3) ** -0.2 * (1+(T/10**6)**0.7)**-1)
    return 2.6 * 10 ** -13 * (T / 10 ** 4) ** -0.8

def alpha_HeII(T):
    return 1.5 * 10 ** -10 * T ** -0.6353

def alpha_HeIII(T):
    return 3.36 * 10 ** -10 * T ** -0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / (4 * 10 ** 6)) ** 0.7) ** -1

def beta_HI(T):
    # print(5.85*10**-11*T**0.5*(1+(T/10**5)**0.5)**-1*exp(-1.578*10**5/T))
    return 5.85 * 10 ** -11 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.578 * 10 ** 5 / T)

def beta_HeI(T):
    return 2.38 * 10 ** -11 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-2.853 * 10 ** 5 / T)

def beta_HeII(T):
    return 5.68 * 10 ** -12 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-6.315 * 10 ** 5 / T)

def xi_HI(T):
    return 1.27 * 10 ** -21 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.58 * 10 ** 5 / T)

def xi_HeI(T):
    return 9.38 * 10 ** -22 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-2.85 * 10 ** 5 / T)

def xi_HeII(T):
    return 4.95 * 10 ** -22 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-6.31 * 10 ** 5 / T)

def eta_HII(T):
    return 6.5 * 10 ** -27 * T ** 0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / 10 ** 6) ** 0.7) ** -1

def eta_HeII(T):
    return 1.55 * 10 ** -26 * T ** 0.3647

def eta_HeIII(T):
    return 3.48 * 10 ** -26 * T ** 0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / (4 * 10 ** 6)) ** 0.7) ** -1

def psi_HI(T):
    return 7.5 * 10 ** -19 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.18 * 10 ** 5 / T)

def psi_HeI(T, neT, n_HeIIT):
    return 9.1 * 10 ** -27 * T ** -0.1687 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.31 * 10 ** 4 / T)

def psi_HeII(T):
    return 5.54 * 10 ** -17 * T ** -0.397 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-4.73 * 10 ** 5 / T)

def omega_HeII(T):
    return 1.24 * 10 ** -13 * T ** -1.5 * exp(-4.7 * 10 ** 5 / T) * (1 + 0.3 * exp(-9.4 * 10 ** 4 / T))

def theta_ff(T):
    return 1.3 * 1.42 * 10 ** -27 * (T) ** 0.5

def zeta_HeII(T):
    return 1.9 * 10 ** -3 * T ** -1.5 * exp(-4.7 * 10 ** 5 / T) * (1 + 0.3 * exp(-9.4 * 10 ** 4 / T))



# HI, HeI, HeII Photoionization cross sections
def sigma_HI(E):

    sigma_0 = 5.475 * 10 ** 4 * 10 ** -18
    E_01 = 4.298 * 10 ** -1 * facE
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = y_0 = y_1 = 0
    x = E / E_01 - y_0
    y = sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma

def sigma_HeI(E):

    sigma_0 = 9.492 * 10 ** 2 * 10 ** -18
    E_01 = 1.361 * 10 ** 1 * facE
    y_a = 1.469
    P = 3.188
    y_w = 2.039
    y_0 = 4.434 * 10 ** -1
    y_1 = 2.136
    x = E / E_01 - y_0
    y = sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma

def sigma_HeII(E):

    sigma_0 = 1.369 * 10 ** 4 * 10 ** -18  # cm**2
    E_01 = 1.72 * facE
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = 0
    y_0 = 0
    y_1 = 0
    x = E / E_01 - y_0
    y = sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma

# H and He IGM densities
def n_H(z):
    return n_H_0 * (1 + z) ** 3

def n_He(z):
    return n_He_0 * (1 + z) ** 3

# Factor for secondary ionization and heating factor
def f_H(n_HII, z_reion):
    x = (nan_to_num(n_HII / n_H(z_reion))).value
    if x.any() < 0:
        return 0.3908 * (1 - 0 ** 0.4092) ** 1.7592
    if x.any() > 1:
        return 0.3908 * (1 - 1 ** 0.4092) ** 1.7592

    return nan_to_num(0.3908 * (1 - x ** 0.4092) ** 1.7592)

def f_He(n_HII, z_reion):
    x = (nan_to_num(n_HII / n_H(z_reion))).value

    if x.any() < 0:
        return 0.0554 * (1 - 0 ** 0.4614) ** 1.6660
    if x.any() > 1:
        return 0.0554 * (1 - 1 ** 0.4614) ** 1.6660

    return nan_to_num(0.0554 * (1 - x ** 0.4614) ** 1.6660)

def f_Heat(n_HII, z_reion):
    xion = (nan_to_num(n_HII / n_H(z_reion))).value
    if xion.any() < 0:
        return 0.9971 * (1 - (1 - 0 ** 0.2663) ** 1.3163)
    if xion.any() > 1:
        return 0.9971 * (1 - (1 - 1 ** 0.2663) ** 1.3163)
    if xion.any() > 10 ** -4:
        return nan_to_num(0.9971 * (1 - (1 - xion ** 0.2663) ** 1.3163))
    return 0.15

def find_Ifront(n, r, z_reion, show=None):
    """
    Finds the ionization front, the position where the ionized fraction is 0.5.

    Parameters
    ----------
    n : array_like
     Ionized hydrogen density along the radial grid.
    r : array_like
     Radial grid in linear space.
    z_reion :
     Redshift of the source.
    show : bool, optional
     Decide whether or not to print the position of the ionization front

     Returns
     -------
     float
      Returns the position of the ionization front, one of the elements in the array r.
    """
    m = 0
    x = n.value / n_H(z_reion).value
    m = argmin(abs(0.5 - x))
    check = show if show is not None else False

    if check == True:
        print('Pos. of Ifront:', r[m])
    return r[m]

def generate_table(M, z, r_grid, n_HI, n_HeI, n_HeII, alpha, sed, filename_table=None,recalculate_table=False):
    '''
    Generate the interpolation tables for the integrals for the radiative transfer equations.

    This function generates 3-D tables for each integral in the radiative transfer equation. The elements of the tables
    are the values of the integrals for a given set of variables (n_HI, n_HeI, n_HeII). This is done by initializing a
    array for each variable within a specified range and then loop over the arrays and use each set of variables to
    evaluate the integrals. The integral range is set up to include UV ionizing photons.

    Parameters
    ----------
    M : float
     Mass of the source.
    z : float
     Redshift of the source.
    r_grid : array_like
     Radial Mpc grid in logspace from the source to a given distance, the starting distance is typically 0.0001 Mpc from
     the source.
    n_HI : array_like
     Neutral hydrogen density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    n_HeI : array_like
     Neutral helium density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    n_HI : array_like
     HeII density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    alpha : int, default -1
     Spectral index for a power-law source.
    sed : callable, optional
     Spectral energy distribution to be used instead of the default power-law function. sed is a function of energy.
    filename_table : str, optional
     Used to import a table and skip the calculation
    recalculate_table : bool, default False
     Parameter to either calculate the table or to check if a table is already given

    Returns
    ------
    dict of {str:dict}
        Dictionary containing two sub-dictionaries: The first one containing the function variables and the second one
        containing the 12 tables for the integrals
    '''
    if filename_table is None: filename_table = 'qwerty'
    if filename_table is not None and not recalculate_table:
        if len(glob(filename_table)):
            Gamma_input_info = pickle.load(open(filename_table, 'rb'))
            print('Table read in.')
        else:
            recalculate_table = True
    else:
        recalculate_table = True

    if recalculate_table:
        print('Creating table...')

        L  = 1.38 * 10 ** 37 * M
        Ag = L / (4 * pi * facr ** 2 * (integrate.quad(lambda x: x ** -alpha, E_0, E_upp)[0]))

        r_min = r_grid[0]
        lgdr  = r_grid[1] - r_grid[0]

        # Spectral energy function, power law for a quasar source
        def I(E):
            if sed is not None: return sed(E)
            miniqsos = E ** -alpha
            return Ag * miniqsos

        # Radiation flux times unit distance
        def N(E, n_HI0, n_HeI0, n_HeII0):
            int = (10 ** r_min) * facr * (10 ** lgdr - 1) * (n_HI0 * sigma_HI(E) + n_HeI0 * sigma_HeI(E) + n_HeII0 * sigma_HeII(E))
            return exp(-int) * I(E)

        IHI_1     = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHI_2     = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHI_3     = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IHeI_1    = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHeI_2    = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHeI_3    = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IHeII     = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IT_HI_1   = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IT_HeI_1  = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IT_HeII_1 = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IT_2a     = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IT_2b     = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        for k2 in tqdm(range(0, n_HI.size, 1)):
            for k3 in range(0, n_HeI.size, 1):
                for k4 in range(0, n_HeII.size, 1):

                    IHI_1[k2, k3, k4]     = integrate.quad(lambda x: sigma_HI(x) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HI, E_upp)[0]

                    IHI_2[k2, k3, k4]     = integrate.quad(lambda x: sigma_HI(x) * (x - E_HI) / (E_HI * x) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HI,E_upp)[0]
                    IHI_3[k2, k3, k4]     = integrate.quad(lambda x: sigma_HeI(x) * (x - E_HeI) / (x * E_HI) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeI, E_upp)[0]

                    IHeI_1[k2, k3, k4]    = integrate.quad(lambda x: sigma_HeI(x) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HeI, E_upp)[0]
                    IHeI_2[k2, k3, k4]    = integrate.quad(lambda x: sigma_HeI(x) * (x - E_HeI) / (x * E_HeI) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeI, E_upp)[0]
                    IHeI_3[k2, k3, k4]    = integrate.quad(lambda x: sigma_HI(x) * (x - E_HI) / (x * E_HeI) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HeI, E_upp)[0]

                    IHeII[k2, k3, k4]     = integrate.quad(lambda x: sigma_HeII(x) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HeII, E_upp)[0]

                    IT_HI_1[k2, k3, k4]   = integrate.quad(lambda x: sigma_HI(x) * (x - E_HI) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HI,E_upp)[0]
                    IT_HeI_1[k2, k3, k4]  = integrate.quad(lambda x: sigma_HeI(x) * (x - E_HeI) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeI, E_upp)[0]
                    IT_HeII_1[k2, k3, k4] = integrate.quad(lambda x: sigma_HeII(x) * (x - E_HeII) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeII, E_upp)[0]

                    IT_2a[k2, k3, k4]     = integrate.quad(lambda x: N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]) * x, E_0, E_upp)[0]
                    IT_2b[k2, k3, k4]     = integrate.quad(lambda x: N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]) * (-4 * kb.value), E_0, E_upp)[0]

        print('...done')

        Gamma_info = {'HI_1': IHI_1, 'HI_2': IHI_2, 'HI_3': IHI_3,
                      'HeI_1': IHeI_1, 'HeI_2': IHeI_2, 'HeI_3': IHeI_3, 'HeII': IHeII,
                      'T_HI_1': IT_HI_1, 'T_HeI_1': IT_HeI_1, 'T_HeII_1': IT_HeII_1,
                      'T_2a': IT_2a, 'T_2b': IT_2b}

        input_info = {'M': M, 'z': z,
                      'r_grid': r_grid,
                      'n_HI': n_HI, 'n_HeI': n_HeI, 'n_HeII': n_HeII}

        Gamma_input_info = {'Gamma': Gamma_info, 'input': input_info}

    return Gamma_input_info

# Generate the 3-D table in order to interpolate the 12 integrals in the radiative transfer equations.
# This table has an upper limit such that UV photons are not included.
def generate_table2(M, z, r_grid, n_HI, n_HeI, n_HeII,alpha,sed, filename_table=None, recalculate_table=False):
    """
    Generate the interpolation tables for the integrals for the radiative transfer equations.

    This function generates 3-D tables for each integral in the radiative transfer equation. The elements of the tables
    are the values of the integrals for a given set of variables (n_HI, n_HeI, n_HeII). This is done by initializing a
    array for each variable within a specified range and then loop over the arrays and use each set of variables to
    evaluate the integrals. The integral range is set up to include UV ionizing photons.

    Parameters
    ----------
    M : float
     Mass of the source, given in solar masses.
    z : float
     Redshift of the source.
    r_grid : array_like
     Radial Mpc grid in logspace from the source to a given distance, the starting distance is typically 0.0001 Mpc from
     the source.
    n_HI : array_like
     Neutral hydrogen density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    n_HeI : array_like
     Neutral helium density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    n_HI : array_like
     HeII density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    alpha : int, default -1
     Spectral index for a power-law source.
    sed : callable, optional
     Spectral energy distribution to be used instead of the default power-law function. sed is a function of energy.
    filename_table : str, optional
     Used to import a table and skip the calculation.
    recalculate_table : bool, default False
     Parameter to either calculate the table or to check if a table is already given.

    Returns
    ------
    dict of {str:dict}
        Dictionary containing two sub-dictionaries: The first dictionary containing the 12 tables for the integrals and
        the second one containing the input variables.
    """
    if filename_table is None: filename_table = 'qwerty'
    if filename_table is not None and not recalculate_table:
        if len(glob(filename_table)):
            Gamma_input_info = pickle.load(open(filename_table, 'rb'))
            print('Table read in.')
        else:
            recalculate_table = True
    else:
        recalculate_table = True

    if recalculate_table:
        print('Creating table...')

        L  = 1.38 * 10 ** 37 * M  # eddington luminosity, eps = 10%
        Ag = L / (4 * pi * facr ** 2 * (integrate.quad(lambda x: x ** -alpha, E_0, E_cut)[0]))

        r_min = r_grid[0]
        lgdr  = r_grid[1] - r_grid[0]

        def I(E):  # spectral energy distribution
            if sed is not None: return sed(E)
            miniqsos = E ** -alpha  # power law source
            return Ag * miniqsos

        def N(E, n_HI0, n_HeI0,n_HeII0):  # this N is not that from the paper; the factor 1/r**2 is missing, this is done for computational purposes

            int = ((10 ** (r_min))) * facr * (10 ** lgdr - 1) * (n_HI0 * sigma_HI(E) + n_HeI0 * sigma_HeI(E) + n_HeII0 * sigma_HeII(E)).value

            return exp(-int) * I(E)

        IHI_1 = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHI_2 = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHI_3 = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IHeI_1 = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHeI_2    = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IHeI_3    = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IHeII     = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IT_HI_1   = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IT_HeI_1  = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IT_HeII_1 = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        IT_2a     = zeros((n_HI.size, n_HeI.size, n_HeII.size))
        IT_2b     = zeros((n_HI.size, n_HeI.size, n_HeII.size))

        for k2 in tqdm(range(0, n_HI.size, 1)):
            for k3 in range(0, n_HeI.size, 1):
                for k4 in range(0, n_HeII.size, 1):

                    IHI_1[k2, k3, k4]     = integrate.quad(lambda x: sigma_HI(x) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HI, E_cut)[0]

                    IHI_2[k2, k3, k4]     = integrate.quad(lambda x: sigma_HI(x) * (x - E_HI) / (E_HI * x) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HI,E_cut)[0]
                    IHI_3[k2, k3, k4]     = integrate.quad(lambda x: sigma_HeI(x) * (x - E_HeI) / (x * E_HI) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeI, E_cut)[0]

                    IHeI_1[k2, k3, k4]    = integrate.quad(lambda x: sigma_HeI(x) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HeI, E_cut)[0]
                    IHeI_2[k2, k3, k4]    = integrate.quad(lambda x: sigma_HeI(x) * (x - E_HeI) / (x * E_HeI) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeI, E_cut)[0]
                    IHeI_3[k2, k3, k4]    = integrate.quad(lambda x: sigma_HI(x) * (x - E_HI) / (x * E_HeI) * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HeI,E_cut)[0]

                    IHeII[k2, k3, k4]     = integrate.quad(lambda x: sigma_HeII(x) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HeII, E_cut)[0]

                    IT_HI_1[k2, k3, k4]   = integrate.quad(lambda x: sigma_HI(x) * (x - E_HI) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]), E_HI,E_cut)[0]
                    IT_HeI_1[k2, k3, k4]  = integrate.quad(lambda x: sigma_HeI(x) * (x - E_HeI) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeI, E_cut)[0]
                    IT_HeII_1[k2, k3, k4] = integrate.quad(lambda x: sigma_HeII(x) * (x - E_HeII) / x * N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]),E_HeII, E_cut)[0]

                    IT_2a[k2, k3, k4]     = integrate.quad(lambda x: N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]) * x, E_0, E_cut)[0]
                    IT_2b[k2, k3, k4]     = integrate.quad(lambda x: N(x, n_HI[k2], n_HeI[k3], n_HeII[k4]) * (-4 * kb.value), E_0, E_cut)[0]

        print('...done')

        Gamma_info = {'HI_1': IHI_1, 'HI_2': IHI_2, 'HI_3': IHI_3,
                      'HeI_1': IHeI_1, 'HeI_2': IHeI_2, 'HeI_3': IHeI_3, 'HeII': IHeII,
                      'T_HI_1': IT_HI_1, 'T_HeI_1': IT_HeI_1, 'T_HeII_1': IT_HeII_1,
                      'T_2a': IT_2a, 'T_2b': IT_2b}

        input_info = {'M': M, 'z': z,
                      'r_grid': r_grid, 'dr_init': dr_init,
                      'n_HI': n_HI, 'n_HeI': n_HeI, 'n_HeII': n_HeII}

        Gamma_input_info = {'Gamma': Gamma_info, 'input': input_info}

    return Gamma_input_info


class Source:
    """
    Source which ionizes the surrounding H and He gas along a radial direction.

    This class initiates a source with a given mass, redshift and a default power law source, which can be changed to
    any spectral energy distribution. Given a starting point and a ending point, the radiative transfer equations are
    solved along the radial direction and the H and He densities are evolved for some given time.

    Parameters
    ----------
    M : float
     Mass of the source in solar masses.
    z : float
     Redshift of the source.
    evol : float
     Evolution time of the densities in Myr.
    r_start : float, optional
     Starting point of the radial Mpc-grid in logspace, default value is log10(0.0001).
    r_end : float, optional
     Ending point of the radial Mpc-grid in logspace, default value is log10(3).
    LE : bool, optional
     Decide whether to include UV ionizing photons, default is True (include).
    alpha : int, optional
     Spectral index for the power law source, default is -1.
    sed : callable, optional
     Spectral energy distribution. Default is a power law source.
    lifetime : float, optional
     Lifetime of the source, after which it will be turned off and the radiative transfer will continue to be calculated
     without a source
    filename_table : str, optional
     Name of the external table, will be imported if available to skip the table generation
    recalculate_table : bool, default False
     Decide whether to import or generate the interpolation table. If nothing is given it will be set to False and then
     be changed whether or not a external table is available.
    """
    def __init__(self, M, z, evol, r_start=None, r_end=None, LE=None, alpha=None,sed = None, lifetime = None, filename_table=None, recalculate_table=False):


        self.M     = M  # mass of the quasar
        self.z     = z  # redshift
        self.evol  = (evol * u.Myr).to(u.s)  # evolution time, typically 3-10 Myr
        self.LE    = LE if LE is not None else True
        self.lifetime = lifetime

        self.alpha = alpha if alpha is not None else 1.
        self.sed   = sed

        self.r_start = r_start * u.Mpc if r_start is not None else 0.0001 * u.Mpc  # starting point from source
        self.r_end   = r_end * u.Mpc if r_end is not None else 3 * u.Mpc  # maximal distance from source


        self.filename_table    = filename_table
        self.recalculate_table = recalculate_table
        self.Gamma_grid_info   = None

    def create_table(self, par=None, filename=None):
        """
        Call the function to create the interpolation tables.

        Parameters
        ----------
        par : dict of {str:float}, optional
         Variables to pass on to the table generator. If none is given the parameters of the Source initialization will
         be used.
        """
        if par is None:
            M, z_reion = self.M, self.z
            alpha, sed = self.alpha, self.sed
            dn = self.grid_param['dn']

            
            r_grid = log10(logspace(log10(self.r_start.value), log10(self.r_end.value), dn))

            N      = r_grid.size
            n_HI   = linspace(0, 10000*(N - 1) * n_H(z_reion).value, 20)
            n_HeI  = linspace(0, 10000*(N - 1) * n_He(z_reion).value, 20)
            n_HeII = linspace(0, 10000*(N - 1) * n_He(z_reion).value, 20)

        else:
            M, z_reion          = par['M'], par['z_reion']
            r_grid              = par['r_grid']
            n_HI, n_HeI, n_HeII = par['n_HI'], par['n_HeI'], par['n_HeII']
            alpha, sed          = par['alpha'], par['sed']

        if self.LE:
            Gamma_grid_info = generate_table(M, z_reion, r_grid, n_HI, n_HeI, n_HeII,alpha,sed)
        else:
            Gamma_grid_info = generate_table2(M, z_reion, r_grid, n_HI, n_HeI, n_HeII,alpha,sed)

        self.Gamma_grid_info = Gamma_grid_info

    def initialise_grid_param(self):
        """
        Initialize the grid parameters for the solver.

        The grid parameters and the initial conditions are saved in a dictionary. A initial time step size and a initial
         radial grid is chosen.T_gamma is the CMB temperature for the given redshift and gamma_2c is used to calculate
         the contribution from the background photons.
        """
        grid_param = {'M': self.M, 'z_reion': self.z}

        T_gamma  = 2.725 * (1 + grid_param['z_reion']) * u.K
        gamma_2c = (alpha_HII(T_gamma.value) * (m_e * kb * T_gamma / (2 * pi)) ** (3 / 2) / h ** 3 * exp(-(3.4 * u.eV).to(u.erg) / (T_gamma * kb))).decompose()

        grid_param['T_gamma']  = T_gamma
        grid_param['gamma_2c'] = gamma_2c

        dt_init = self.evol/25  # time interval
        t_life  = self.evol.value

        dn      = 50
        r_start = self.r_start
        r_end   = self.r_end

        grid_param['dn']      = dn
        grid_param['r_start'] = r_start
        grid_param['r_end']   = r_end
        grid_param['t_life']  = t_life
        grid_param['dt_init'] = dt_init

        self.grid_param = grid_param

    def solve(self):
        """
        Solves the radiative transfer equation for the given source and the parameters.

        The solver calls grid parameters and initializes the starting grid with the initial conditions for the densities
        and the temperature. Using the time step, the radiative transfer equations are used to update the k-th cell from
        the radial grid for a certain time step dt_init. Then the solver moves on to the (k+1)-th cell, and uses the
        values calculated from the k-th cell in order to calculate the optical depth, which requires information of the
        densities from all prior cells. For each cell, we sum up the three densities from the starting cell up to the
        current cell and use these values to evaluate the 12 integrals in the equations, which is done by interpolation
        of the tables we generated previously. After each cell is updated for some time dt_init, we start again with the
        first cell and use the calculation from the previous time step as the initial condition and repeat the same
        process until the radial cells are updated l times such that l*dt_init has reached the evolution time. After the
        solver is finished we compare the ionization fronts of two consecutive runs and require an accuracy of 5% in
        order to finish the calculations. If the accuracy is not reached we store the values from the run and start
        again with time step size dt_init/2 and a radial grid with half the step size from the previous run.
        This process is repeated until the desired accuracy is reached.
        """
        print('Solving the radiative equations...')
        self.initialise_grid_param()
        z_reion  = self.grid_param['z_reion']
        gamma_2c = self.grid_param['gamma_2c']
        T_gamma = self.grid_param['T_gamma']

        dt_init  = self.grid_param['dt_init']
        dn       = self.grid_param['dn']

        r_grid0 = log10(logspace(log10(self.grid_param['r_start'].value), log10(self.grid_param['r_end'].value), dn))

        n_HII_grid   = zeros_like(r_grid0) * cm_3
        n_HeII_grid  = zeros_like(r_grid0) * cm_3
        n_HeIII_grid = zeros_like(r_grid0) * cm_3

        while True:

            time_grid = []
            Ion_front_grid = []

            dn = self.grid_param['dn']

            r_grid = log10(logspace(log10(self.grid_param['r_start'].value), log10(self.grid_param['r_end'].value), dn))

            n_HII0       = copy(n_HII_grid[:]) * cm_3
            n_HeII0      = copy(n_HeII_grid[:]) * cm_3
            n_HeIII0     = copy(n_HeIII_grid[:]) * cm_3
            n_HII_grid   = zeros_like(r_grid) * cm_3
            n_HeII_grid  = zeros_like(r_grid) * cm_3
            n_HeIII_grid = zeros_like(r_grid) * cm_3

            T_grid  = zeros_like(r_grid) * u.K
            T_grid += T_gamma * (1 + z_reion) ** 1 / (1 + 250)

            l = 0

            print('Number of time steps: ', int(math.ceil(self.grid_param['t_life'] / dt_init.value)))

            N      = r_grid.size
            n_HI   = linspace(0, 10000*(N - 1) * n_H(z_reion), 20)
            n_HeI  = linspace(0, 10000*(N - 1) * n_He(z_reion), 20)
            n_HeII = linspace(0, 10000*(N - 1) * n_He(z_reion), 20)
            points = (n_HI, n_HeI, n_HeII)

            self.create_table()

            Gamma_info = self.Gamma_grid_info['Gamma']

            JHI_1, JHI_2, JHI_3           = Gamma_info['HI_1'], Gamma_info['HI_2'], Gamma_info['HI_3']
            JHeI_1, JHeI_2, JHeI_3, JHeII = Gamma_info['HeI_1'], Gamma_info['HeI_2'], Gamma_info['HeI_3'], Gamma_info['HeII']
            JT_HI_1, JT_HeI_1, JT_HeII_1  = Gamma_info['T_HI_1'], Gamma_info['T_HeI_1'], Gamma_info['T_HeII_1']
            JT_2a, JT_2b                  = Gamma_info['T_2a'], Gamma_info['T_2b']

            while l * self.grid_param['dt_init'].value <= self.grid_param['t_life']:

                print('Current Time step: ', l)

                # Calculate the redshift z(t)
                age = pl.age(z_reion)
                age = age.to(u.s)
                age += l * self.grid_param['dt_init']
                func = lambda z: pl.age(z).to(u.s).value - age.value
                zstar = fsolve(func, z_reion)

                # Initialize the values to evaluate the integrals
                K_HI = 0 * cm_3
                K_HeI = 0 * cm_3
                K_HeII = 0 * cm_3

                for k in (arange(0, r_grid.size, 1)):

                    lgdr = r_grid[1] - r_grid[0]

                    if k > 0:
                        n_HI00 = n_H(z_reion) - n_HII_grid[k - 1]

                        if n_HI00 < 0:
                            n_HI00 = 0
                        n_HeI00 = n_He(z_reion) - n_HeII_grid[k - 1] - n_HeIII_grid[k - 1]

                        if n_HeI00 < 0:
                            n_HeI00 = 0 * cm_3

                        K_HI += nan_to_num(n_HI00) * 10 ** (lgdr * (k - 1))
                        K_HeI += nan_to_num(n_HeI00) * 10 ** (lgdr * (k - 1))
                        K_HeII += abs(nan_to_num(n_HeII_grid[k - 1])) * 10 ** (lgdr * (k - 1))


                    if self.lifetime is not None and l * self.grid_param['dt_init'].value > ((lifetime * u.Myr).to(u.s)).value:

                        I1_HI = 0 * s1
                        I2_HI = 0 * s1
                        I3_HI = 0 * s1

                        I1_HeI = 0 * s1
                        I2_HeI = 0 * s1
                        I3_HeI = 0 * s1

                        I1_HeII = 0 * s1

                        I1_T_HI = 0 * s1
                        I1_T_HeI = 0 * s1
                        I1_T_HeII = 0 * s1

                        I2_Ta = 0 * s1
                        I2_Tb = 0 * s1

                    else:

                        r2 = (4 * pi * (10 ** r_grid[k]) ** 2)

                        I1_HI = interpolate.interpn(points, JHI_1, (K_HI, K_HeI, K_HeII), method='linear') / r2
                        I2_HI = interpolate.interpn(points, JHI_2, (K_HI, K_HeI, K_HeII), method='linear') / r2
                        I3_HI = interpolate.interpn(points, JHI_3, (K_HI, K_HeI, K_HeII), method='linear') / r2

                        I1_HeI = interpolate.interpn(points, JHeI_1, (K_HI, K_HeI, K_HeII), method='linear') / r2
                        I2_HeI = interpolate.interpn(points, JHeI_2, (K_HI, K_HeI, K_HeII), method='linear') / r2
                        I3_HeI = interpolate.interpn(points, JHeI_3, (K_HI, K_HeI, K_HeII), method='linear') / r2

                        I1_HeII = interpolate.interpn(points, JHeII, (K_HI, K_HeI, K_HeII), method='linear') / r2

                        I1_T_HI   = interpolate.interpn(points, JT_HI_1, (K_HI, K_HeI, K_HeII), method='linear') / r2
                        I1_T_HeI  = interpolate.interpn(points, JT_HeI_1, (K_HI, K_HeI, K_HeII), method='linear') / r2
                        I1_T_HeII = interpolate.interpn(points, JT_HeII_1, (K_HI, K_HeI, K_HeII), method='linear') / r2

                        I2_Ta = interpolate.interpn(points, JT_2a, (K_HI, K_HeI, K_HeII), method='linear') / r2
                        I2_Tb = interpolate.interpn(points, JT_2b, (K_HI, K_HeI, K_HeII), method='linear') / r2

                    def gamma_HI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx, Tx):
                        """
                        Calculate gamma_HI given the densities and the temperature

                        Parameters
                        ----------
                        n_HIIx : float
                         Ionized hydrogen density in cm**-3.
                        n_HeIx : float
                         Neutral helium density in cm**-3.
                        n_HeIIx : float
                         Single ionized helium density in cm**-3.
                        n_HeIIIx : float
                         Double ionized helium density in cm**-3.
                        Tx : float
                         Temperature of the gas in K.

                        Returns
                        -------
                        float
                         Gamma_HI for the radiative transfer equation.
                        """
                        n_e = n_HIIx + n_HeIIx + 2 * n_HeIIIx

                        if n_H(z_reion).value == n_HIIx or n_He(z_reion).value - n_HeIIx - n_HeIIIx == 0:
                            factor = 0
                        else:
                            factor = abs((n_He(z_reion).value - n_HeIIx - n_HeIIIx) / (n_H(z_reion).value - n_HIIx))

                        return (gamma_2c.value + beta_HI(Tx) * n_e + I1_HI + f_H(n_HIIx, z_reion) * I2_HI + f_H(n_HIIx, z_reion) * factor * I3_HI)

                    def gamma_HeI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx):
                        """
                        Calculate gamma_HeI given the densities and the temperature

                        Parameters
                        ----------
                        n_HIIx : float
                         Ionized hydrogen density in cm**-3.
                        n_HeIx : float
                         Neutral helium density in cm**-3.
                        n_HeIIx : float
                         Single ionized helium density in cm**-3.
                        n_HeIIIx : float
                         Double ionized helium density in cm**-3.

                        Returns
                        -------
                        float
                         Gamma_HeI for the radiative transfer equation.
                        """
                        if n_H(z_reion).value == n_HIIx or n_He(z_reion).value - n_HeIIx - n_HeIIIx == 0:
                            factor = 0
                        else:
                            factor = abs(nan_to_num(n_H(z_reion).value - n_HIIx) / (n_He(z_reion).value - n_HeIIx - n_HeIIIx))

                        return (I1_HeI + f_He(n_HIIx, z_reion) * I2_HeI + f_He(n_HIIx, z_reion) * factor * I3_HeI)

                    def gamma_HeII(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx):
                        """
                        Calculate gamma_HeII given the densities and the temperature

                        Parameters
                        ----------
                        n_HIIx : float
                         Ionized hydrogen density in cm**-3.
                        n_HeIx : float
                         Neutral helium density in cm**-3.
                        n_HeIIx : float
                         Single ionized helium density in cm**-3.
                        n_HeIIIx : float
                         Double ionized helium density in cm**-3.

                        Returns
                        -------
                        float
                         Gamma_HeII for the radiative transfer equation.
                        """
                        return I1_HeII

                    def rhs(t,n):
                        """
                        Calculate the RHS of the radiative transfer equations.

                        RHS of the coupled nHII, n_HeII, n_HeIII and T equations. The equations are labelled as A,B,C,
                        and D and the rest of the variables are the terms contained in the respective equations.

                        Parameters
                        ----------
                        t : float
                         Time of evaluation in s.
                        n : array-like
                         1-D array containing the variables nHII, nHeII, nHeIII, T for evaluating the RHS.

                        Returns
                        -------
                        array_like
                         The RHS of the radiative transfer equations.
                        """
                        n_HIx = n_H(z_reion).value - n[0]

                        n_HIIx = n[0]
                        if n_HIx < 0:
                            n_HIx = 0
                            n_HIIx = n_H(z_reion).value
                        if n_HIIx > n_H(z_reion).value:
                            n_HIIx = n_H(z_reion).value
                            n_HIx = 0
                        if n_HIIx < 0:
                            n_HIIx = 0
                            n_HIx = n_H(z_reion).value

                        n_HeIx   = n_He(z_reion).value - (n[1] + n[2])
                        n_HeIIx  = n[1]
                        n_HeIIIx = n[2]

                        if n_HeIx > n_He(z_reion).value:
                            n_HeIx   = n_He(z_reion).value
                            n_HeIIx  = 0
                            n_HeIIIx = 0
                        if n_HeIx < 0:
                            n_HeIx = 0

                        if n_HeIIx > n_He(z_reion).value:
                            n_HeIIx  = n_He(z_reion).value
                            n_HeIx   = 0
                            n_HeIIIx = 0
                        if n_HeIIx < 0:
                            n_HeIIx = 0

                        if n_HeIIIx > n_He(z_reion).value:
                            n_HeIIIx = n_He(z_reion).value
                            n_HeIIx  = 0
                            n_HeIx   = 0
                        if n_HeIIIx < 0:
                            n_HeIIIx = 0

                        Tx = n[3]


                        n_ee = n_HIIx + n_HeIIx + 2 * n_HeIIIx

                        mu  = (n_H(z_reion).value + 4 * n_He(z_reion).value) / (n_H(z_reion).value + n_He(z_reion).value + n_ee)
                        n_B = n_H(z_reion).value + n_He(z_reion).value + n_ee

                        A1_HI    = xi_HI(Tx) * n_HIx * n_ee
                        A1_HeI   = xi_HeI(Tx) * n_HeIx * n_ee
                        A1_HeII  = xi_HeII(Tx) * n_HeIIx * n_ee
                        A2_HII   = eta_HII(Tx) * n_HIIx * n_ee
                        A2_HeII  = eta_HeII(Tx) * n_HeIIx * n_ee
                        A2_HeIII = eta_HeIII(Tx) * n_HeIIIx * n_ee

                        A3 = omega_HeII(Tx) * n_ee * n_HeIIIx

                        A4_HI   = psi_HI(Tx) * n_HIx * n_ee
                        A4_HeI  = psi_HeI(Tx, n_ee, n_HeIIx) * n_ee
                        A4_HeII = psi_HeII(Tx) * n_HeIIx * n_ee

                        A5 = theta_ff(Tx) * (n_HIIx + n_HeIIx + 4 * n_HeIIIx) * n_ee

                        H = pl.H(zstar)
                        H = H.to(u.s ** -1)

                        A6 = (2 * H * kb * Tx*u.K * n_B / mu).value

                        A = gamma_HI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx, Tx) * n_HIx - alpha_HII(
                            Tx) * n_HIIx * n_ee
                        B = gamma_HeI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx) * n_HeIx + beta_HeI(
                            Tx) * n_ee * n_HeIx - beta_HeII(Tx) * n_ee * n_HeIIx - alpha_HeII(
                            Tx) * n_ee * n_HeIIx + alpha_HeIII(Tx) * n_ee * n_HeIIIx - zeta_HeII(
                            Tx) * n_ee * n_HeIIx
                        C = gamma_HeII(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx) * n_HeIIx + beta_HeII(
                            Tx) * n_ee * n_HeIIx - alpha_HeIII(Tx) * n_ee * n_HeIIIx
                        Dd = (Tx / mu) * (-mu / (n_H(z_reion).value + n_He(z_reion).value + n_ee)) * (A + B + 2 * C)
                        D  = (2 / 3) * mu / (kb.value * n_B) * (f_Heat(n_HIIx, z_reion) * n_HIx * I1_T_HI + f_Heat(n_HIIx,
                                                                                                            z_reion) * n_HeIx * I1_T_HeI + f_Heat(
                            n_HIIx, z_reion) * n_HeIIx * I1_T_HeII + sigma_s.value * n_ee / (m_e * c ** 2).value * (
                                                                     I2_Ta + Tx  * I2_Tb) - (
                                                                     A1_HI + A1_HeI + A1_HeII + A2_HII + A2_HeII + A2_HeIII + A3 + A4_HI + A4_HeI + A4_HeII + A5 + A6)) + Dd

                        return ravel(array([A, B, C, D]))

                    y0    = zeros(4)
                    y0[0] = n_HII_grid[k].value
                    y0[1] = n_HeII_grid[k].value
                    y0[2] = n_HeIII_grid[k].value
                    y0[3] = T_grid[k].value
                    sol   = integrate.solve_ivp(rhs, [l * dt_init.value, (l + 1) * dt_init.value], y0, method = 'LSODA')

                    n_HII_grid[k] = sol.y[0, -1] * cm_3

                    if n_HII_grid[k] > n_H(z_reion):
                        n_HII_grid[k] = n_H(z_reion)

                    n_HeII_grid[k]  = sol.y[1, -1] * cm_3

                    n_HeIII_grid[k] = sol.y[2, -1] * cm_3

                    if n_HeIII_grid[k] > n_He(z_reion):
                        n_HeIII_grid[k] = n_He(z_reion)
                    if n_He(z_reion) - n_HeII_grid[k] - n_HeIII_grid[k] < 0:
                        n_HeII_grid[k] = n_He(z_reion) - n_HeIII_grid[k]
                    T_grid[k] = sol.y[3, -1] * u.K

                time_grid.append(l * self.grid_param['dt_init'].value)
                Ion_front_grid.append(find_Ifront(n_HII_grid, r_grid, z_reion))
                l += 1

            r1 = find_Ifront(n_HII0, 10 ** r_grid0, z_reion, show=True)
            r2 = find_Ifront(n_HII_grid, 10 ** r_grid, z_reion, show=True)

            print('The accuracy is: ', abs((r1 - r2) / min(abs(r1), abs(r2))), ' -> 0.05 needed')
            if abs((r1 - r2) / min(abs(r1), abs(r2))) > 0.05 or r2 == self.r_start.value:
                if r2 == self.r_start.value:
                    print('Ionization front is still at the starting point. Starting again with smaller steps...')
                r_grid0 = log10(logspace(log10(self.grid_param['r_start'].value), log10(self.grid_param['r_end'].value), dn))
                self.grid_param['dn'] *= 2
                self.grid_param['dt_init'] /= 2

            else:
                print('... radiative equations solved')
                break
        self.n_HI_grid = n_H(self.z) - n_HII_grid
        self.n_HII_grid = n_HII_grid
        self.n_HeI_grid = n_He(self.z) - n_HeII_grid - n_HeIII_grid
        self.n_HeII_grid = n_HeII_grid
        self.n_HeIII_grid = n_HeIII_grid
        self.n_H = n_H(self.z)
        self.n_He = n_He(self.z)
        self.T_grid = T_grid
        self.r_grid = r_grid
        self.time_grid = array([time_grid])
        self.Ion_front_grid = array([Ion_front_grid])

    def r(self):
        return 10**self.r_grid

    def nHI(self):
        return self.n_HI_grid

    def nHII(self):
        return self.n_HII_grid

    def nHeI(self):
        return self.n_HeI_grid

    def nHeII(self):
        return self.n_HeII_grid

    def nHeIII(self):
        return self.n_HeIII_grid

    def T(self):
        return self.T_grid

    def nH(self):
        return self.n_H

    def nHe(self):
        return self.n_He

    def xHI(self):
        return self.n_HI_grid / self.n_H

    def xHII(self):
        return self.n_HII_grid / self.n_H

    def xHeI(self):
        return self.n_HeI_grid / self.n_He

    def xHeII(self):
        return self.n_HeII_grid / self.n_He

    def xHeIII(self):
        return self.n_HeIII_grid / self.n_He

    def Ifront(self):  # return the ionization front of the source, the distance where n_HII = n_HI
        m = 0
        x = self.n_HII_grid / n_H(self.z)
        m = argmin(abs(0.5 - x))

        print('Pos. of Ifront:', self.r_grid[m])
        return self.r_grid[m]





