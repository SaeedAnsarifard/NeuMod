import numpy as np

# Global constants
FERMI_CONSTANT = 1.166  # e-11 MeV^-2
ELECTRON_MASS = 0.511  # MeV
HBAR_C = 1.97  # e-11 MeV cm

# Load data
phi = np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack=True)[6, :]
n_e = 6 * 10**np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack=True)[2, :]   # 1e23 cm^-3

def PseudoDirac(param, ls, enu):
    """
    Calculate the survival probabilities for pseudo-Dirac neutrinos.
    
    Parameters:
    - param: Dictionary containing the physical parameters ('M12', 'T12', 'T13', 'mum1', 'mum2', 'mum3').
    - ls: Array of solar distances in AU.
    - enu: Array of neutrino energies in MeV.
    
    Returns:
    - pel: Electron neutrino survival probabilities.
    - psl: Sterile neutrino survival probabilities.
    """
    ls_meters = 1.496e11 * ls  # Convert AU to meters
    pel = np.zeros((len(ls), len(enu)))
    psl = np.zeros((len(ls), len(enu)))

    util = np.ones((len(n_e), len(enu)))
    ne = n_e[:, np.newaxis] * util
    e = enu[np.newaxis, :] * util

    ve = 2 * np.sqrt(2) * FERMI_CONSTANT * ne * HBAR_C**3 * 1e-9 * e
    cos_2theta12 = np.cos(np.radians(param['T12']) * 2)
    sin_2theta12 = np.sin(np.radians(param['T12']) * 2)
    cos_2theta13 = np.cos(np.radians(param['T13']))**4

    den = np.sqrt((param['M12'] * cos_2theta12 - ve)**2 + (param['M12'] * sin_2theta12)**2)
    nom = param['M12'] * cos_2theta12 - ve
    tm = 0.5 * np.arccos(nom / den)

    sin_theta12_2 = np.sin(np.radians(param['T12']))**2
    cos_theta12_2 = np.cos(np.radians(param['T12']))**2
    sin_theta13_4 = np.sin(np.radians(param['T13']))**4

    for j, l in enumerate(ls_meters):
        ae1 = cos_theta12_2 * cos_2theta13 * np.cos(tm)**2 * np.cos(10 * param['mum1'] * l / (HBAR_C * 2 * e))**2
        ae2 = sin_theta12_2 * cos_2theta13 * np.sin(tm)**2 * np.cos(10 * param['mum2'] * l / (HBAR_C * 2 * e))**2
        ae3 = sin_theta13_4 * np.cos(10 * param['mum3'] * l / (HBAR_C * 2 * e))**2

        pel[j] = np.sum(phi[:, np.newaxis] * (ae1 + ae2 + ae3), axis=0)

        as1 = cos_2theta13 * np.cos(tm)**2 * np.sin(10 * param['mum1'] * l / (HBAR_C * 2 * e))**2
        as2 = cos_2theta13 * np.sin(tm)**2 * np.sin(10 * param['mum2'] * l / (HBAR_C * 2 * e))**2
        as3 = sin_theta13_4 * np.sin(10 * param['mum3'] * l / (HBAR_C * 2 * e))**2

        psl[j] = np.sum(phi[:, np.newaxis] * (as1 + as2 + as3), axis=0)

    return pel, psl
