import numpy as np
from datetime import datetime
from skyfield.api import load, utc

# Global constants
FERMI_CONSTANT = 1.166  # e-11 MeV^-2
ELECTRON_MASS = 0.511  # MeV
HBAR_C = 1.97  # e-11 MeV cm
LIGHT_SPEED = 2.998 # 1e8 m/s
RHO_DM2  = np.sqrt(2 * 0.4 * 7.65) #e-21 GeV^2
ECCENTRICITY = (1.521 - 1.471)/(1.521 + 1.471)


time_scale = load.timescale()  # Create a timescale object

# Load data
phi = np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack=True)[6, :]
n_e = 6 * 10**np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack=True)[2, :]   # 1e23 cm^-3


def MSW(param, enu):
    """
    Calculate the survival probabilities for MSW neutrinos.
    
    Parameters:
    - param: Dictionary containing the physical parameters ('M12', 'SinT12', 'T13').
    - enu: Array of neutrino energies in MeV.
    
    Returns:
    - pee: Electron neutrino survival probabilities.
    """
        
    util = np.ones((len(n_e), len(enu)))
    ne = n_e[:, np.newaxis] * util
    e = enu[np.newaxis, :] * util

    ve = 2 * np.sqrt(2) * FERMI_CONSTANT * ne * HBAR_C**3 * 1e-9 * e
        
    sin_theta12_2 = param['SinT12']
    cos_theta12_2 = 1 - sin_theta12_2
    
    cos_2theta12 = 1 - 2 * sin_theta12_2
    sin_2theta12 = 2 * np.sqrt(param['SinT12'] -  param['SinT12']**2)
    
    sin_theta13 = np.sin(np.radians(param['T13']))
    cos_theta13 = np.cos(np.radians(param['T13']))
    
    den = np.sqrt((param['M12'] * cos_2theta12 - ve)**2 + (param['M12'] * sin_2theta12)**2)
    nom = param['M12'] * cos_2theta12 - ve
    
    tm = 0.5 * np.arccos(nom / den)
  
    cos_tm_distributed = np.sum(phi[:, np.newaxis] * np.cos(tm)**2, axis=0)
    sin_tm_distributed = np.sum(phi[:, np.newaxis] * np.sin(tm)**2, axis=0)
    
    ae1 = cos_theta13**4 * cos_theta12_2 * cos_tm_distributed
    ae2 = cos_theta13**4 * sin_theta12_2 * sin_tm_distributed
    ae3 = sin_theta13**4
    
    return ae1 + ae2 + ae3


def PseudoDirac(param, ls, enu):
    """
    Calculate the survival probabilities for pseudo-Dirac neutrinos.
    
    Parameters:
    - param: Dictionary containing the physical parameters ('M12', 'SinT12', 'T13', 'mum1', 'mum2', 'mum3').
    - ls: Array of solar distances in AU.
    - enu: Array of neutrino energies in MeV.
    
    Returns:
    - pel: Electron neutrino survival probabilities.
    - psl: Sterile neutrino survival probabilities.
    """
    ls_meters = 1.496 * ls  # 1e11 Convert AU to meters
    pel = np.zeros((len(ls), len(enu)))
    psl = np.zeros((len(ls), len(enu)))

    util = np.ones((len(n_e), len(enu)))
    ne = n_e[:, np.newaxis] * util
    e = enu[np.newaxis, :] * util

    ve = 2 * np.sqrt(2) * FERMI_CONSTANT * ne * HBAR_C**3 * 1e-9 * e
        
    sin_theta12_2 = param['SinT12']
    cos_theta12_2 = 1 - sin_theta12_2
    
    cos_2theta12 = 1 - 2 * sin_theta12_2
    sin_2theta12 = 2 * np.sqrt(param['SinT12'] -  param['SinT12']**2)
    
    sin_theta13 = np.sin(np.radians(param['T13']))
    cos_theta13 = np.cos(np.radians(param['T13']))
    
    
    den = np.sqrt((param['M12'] * cos_2theta12 - ve)**2 + (param['M12'] * sin_2theta12)**2)
    nom = param['M12'] * cos_2theta12 - ve
    
    tm = 0.5 * np.arccos(nom / den)
  
    cos_tm_distributed = np.sum(phi[:, np.newaxis] * np.cos(tm)**2, axis=0)[np.newaxis,:]
    sin_tm_distributed = np.sum(phi[:, np.newaxis] * np.sin(tm)**2, axis=0)[np.newaxis,:]
    
    cos_tm_distributed = np.repeat(cos_tm_distributed,len(ls),axis=0)
    sin_tm_distributed = np.repeat(sin_tm_distributed,len(ls),axis=0)
    
    ls_meters = ls_meters[:,np.newaxis]
    ls_meters = np.repeat(ls_meters,len(enu),axis=1)
    
    e_extend = enu[np.newaxis,:]
    e_extend = np.repeat(e_extend,len(ls),axis=0)
    
    ae1 = cos_theta13**4 * cos_theta12_2 * cos_tm_distributed * np.cos(10 * param['mum1'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    ae2 = cos_theta13**4 * sin_theta12_2 * sin_tm_distributed * np.cos(10 * param['mum2'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    ae3 = sin_theta13**4 * np.cos(10 * param['mum3'] * ls_meters / (HBAR_C * 2 * e_extend))**2

    pel = ae1 + ae2 + ae3

    as1 = cos_theta13**2 * cos_tm_distributed * np.sin(10 * param['mum1'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    as2 = cos_theta13**2 * sin_tm_distributed * np.sin(10 * param['mum2'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    as3 = sin_theta13**2 * np.sin(10 * param['mum3'] * ls_meters / (HBAR_C * 2 * e_extend))**2

    psl = as1 + as2 + as3

    return pel, psl



def ULDM(param, enu):
    """
    Calculate the survival probabilities for solar neutrinos in presence of ultra light dark matter field.
    
    Parameters:
    - param: Dictionary containing the physical parameters 
    ('M12', 'SinT12', 'T13', 'mu1', 'mu2', 'mdm', 'epsx', 'epsy', 'alpha').
    mu1 and mu2 are in GeV^{-1}
    alpha is in degrees
    mdm is in 1e-21 eV
    
    - enu: Array of neutrino energies in MeV.
    
    Returns:
    - pel: Electron neutrino survival probabilities.
    - psl: Sterile neutrino survival probabilities.
    - ls: solar distances in AU and solar angle in degrees.
    """
        
    ls, year, angle = SunEarthDistance(ParseDate('2020,1,1'), 365.25, 0.5)
    year_list = year * 3.6525 * 2.4 * 6. * 6. / 6.6 # in 1e21 eV^-1
    ls_meters = 1.496 * ls  # 1e11 Convert AU to meters
    th_radians = np.radians(angle) 

    pel = np.zeros((len(ls), len(enu)))
    psl = np.zeros((len(ls), len(enu)))

    util = np.ones((len(n_e), len(enu)))
    ne = n_e[:, np.newaxis] * util
    e = enu[np.newaxis, :] * util

    ve = 2 * np.sqrt(2) * FERMI_CONSTANT * ne * HBAR_C**3 * 1e-9 * e
        
    sin_theta12_2 = param['SinT12']
    cos_theta12_2 = 1 - sin_theta12_2
    
    cos_2theta12 = 1 - 2 * sin_theta12_2
    sin_2theta12 = 2 * np.sqrt(param['SinT12'] -  param['SinT12']**2)
    
    sin_theta13 = np.sin(np.radians(param['T13']))
    cos_theta13 = np.cos(np.radians(param['T13']))
    
    den = np.sqrt((param['M12'] * cos_2theta12 - ve)**2 + (param['M12'] * sin_2theta12)**2)
    nom = param['M12'] * cos_2theta12 - ve
    
    tm = 0.5 * np.arccos(nom / den)
  
    cos_tm_distributed = np.sum(phi[:, np.newaxis] * np.cos(tm)**2, axis=0)[np.newaxis,:]
    sin_tm_distributed = np.sum(phi[:, np.newaxis] * np.sin(tm)**2, axis=0)[np.newaxis,:]
    
    cos_tm_distributed = np.repeat(cos_tm_distributed, len(ls), axis=0)
    sin_tm_distributed = np.repeat(sin_tm_distributed, len(ls), axis=0)
    
    ls_meters = ls_meters[:,np.newaxis]
    ls_meters = np.repeat(ls_meters, len(enu), axis=1)

    th_radians = th_radians[:,np.newaxis]
    th_radians = np.repeat(th_radians, len(enu), axis=1)

    year_list = year_list[:,np.newaxis]
    year_list = np.repeat(year_list, len(enu), axis=1)

    e_extend = enu[np.newaxis,:]
    e_extend = np.repeat(e_extend, len(ls), axis=0)
    
    polar_vec = np.sqrt((1 + ( param['epsx'] * np.cos(th_radians) - param['epsy'] * np.sin(th_radians))**2))
    
    
    mass_var = 1e9 * (RHO_DM2/param['mdm']) * (np.cos(param['mdm'] * year_list + np.radians(param['alpha'])) 
                                  - np.cos(param['mdm'] * (year_list + (1e-2/6.6) * ls_meters/LIGHT_SPEED) + np.radians(param['alpha'])))

    
    ae1 = cos_theta13**4 * cos_theta12_2 * cos_tm_distributed * np.cos(param['mu1'] * polar_vec * mass_var)**2
    ae2 = cos_theta13**4 * sin_theta12_2 * sin_tm_distributed * np.cos(param['mu2'] * polar_vec * mass_var)**2
        
    pel = ae1 + ae2 

    as1 = cos_theta13**2 * cos_tm_distributed * np.sin(param['mu1'] * polar_vec * mass_var)**2
    as2 = cos_theta13**2 * sin_tm_distributed * np.sin(param['mu2'] * polar_vec * mass_var)**2
    as3 = sin_theta13**2 * np.sin(param['mu3'] * polar_vec * mass_var)**2

    psl = as1 + as2 + as3

    return pel, psl, ls_meters/1.496, angle

def ParseDate(date_str):
    """Parse a date string in 'year,month,day' format and return the Skyfield utc date."""
    year, month, day = map(int, date_str.split(','))
    date = datetime(year, month, day, 0, 0, 0, tzinfo=utc)
    return time_scale.utc(date)
    


def SunEarthDistance(start_date, total_days, time_step):
    """Calculate Sun-Earth distance over a period."""
    
    """Load the JPL ephemeris DE421 (covers 1900-2050).
    https://ui.adsabs.harvard.edu/abs/2019ascl.soft07024R """

    planets     = load('./JPL_ephemeris/de421.bsp')
    sun,earth   = planets['sun'],planets['earth']
    t_array     = np.arange(0, total_days, time_step)
    dtheory_sun = np.zeros(len(t_array))
    day_sun     = np.zeros(len(t_array))
    lat_sun     = np.zeros(len(t_array))
    
    for i,dt in enumerate(t_array):
        tstep = start_date + dt
        
        astrometric_sun    = earth.at(tstep).observe(sun)
        lat, lon, distance = astrometric_sun.radec()
        dtheory_sun[i]     = distance.au
        lat_sun[i]         = lat._degrees
        day_sun[i]         = np.mod(dt,365.25)/365.25   # :)

    day_sun -= day_sun[dtheory_sun==np.min(dtheory_sun)]
    day_sun[day_sun<0] += 1
    lat_sun -= lat_sun[dtheory_sun == np.min(dtheory_sun)]
    lat_sun[lat_sun<0] += 360
    return dtheory_sun, day_sun, lat_sun
