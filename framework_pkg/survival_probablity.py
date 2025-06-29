import os
import sys
import numpy as np

from numba import prange

from datetime import datetime
from skyfield.api import load, utc

from numpy import arcsin
from math import sqrt, pi, radians

# Get the current working directory
peanuts_path = os.path.abspath(os.path.join(os.getcwd(), "external", "PEANUTS"))

# Add PEANUTS to the Python path
sys.path.append(peanuts_path)

from peanuts.pmns import PMNS
from peanuts.solar import SolarModel, solar_flux_mass
from peanuts.earth import EarthDensity
from peanuts.evolutor import FullEvolutor

# Global constants
FERMI_CONSTANT = 1.166  # e-11 MeV^-2
ELECTRON_MASS = 0.511  # MeV
HBAR_C = 1.97  # e-11 MeV cm
LIGHT_SPEED = 2.998 # 1e8 m/s
RHO_DM2  = np.sqrt(2 * 0.4 * 7.65) #e-21 GeV^2
ASTRO_UNIT    =  1.496 #1e11 m
ASTRO_UNIT_EV =  0.758 #1e18 eV^-1
ASTRO_UNIT_S  = 499 # s

time_scale = load.timescale()  # Create a timescale object

def MSW(param, enu, depth=1e3):
    """
    Calculate the survival probabilities for MSW neutrinos.
    
    Parameters:
    - param: Dictionary containing the physical parameters ('M12', 'SinT12', 'T13').
    - enu: Array of neutrino energies in MeV.
    - depth: is the underground detector depth, in units of meters.
    The default value is matched with Super Kamiokande.
    
    Returns:
    - pee: Electron neutrino survival probabilities.
    """
        
    th12 = arcsin(sqrt(param['SinT12']))
    th13 = radians(param['T13'])
    th23 = 0.85521

    d = 3.4034
    pmns = PMNS(th12, th13, th23, d)

    DeltamSq21 = param['M12']
    DeltamSq3l = 2.46e-3
    
    # Get arguments
    fraction = "8B"
    
    
    # Get the current directory path
    path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(path)
    path = os.path.join(parent_dir, 'external', 'PEANUTS')
    solar_file  = path+'/Data/bs2005agsopflux.csv'
    solar_model = SolarModel(solar_file)

    radius_profile = solar_model.radius()
    density_profile = solar_model.density()
    flux_distributin = solar_model.fraction(fraction)
    
    # Earth density
    density_file = path+'/Data/Earth_Density.csv' 
    earth_density = EarthDensity(density_file=density_file)
    eta_day = np.linspace(np.pi/2 , np.pi, 5)
    eta_night = np.linspace(0 , np.pi/2, 5)

    mass_weights = np.zeros((enu.shape[0],3))
    I_evolved = np.zeros((enu.shape[0],3,2))
    
    for i in prange(enu.shape[0]):
        mass_weights[i] = solar_flux_mass(th12, th13, DeltamSq21, DeltamSq3l, enu[i], radius_profile, density_profile, flux_distributin)
        evolved_day = []
        evolved_night = []
        for eta in (eta_day):
            evol = FullEvolutor(earth_density, DeltamSq21, DeltamSq3l, pmns, enu[i], eta, depth, False)
            evolved_day.append((np.square(np.abs((evol @ pmns.pmns))))[0,:])
            
        for eta in (eta_night):
            evol = FullEvolutor(earth_density, DeltamSq21, DeltamSq3l, pmns, enu[i], eta, depth, False)
            evolved_night.append(np.square(np.abs((evol @ pmns.pmns)))[0,:])

        I_evolved[i,:,0] = np.mean(np.array(evolved_day),axis=0)
        I_evolved[i,:,1] = np.mean(np.array(evolved_night),axis=0)

        
    return I_evolved, mass_weights  

'''
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
    # ls_meters = 1.496 * ls  # 1e11 Convert AU to meters
    # pel = np.zeros((len(ls), len(enu)))
    # psl = np.zeros((len(ls), len(enu)))

    # util = np.ones((len(n_e), len(enu)))
    # ne = n_e[:, np.newaxis] * util
    # e = enu[np.newaxis, :] * util

    # ve = 2 * np.sqrt(2) * FERMI_CONSTANT * ne * HBAR_C**3 * 1e-9 * e
        
    # sin_theta12_2 = param['SinT12']
    # cos_theta12_2 = 1 - sin_theta12_2
    
    # cos_2theta12 = 1 - 2 * sin_theta12_2
    # sin_2theta12 = 2 * np.sqrt(param['SinT12'] -  param['SinT12']**2)
    
    # sin_theta13 = np.sin(np.radians(param['T13']))
    # cos_theta13 = np.cos(np.radians(param['T13']))
    
    
    # den = np.sqrt((param['M12'] * cos_2theta12 - ve)**2 + (param['M12'] * sin_2theta12)**2)
    # nom = param['M12'] * cos_2theta12 - ve
    
    # tm = 0.5 * np.arccos(nom / den)
  
    # cos_tm_distributed = np.sum(phi[:, np.newaxis] * np.cos(tm)**2, axis=0)[np.newaxis,:]
    # sin_tm_distributed = np.sum(phi[:, np.newaxis] * np.sin(tm)**2, axis=0)[np.newaxis,:]
    
    # cos_tm_distributed = np.repeat(cos_tm_distributed,len(ls),axis=0)
    # sin_tm_distributed = np.repeat(sin_tm_distributed,len(ls),axis=0)
    
    # ls_meters = ls_meters[:,np.newaxis]
    # ls_meters = np.repeat(ls_meters,len(enu),axis=1)
    
    # e_extend = enu[np.newaxis,:]
    # e_extend = np.repeat(e_extend,len(ls),axis=0)
    
    # ae1 = cos_theta13**4 * cos_theta12_2 * cos_tm_distributed * np.cos(10 * param['mum1'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    # ae2 = cos_theta13**4 * sin_theta12_2 * sin_tm_distributed * np.cos(10 * param['mum2'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    # ae3 = sin_theta13**4 * np.cos(10 * param['mum3'] * ls_meters / (HBAR_C * 2 * e_extend))**2

    # pel = ae1 + ae2 + ae3

    # as1 = cos_theta13**2 * cos_tm_distributed * np.sin(10 * param['mum1'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    # as2 = cos_theta13**2 * sin_tm_distributed * np.sin(10 * param['mum2'] * ls_meters / (HBAR_C * 2 * e_extend))**2
    # as3 = sin_theta13**2 * np.sin(10 * param['mum3'] * ls_meters / (HBAR_C * 2 * e_extend))**2

    # psl = as1 + as2 + as3

    # return pel, psl
    
    return None

def ULDM(param, enu, eta, theta, distance, day, depth=1e3):
    """
    Calculate the survival probabilities for solar neutrinos in presence of ultra light dark matter field.
    
    Parameters:
    - param: Dictionary containing the physical parameters 
    ('M12', 'SinT12', 'T13', 'mu1', 'mu2', 'mu3', 'mdm', 'eps', 'alpha_eps', 'alpha').
    mu1 and mu2, and mu3 are in 10^{-7} GeV^{-1}
    alpha is in degrees
    mdm is in 1e-18 eV
    0 <= eps <= 1
    0 <= alpha_eps <= pi

    - enu: Array of neutrino energies in MeV.
    - eta: Array of diurnal angles in radian. pi/2 <= eta <= pi for day 
    - theta: Array of annual angles in radian.
    - distance: Array of distance between earth and sun corresponding to theta. it is in unit of 1 AU
    - day: Array of earth day corresponding to theta in unit of year.
    - depth: is the underground detector depth, in units of meters. The default value is matched with Super Kamiokande.
    
    Returns:
    - pee: Electron neutrino survival probabilities.
    - pes: Sterile neutrino probabilities.
    """


    # 1 year = \approx 4.8 10^{22} ev^{-1}
    day_list  = day * 365.25 * 2.4 * 6. * 6. * 1.519 # in 1e18 eV^-1

    polar_vec = np.sqrt((1 -  param['eps']**2 * np.cos( np.radians(param['alpha_eps']) - theta )**2 )) 
    dimensionless_dipole = RHO_DM2 * 0.1 * ASTRO_UNIT_EV * distance * polar_vec 
    dimensionless_dm_mass= 0.5 * param['mdm'] * ASTRO_UNIT_EV * distance
    sinc_dm_mass = np.sinc(dimensionless_dm_mass)
    mass_var     = np.sin(param['mdm'] * day_list + param['alpha'] - dimensionless_dm_mass)


    uldm_term1 = np.cos(param['mu1'] * dimensionless_dipole * sinc_dm_mass * mass_var)**2 / distance**2 
    uldm_term2 = np.cos(param['mu2'] * dimensionless_dipole * sinc_dm_mass * mass_var)**2 / distance**2 
    uldm_term3 = np.cos(param['mu3'] * dimensionless_dipole * sinc_dm_mass * mass_var)**2 / distance**2 
    
    uldm_term  = np.array([uldm_term1,uldm_term2,uldm_term3]).T 

    th12 = arcsin(sqrt(param['SinT12']))
    th13 = radians(param['T13'])
    th23 = 0.85521

    DeltamSq21 = param['M12']
    DeltamSq3l = 2.46e-3
    
    # Get arguments
    fraction = "8B"
    
    # Get the current directory path
    path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(path)
    path = os.path.join(parent_dir, 'external', 'PEANUTS')
    solar_file  = path+'/Data/bs2005agsopflux.csv'
    solar_model = SolarModel(solar_file)

    radius_profile = solar_model.radius()
    density_profile = solar_model.density()
    flux_distributin = solar_model.fraction(fraction)
    mass_weights_active = np.zeros((distance.shape[0], enu.shape[0], 3))
    mass_weights_sterile = np.zeros((distance.shape[0], enu.shape[0], 3))
    for i in prange(enu.shape[0]):
        mass_weights_naked = solar_flux_mass(th12, th13, DeltamSq21, DeltamSq3l, enu[i], radius_profile, density_profile, flux_distributin)
        mass_weights_active[:, i, :] = uldm_term * mass_weights_naked
        mass_weights_sterile[:, i, :] = (1 - uldm_term) * mass_weights_naked
    
    d = 3.4034
    pmns = PMNS(th12, th13, th23, d)
    
    # Earth density
    density_file = path+'/Data/Earth_Density.csv' 
    earth_density = EarthDensity(density_file=density_file)

    pee = np.zeros((distance.shape[0], eta.shape[0],  enu.shape[0]))
    pes = np.zeros((distance.shape[0], eta.shape[0],  enu.shape[0]))
    for j in range(len(eta)):
        evol = EarthEvolution(enu, eta[j], earth_density, depth, DeltamSq21, DeltamSq3l, pmns)
        evolved = np.square(np.abs((evol @ pmns.pmns[np.newaxis])))
        evolved = evolved[np.newaxis, :, :] * np.ones((distance.shape[0],1,1,1))
        pee[:, j, :] = (evolved @ mass_weights_active[:, :, :, np.newaxis])[:, :, 0, 0]
        
    return pee, pes 


def EarthEvolution(enu, eta_angle, earth_density, depth, DeltamSq21, DeltamSq3l, pmns):
    evol = np.zeros((enu.shape[0],3,3), dtype=np.complex128)
    for i in prange(enu.shape[0]):
        evol[i] = FullEvolutor(earth_density, DeltamSq21, DeltamSq3l, pmns, enu[i], eta_angle, depth, False)
    return evol
'''

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
