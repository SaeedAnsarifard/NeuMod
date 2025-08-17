import os
import sys
import numpy as np

from numba import prange

from datetime import datetime
from skyfield.api import load, utc

#from numpy import arcsin
from math import pi, radians

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


class SurvivalProbablity:

    def __init__(self, lam, fraction='8B', depth=1e3):
        self.depth = depth

        self.th13 = radians(8.53)
        self.th23 = radians(48.5)
        self.d = 3.4034
        self.DeltamSq3l = 2.534e-3
        self.fraction = fraction
        
        # Initialize th12 and DeltamSq21 without values
        self.th12 = None
        self.DeltamSq21 = None

        solar_model, self.earth_density = self._setup_solar_earth_model()
        self.radius_profile = solar_model.radius()
        self.density_profile = solar_model.density()
        self.flux_distributin = solar_model.fraction(self.fraction)

        self.t_year = np.linspace(0, 1, 365) # 1 days resolution
        self.t_day = np.linspace(0, 0.5, int(12 * 10))

        self.eta_info, self.eta_list = self._EtaMaker(lam)

    def _setup_solar_earth_model(self):
        # Get the current directory path
        path = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(path)
        path = os.path.join(parent_dir, 'external', 'PEANUTS')
        
        solar_file  = path+'/Data/bs2005agsopflux.csv'
        density_file = path+'/Data/Earth_Density.csv' 

        earth_density = EarthDensity(density_file=density_file)
        solar_model = SolarModel(solar_file)
        return solar_model, earth_density

    def _EtaMaker(self, lam):
        from collections import defaultdict

        t_year = self.t_year 
        t_day = self.t_day 

        # Constants
        cos_lam = np.cos(np.radians(lam))
        sin_lam = np.sin(np.radians(lam))

        # Filtering parameters
        epsilon = 2e-3  # Eta Resolution threshold

        # Create structured array to store all η information
        dtype = [('eta', float), ('i', int), ('j', int), ('is_unique', bool), 
                ('i_original', int), ('j_original', int)]
        eta_info = np.empty((len(t_year), len(t_day)), dtype=dtype)

        # Dictionary to track unique η values and their first occurrence
        eta_lookup = defaultdict(list)

        for i, y in enumerate(t_year):
            sin_delta = -np.sin(np.radians(23.4)) * np.cos(2 * np.pi * y)
            cos_delta = np.sqrt(1 - sin_delta**2)
            
            for j, d in enumerate(t_day):
                current_eta = cos_lam * cos_delta * np.cos(2 * np.pi * d) - sin_lam * sin_delta
                if current_eta < np.cos(1.05 * pi / 2):
                    #the evolutor is approximated at value of eta = pi for eta > 1.05 * pi / 2 
                    current_eta = -1.

                
                # Check if this η is close to any previously stored η
                is_unique = True
                i_orig, j_orig = i, j
                
                # Find all buckets this η could belong to (for epsilon tolerance)
                eta_key = round(current_eta / epsilon)
                
                if eta_key in eta_lookup:
                    # Check all η values in this bucket
                    for (stored_eta, stored_i, stored_j) in eta_lookup[eta_key]:
                        if np.isclose(current_eta, stored_eta, atol=epsilon):
                            is_unique = False
                            i_orig, j_orig = stored_i, stored_j
                            break
                
                # Store the information
                eta_info[i,j] = (current_eta, i, j, is_unique, i_orig, j_orig)
                
                # If unique, add to our lookup dictionary
                if is_unique:
                    eta_lookup[eta_key].append((current_eta, i, j))


        unique_mask = eta_info['is_unique']
        eta_list = np.arccos(eta_info[unique_mask]['eta'])
        return eta_info, eta_list
    
    def _MSW(self, th12, DeltamSq21, enu):

        self.th12 = th12
        self.DeltamSq21 = DeltamSq21

        pmns = PMNS(self.th12, self.th13, self.th23, self.d)
        
        mass_weights = np.zeros((enu.shape[0],3))
        U_evolved = np.zeros((self.eta_info.shape[0], self.eta_info.shape[1], enu.shape[0], 3))

        matter_effect_index = len(enu[enu<3.5])
        unique_mask = self.eta_info['is_unique']
        for i in prange(0, matter_effect_index):
            mass_weights[i] = solar_flux_mass(self.th12, self.th13, self.DeltamSq21, self.DeltamSq3l, enu[i],
                                              self.radius_profile, self.density_profile, self.flux_distributin)
            
            U_evolved[:,:,i,:] = np.square(np.abs((pmns.pmns)))[0,:] 

        for i in prange(matter_effect_index, enu.shape[0]):
            mass_weights[i] = solar_flux_mass(self.th12, self.th13, self.DeltamSq21, self.DeltamSq3l, enu[i],
                                              self.radius_profile, self.density_profile, self.flux_distributin)
            
            evol_list = np.empty((len(self.eta_list), 3))
            for j, eta in enumerate (self.eta_list):
                evol = FullEvolutor(self.earth_density, self.DeltamSq21, self.DeltamSq3l, pmns, enu[i], eta, self.depth, False)
                evol_list[j] = np.square(np.abs((evol @ pmns.pmns)))[0,:]
                
            U_evolved[unique_mask,i,:] = evol_list
            U_evolved[~unique_mask,i,:] = U_evolved[self.eta_info['i_original'][~unique_mask],
                                                    self.eta_info['j_original'][~unique_mask],i,:]

        return U_evolved, mass_weights  
    
        
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
