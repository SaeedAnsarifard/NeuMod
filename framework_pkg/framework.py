import numpy as np
import pandas as pd
from numba import njit, prange

from datetime import datetime
from skyfield.api import load, utc

from framework_pkg.survival_probablity import SurvivalProbablity 

# Global Constants
FERMI_CONSTANT = 1.166  # e-11 MeV^-2
HBAR_C = 1.97  # e-11 MeV cm
HBARC_FERMI_CONSTANT = 1.97 * 1.166  #x 1e-11 MeV.cm x 1e-11 MeV^-2
ELECTRON_MASS = 0.511  # MeV
WEAK_MIXING_ANGLE = 0.2315
RHO = 1.0126
ALPHA = 1 / (137 * np.pi)  # Fine-structure constant (radiative correction term)
ECCENTRICITY = 0.016698


time_scale = load.timescale()  # Create a timescale object

class FrameWork:
    """
    Computes B8 prediction in unit of [10^6 cm^-2 s^-1] at each time bin from initial to final date,
    according to the Super-Kamiokande experiment response function.
    
    Parameters:
    - real_exposure: If True, uses real experimental exposure times from './Data/time_exposures.txt'.
                     If False, simulates ideal exposure with 10% random downtime (default is False). 

    - threshold: Minimum electron recoil energy threshold in MeV. Events below this energy are masked.
                 (default is 3.5 MeV).

    - bin_max: Maximum number of bins used in spectral analysis. Must be between 1 and 23 (inclusive).
               (default is 23).

    - lat: Latitude of the experiment location in degrees (default is 36°).

    - efficiency_correction: Whether to apply the Super-Kamiokande efficiency function correction.
                            Should be True for total event rate calculations (default is True).

    - resolution_correction: Whether to apply the Super-Kamiokande response function correction.
                            Should be True for spectral analysis (default is False).
    """

    def __init__(self, threshold=3.5, bin_max=23, lat=36, efficiency_correction=True, resolution_correction=False):
        
        if not (1 <= bin_max <= 23):
            raise ValueError("bin_max must be between 1 and 23 (inclusive)")
        
        self._threshold = threshold
        self._bin_max = bin_max
        self._lat = lat
        self._efficiency_correction = efficiency_correction
        self._resolution_correction = resolution_correction
        self.survival_probability = SurvivalProbablity(self.lat)

        self._time_config()
        self.unoscillated_expected_spectrum = None
        self.unoscillated_expected_event_rate = None
        self._energy_config(np.linspace(0.1,15.5,500))

        # Default Values        
        self.U_evolved = None
        self.mass_weights = None
        
        self.time_periods = None
        
        self.p_uldm = None
        self.pee = None
        self.pes = None

        #Default Parameters
        self.param = {'SinT12': 0.319, 'M12': 7.54e-5}

    @property
    def threshold(self):
        return self._threshold

    @property
    def bin_max(self):
        return self._bin_max

    @property
    def lat(self):
        return self._lat

    @property
    def efficiency_correction(self):
        return self._efficiency_correction

    @property
    def resolution_correction(self):
        return self._resolution_correction

    def __getitem__(self, param_update):
        """
        Compare the oscilated and unoscillated signal given updated parameters.

        Parameters:
            param_update (dict): Dictionary containing updated parameter values.

        Returns:
            
        """
        self.param.update(param_update)

        th12 =  np.arcsin(np.sqrt(self.param['SinT12']))
        DeltamSq21 = self.param['M12']

        self.U_evolved, self.mass_weights = self.survival_probability._MSW(th12, DeltamSq21, self.energy_nu)
        integral_recoil = self._recoil_integral()

        final_result = {}
        for period in self.time_periods:
            final_result[period] = integral_recoil[period] / ((( 1 -  ECCENTRICITY * np.cos(self.theta_p) )**2)[:, np.newaxis] * self.unoscillated_term[np.newaxis, :]) 
            
        return final_result
    
    def _time_config(self):
        # if self._add:
        #     self.time_day = np.loadtxt(self._add)
        # else:
        #     self.time_day = test_exposure(self.lat)
        

        NUM_DAY = (pd.Timestamp('2018-5-30') - pd.Timestamp('2008-09-15')).days
        self.time_day = np.linspace(0,NUM_DAY-1,NUM_DAY)

        # 1 s \approx 1.519 10^{15} ev^{-1}
        self.time_ev = self.time_day * 2.4 * 6. * 6. * 1.519 # in 1e18 eV^-1
    
        t_p = (pd.Timestamp('2009-01-03') - pd.Timestamp('2008-09-15')).days  # Days to perihelion
        self.theta_p = (2 * np.pi / 365.25) * (self.time_day - t_p)
        self.t_year = np.mod(self.time_day - t_p , 365.25) / 365.25
        self.year_closest_indices = np.argmin(np.abs(self.survival_probability.t_year[:, np.newaxis] - self.t_year), axis=0)

    def _energy_config(self, energy_nu, res=30):
        
        self.energy_nu = energy_nu
        
        # Load neutrino energy spectrum (B8 spectrum)
        spectrumB8 = np.loadtxt('./Spectrum/B8_spectrum.txt')
        
        self.spectrum_nu = np.interp(energy_nu, spectrumB8[:, 0], spectrumB8[:, 1])

        # Calculate recoil energy (in MeV)
        self.energy_recoil = energy_nu / (1 + ELECTRON_MASS / (2 * energy_nu))
        self.energy_recoil = np.concatenate((np.logspace(-5,np.log10(self.energy_recoil[0]),res)[:-1], self.energy_recoil))

        # neutrino electron/moun elastic scattering cross section
        self.cs_electron = compute_cross_section(energy_nu, self.energy_recoil, 1)
        self.cs_muon = compute_cross_section(energy_nu, self.energy_recoil, -1)
        self._weight_config()
                
    def _weight_config(self):
        # Load spectrum data from file
        self.spectrum_data = np.loadtxt('./Data/B8_SuperK_Spectrum_2023.txt')
        self.energy_obs = self.spectrum_data[:self.bin_max, :2]

        self.response = response_function(self.energy_obs, self.energy_recoil)
        self.efficiency = efficiency_function(self.energy_recoil, self.threshold)
        
        self.weight = []
        if self.efficiency_correction:
            # Map integrals to total signal with reduction efficiency 
            self.weight.append(self.efficiency)
        if self.resolution_correction:
            # Map integrals to observed energy bins
            for i in range (self.response.shape[0]):
                self.weight.append(self.response[i])

        if len(self.weight)==0:
            # Map integrals to total ideal signal without reduction efficiency 
            self.weight = np.ones((1,len(self.energy_recoil)))
        else:
            self.weight = np.array(self.weight)
        self.nonzero_indices = self._collect_nonzero_indices()

        # Neutrino flux normalization from SNO
        self.SNO_norm = 1e-4 * 5.25  # x 10^10 cm^-2 s^-1
        self.target_number = (10/18) * (1/1.67) * 6. * 6. * 24. #per day per kilo ton 10^35
        self.total_volume = 22.5  # Total detector volume in kilotons
        
        # Compute response function and unoscillated integral
        self.unoscillated_term = compute_unoscilated_signal(
            self.energy_recoil,
            self.energy_nu,
            self.spectrum_nu,
            self.cs_electron,
            self.weight
        )

        if self.resolution_correction: 
            # Compute unoscillated expected spectrum per day per 22.5 kiloton
            if self.efficiency_correction:
                self.unoscillated_expected_spectrum = (
                    self.total_volume
                    * self.SNO_norm
                    * self.target_number
                    * self.unoscillated_term[1:]
                )
            else:
                self.unoscillated_expected_spectrum = (
                    self.total_volume
                    * self.SNO_norm
                    * self.target_number
                    * self.unoscillated_term
                )
        elif self.efficiency_correction:
            pass
        else:
            # Compute unoscillated events per day per 32.5 kiloton
            self.unoscillated_expected_event_rate = (
                32.5
                * self.SNO_norm
                * self.target_number
                * self.unoscillated_term
            )
        
    def _collect_nonzero_indices(self):
        # Stack all arrays into a 2D array of shape (n, m)
        stacked_arrays = np.stack(self.weight)
        # Check for each index m if any of the n arrays has value > 1e-6
        condition = np.any(stacked_arrays > 1e-6, axis=0)
        # Get the indices where condition is True
        indices = np.where(condition)[0]
        # Convert to a set if needed
        return set(indices)
        
    def _day_night_probability(self):
        """Calculate day and night survival probabilities considering ULDM modulation."""
                
        self.time_periods = ['day', 'night']

        U_evolved_period = {}
        for period in self.time_periods:
            U_evolved_period[period] = np.zeros((self.t_year.shape[0], self.U_evolved.shape[2], self.U_evolved.shape[3]))
        
        # Explicit day/night conditions             
        day_condition = self.survival_probability.eta_info['eta'] <= 0
        night_condition = ~day_condition
        conditions = [day_condition, night_condition]  
        
        for period, cond in zip(self.time_periods, conditions):
            for j in range(cond.shape[0]):
                mask = (self.year_closest_indices == j)
                U_evolved_period[period][mask] = np.mean(self.U_evolved[j, cond[j]], axis=0)
        
        # Calculate ULDM modulation
        puldm = (1 - self.param["eps"] * np.cos(self.theta_p - self.param["alpha_eps"])**2) * \
                np.sin(self.param["mdm"] * self.time_ev + self.param["alpha"])**2
        mu_couplings = np.array([self.param["mu1"], self.param["mu2"], self.param["mu3"]])
        self.p_uldm = puldm[:, np.newaxis] * mu_couplings
        
        # Calculate probabilities
        self.pee = {}
        for period in self.time_periods:
            self.pee[period] = np.sum((1 - self.p_uldm)[:, np.newaxis, :] * U_evolved_period[period] * self.mass_weights, axis=2)
        
        self.pes = np.sum(self.p_uldm[:, np.newaxis, :] * self.mass_weights, axis=2) 

    def _neutrino_integral(self):
        """Calculate integrated electron and muon neutrino event rates for day/night periods.
        
        Returns:
            tuple: (integral_electron, integral_muon) where each is a dict with 'day'/'night' keys
                containing 2D arrays of shape (time_steps, recoil_bins)
        """
        self._day_night_probability()

        num_recoil_bins = len(self.energy_recoil)
        nu_bins = len(self.energy_nu)
        
        # Initialize output dictionaries
        integral_electron = {part: np.zeros((self.time_day.shape[0], num_recoil_bins)) 
                            for part in self.time_periods}
        integral_muon = {part: np.zeros((self.time_day.shape[0], num_recoil_bins)) 
                        for part in self.time_periods}

        # Determine integration bounds for each recoil bin
        for k in self.nonzero_indices:
            # Handle edge case where we need to truncate the integration range
            if k <= num_recoil_bins - nu_bins:
                # Full integration range
                start_idx = 0
                energy_slice = slice(None)  # Equivalent to ':'
            else:
                # Truncated integration range
                start_idx = k + nu_bins - num_recoil_bins
                energy_slice = slice(start_idx, None)
            
            # Get slices for this bin
            e_nu = self.energy_nu[energy_slice]
            cs_e = self.cs_electron[k, energy_slice]
            cs_m = self.cs_muon[k, energy_slice]
            spectrum = self.spectrum_nu[energy_slice]
            
            for period in self.time_periods:
                # Electron neutrino integral
                integrand_e = self.pee[period][:, energy_slice] * spectrum * cs_e
                integral_electron[period][:, k] = np.trapz(integrand_e, e_nu, axis=1)
                
                # Muon neutrino integral
                integrand_m = (1 - self.pee[period][:, energy_slice] - self.pes[:, energy_slice]) * spectrum * cs_m
                integral_muon[period][:, k] = np.trapz(integrand_m, e_nu, axis=1)
        
        return integral_electron, integral_muon
    
    def _recoil_integral(self):
        """Compute the recoil energy integrals for electron and muon neutrinos (day/night).
        
        Returns:
            tuple: Two dictionaries (`integral_electron_recoil`, `integral_muon_recoil`), 
                each containing 'day' and 'night' keys with arrays of shape (time_steps, weights).
        """
        integral_electron, integral_muon = self._neutrino_integral()
        num_time_steps = self.time_day.shape[0]
        num_weights = len(self.weight)
        
        # Initialize output arrays
        integral_electron_recoil = {
            period: np.zeros((num_time_steps, num_weights)) 
            for period in self.time_periods
        }
        integral_muon_recoil = {
            period: np.zeros((num_time_steps, num_weights)) 
            for period in self.time_periods
        }

        # Vectorized computation (avoid Python loops)
        for period in self.time_periods:
            # Shape: (num_time_steps, num_recoil_bins) * (num_weights, num_recoil_bins)
            weighted_electron = self.weight[np.newaxis, :, :] * integral_electron[period][:, np.newaxis, :]
            weighted_muon = self.weight[np.newaxis, :, :] * integral_muon[period][:, np.newaxis, :]
            
            # Integrate over recoil energy (axis=2)
            integral_electron_recoil[period][:, :] = np.trapz(weighted_electron, self.energy_recoil, axis=2)
            integral_muon_recoil[period][:, :] = np.trapz(weighted_muon, self.energy_recoil, axis=2)
        
        combined_integral = {period: integral_electron_recoil[period] + integral_muon_recoil[period]
                             for period in self.time_periods
                             }
        return combined_integral
    



    def _bin_data(self, data):
        """Bins the data based on unique distance in the data."""
        error = np.mean(data[:,4:6],axis=1)
        data_new = [[],[],[],[]]
        d_unique = np.unique(data[:,6])
        for i in range(0,len(d_unique)-1,2):
            cond = (data[:,6] == d_unique[i])|(data[:,6] == d_unique[i+1])
            data_new[0].append(0.5*(d_unique[i]+d_unique[i+1]))
            data_new[1].append(np.mean(data[cond,3]))
            data_new[2].append(np.sqrt(np.sum(error[cond]**2))/len(error[cond]))
            data_new[3].append(0.5*(d_unique[i]-d_unique[i+1]))
        return np.array(data_new).T
        
    def _variable_maker(self, dtheta=0.01):
        eta = np.arange(0, np.pi, 0.01)
        theta = np.arange(0, 2 * np.pi, dtheta)
        distance =  (1 - ECCENTRICITY**2) / (1 + ECCENTRICITY * np.cos(theta)) 
        norm_time = np.trapz(distance**2, theta)
        days = []
        cond = np.full(distance.shape, False)
        for i in range (len(distance)):
            delta_t = np.trapz(distance[:i]**2, theta[:i]) 
            if delta_t != 0:
                days.append(delta_t / norm_time)
                cond[i] = True

        days = np.array(days) 
        distance  = distance[cond]
        theta = theta[cond]
        return eta, theta, distance, days
    
def bin_prediction(raw_result, exposure, total_days, spectrum=False):
    num_energy_bins = raw_result['day'].shape[1]
    
    time_bins = np.unique(exposure[:,0])
    num_time_bins = len(time_bins)


    R_bin_total = np.zeros((num_time_bins, num_energy_bins))
    R_bin_day = np.zeros((num_time_bins, num_energy_bins))
    R_bin_night = np.zeros((num_time_bins, num_energy_bins))

    for i in range (num_time_bins):
        cond_exp_days = exposure[:,0]==time_bins[i]
        expected_days = exposure[cond_exp_days, 1]
        expected_indices = np.where(np.isin(total_days, expected_days))[0]
        for j in range (num_energy_bins):
            R_bin_day[i,j] = (5.25 / np.sum(exposure[cond_exp_days,2])) * np.sum(exposure[cond_exp_days,2] * raw_result['day'][expected_indices,j])
            R_bin_night[i,j] = (5.25 / np.sum(exposure[cond_exp_days,3])) * np.sum(exposure[cond_exp_days,3] * raw_result['night'][expected_indices,j])
            
            R_bin_total[i,j] = (5.25 / np.sum(exposure[cond_exp_days,2:])) * np.sum(exposure[cond_exp_days,2] * raw_result['day'][expected_indices,j] 
                                                                        + exposure[cond_exp_days,3] * raw_result['night'][expected_indices,j])
    if spectrum:
        return R_bin_day / 5.25, R_bin_night / 5.25 , R_bin_total / 5.25
    
    return R_bin_day, R_bin_night, R_bin_total

def compute_cross_section(e_nu, t_e, i=1):
    """
    Differential cross-section (dσ/dT_e) as a function of electron recoil (T_e) and neutrino energy (E_ν).
    
    Bahcall, John N., Marc Kamionkowski, and Alberto Sirlin."Solar neutrinos: Radiative corrections in neutrino-electron scattering experiments."Physical Review D 51.11 (1995): 6146.
    
    https://pdg.lbl.gov/2019/reviews/rpp2019-rev-standard-model.pdf
    
    We do not consider radiative corrections.
    
    Parameters:
    - e_nu: Neutrino energy (MeV).
    - t_e: Electron recoil energy (MeV).
    - i: Index for choosing between different interaction scenarios (1 or -1).
    
    Returns:
    - Differential cross-section value in units of 10^-45 cm^2.
    """
    
    x  = np.sqrt(1 + 2 * ELECTRON_MASS / t_e)[:,np.newaxis]*np.ones(len(e_nu))
    it = (1 / 6) * ((1 / 3) + (3 - x**2) * ((x / 2) * (np.log(x + 1) - np.log(x - 1)) - 1))
    
    if i == 1:
        kappa = 0.9791 + 0.0097 * it
        gl = RHO * (0.5 - kappa * WEAK_MIXING_ANGLE) - 1
    elif i == -1:
        kappa = 0.9970 - 0.00037 * it
        gl = RHO * (0.5 - kappa * WEAK_MIXING_ANGLE)
    
    gr = -RHO * kappa * WEAK_MIXING_ANGLE
    
    z = t_e[:,np.newaxis] / e_nu[np.newaxis,:]
    
    # Terms for the differential cross-section calculation
    a1 = gl**2 * (1 + ALPHA * 0)  # fm = 0
    a2 = gr**2 * (1 + ALPHA * 0) * (1 - z)**2  # fp = 0
    a3 = gr * gl * (1 + ALPHA * 0) * (ELECTRON_MASS / e_nu[np.newaxis,:]) * z  # fmp = 0
    
    # Differential cross-section in units of 10^-45 cm^2
    return 2 * HBARC_FERMI_CONSTANT**2 * (ELECTRON_MASS / np.pi) * (a1 + a2 - a3) * 10

def compute_unoscilated_signal(energy_recoil, energy_nu, spectrum_nu, cs_electron, weight):
    """Compute the unoscillated signal. The cross section is in unit of 10^{-45} cm^2"""
    
    r  = np.zeros(energy_recoil.shape)
    k = 0 
    for z in range (len(energy_recoil)):
        if z < len(energy_recoil) - len(energy_nu):
            r[z] = np.trapz(spectrum_nu * cs_electron[z,:], energy_nu) 
        else:
            r[z] = np.trapz(spectrum_nu[k:] * cs_electron[z,k:], energy_nu[k:])
            k = k + 1
    
    num_event = np.zeros(len(weight))
    for i in range(len(weight)):
        num_event[i] = np.trapz( r * weight[i] , energy_recoil)
    return num_event
 
def response_function(energy_obs, energy_recoil):
    """Compute the detector's response function. It is used to peredict the Super-Kamiokande spectrum data """
    r   = np.zeros((len(energy_obs),len(energy_recoil)))
    for j in range (len(energy_obs)):
        e_nu = np.linspace(energy_obs[j,0],energy_obs[j,1])
        for i,t in enumerate(energy_recoil):
            sig  = -0.05525 + 0.3162 * np.sqrt( t + ELECTRON_MASS ) + 0.04572 * ( t + ELECTRON_MASS )
            a    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-e_nu)**2/sig**2)
            r[j,i] = np.trapz(a,e_nu)
    return r

def efficiency_function(energy_recoil, threshold):
    """
    Taken from PhysRevD.109.092001 (Solar neutrino measurements using the full data period of Super-Kamiokande-IV)
    """

    superk_efficiency = np.array(pd.read_csv('./Data/superk_efficiency.csv'))
    superk_efficiency[0,0] = 3.5
    x = superk_efficiency[superk_efficiency[:,0]>= threshold,0]
    xp = superk_efficiency[superk_efficiency[:,0]>= threshold,1]
    right = superk_efficiency[-1,1]
    return np.interp(energy_recoil, x, xp, left=0, right=right)

def ideal_exposure(lat, one_bin=True, dwontime=0.1):
    """
    
    Parameters:
    -----------
    lat : float, optional
        Latitude in degrees (-90 to 90).
    
    Returns:
    --------

    """

    # Constants
    OBLIQUITY_RADIANS = np.radians(23.44)  # Earth's axial tilt (ε)
    DAYS_PER_YEAR = 365.25

    NUM_DAY = (pd.Timestamp('2018-5-30') - pd.Timestamp('2008-09-15')).days # SK phase 4
    days_in_year = np.linspace(0, NUM_DAY-1, NUM_DAY)
    
    cos_lam = np.cos(np.radians(lat))
    sin_lam = np.sin(np.radians(lat))
    
    
    t_s = (pd.Timestamp('2008-12-21') - pd.Timestamp('2008-09-15')).days
    
    t_in_day = np.linspace(0,0.5,2000)
    time_exposure = [[] for i in range(4)]


    if one_bin:
        bin_num = np.zeros(len(days_in_year))
    else:
        bin_num = np.linspace(0,len(days_in_year)-1,len(days_in_year))


    for i,day in enumerate (days_in_year):
        sin_delta = -np.sin(OBLIQUITY_RADIANS) * np.cos(2 * np.pi * (day - t_s) / DAYS_PER_YEAR)
        cos_delta = np.sqrt(1 - sin_delta**2)

        cos_eta = cos_lam * cos_delta * np.cos(2 * np.pi * t_in_day) - sin_lam * sin_delta

        time_exposure[0].append(bin_num[i])
        time_exposure[1].append(day)

        if len(t_in_day[cos_eta >= 0]) == len(t_in_day):
            time_exposure[2].append(0)
            time_exposure[3].append(24)

        elif  0 < len(t_in_day[cos_eta >=0]) < len(t_in_day): 
            time_exposure[2].append(24 - t_in_day[cos_eta >=0][-1] * 24 * 2)
            time_exposure[3].append(t_in_day[cos_eta >=0][-1] * 24 * 2)
        else:
            time_exposure[2].append(24)
            time_exposure[3].append(0)

    time_exposure = np.array(time_exposure).T

    # Randomly choose indices to keep (without replacement) for considering the 10% dead time
    remove_indices = np.random.choice(NUM_DAY, size = int(NUM_DAY * dwontime), replace=False)
    mask = np.ones(NUM_DAY, dtype=bool)
    mask[remove_indices] = False
    return time_exposure[mask]