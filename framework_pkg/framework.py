import numpy as np
import pandas as pd
from numba import njit, prange

from datetime import datetime
from skyfield.api import load, utc

from framework_pkg.survival_probablity import SunEarthDistance, ParseDate

from framework_pkg.survival_probablity import MSW

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
    - threshold: Mask neutrino energy less than threshold (default is 3.5 Mev).

    - efficiency_correction: considering the Super-Kamiokande efficiency function.
    It should be True in case of total event rate (default is True).

    - resolution_correction: considering the Super-Kamiokande response function.
    It should be True in case of spectrum analsys (default is False).
    """

    def __init__(self, threshold=3.5, efficiency_correction=True, resolution_correction=False):
        self.resolution_correction = resolution_correction
        self.efficiency_correction = efficiency_correction

        self.time_day, self.time_ev, self.theta_p = self._time_config('./Data/time_exposures.txt')
        #self.T_night, self.T_day, self.cos_eta, self.dt_dcos_eta, self.ind = self._compute_day_lenght(36)
        
        # Neutrino flux normalization from SNO
        self.SNO_norm = 1e-4 * 5.25  # x 10^10 cm^-2 s^-1
        self.target_number = (10/18) * (1/1.67) * 6. * 6. * 24. #per day per kilo ton 10^35
        self.total_volume = 22.5  # Total detector volume in kilotons

        # Load neutrino energy spectrum (B8 spectrum)
        spectrumB8 = np.loadtxt('./Spectrum/B8_spectrum.txt')
        
        self.energy_nu = np.linspace(0.1,15.5,500)
        self.spectrum_nu = np.interp(self.energy_nu, spectrumB8[:, 0], spectrumB8[:, 1])

        #self.spectrum_nu = spectrumB8[:, 1]
        #self.energy_nu = spectrumB8[:, 0] 

        # Calculate recoil energy (in MeV)
        self.energy_recoil = self.energy_nu / (1 + ELECTRON_MASS / (2 * self.energy_nu))
        self.energy_recoil = np.concatenate((np.logspace(-5,-2,30), self.energy_recoil))
 
        # neutrino electron/moun elastic scattering cross section
        self.cs_electron = self._compute_cross_section(self.energy_nu, self.energy_recoil, 1)
        self.cs_muon = self._compute_cross_section(self.energy_nu, self.energy_recoil, -1)

        # Load spectrum data from file
        self.spectrum_data = np.loadtxt('./Data/B8_SuperK_Spectrum_2023.txt')
        self.energy_obs = self.spectrum_data[:, :2]

        self.response_function = self._response_function(self.energy_obs, self.energy_recoil)
        self.efficiency = self._efficiency_function(self.energy_recoil, threshold)

        if self.resolution_correction:
            # Map integrals to observed energy bins
            self.weight = self.response_function
        elif self.efficiency_correction:
            # Map integrals to total signal with reduction efficiency 
            self.weight = self.efficiency
        else:
            # Map integrals to total ideal signal without reduction efficiency 
            self.weight = [1]

        # Compute response function and unoscillated integral
        self.unoscillated_term = self._compute_unoscilated_signal(
            self.energy_recoil,
            self.energy_nu,
            self.spectrum_nu,
            self.cs_electron,
            self.weight
        )
        self.pridection = 0 
    
        if self.resolution_correction: 
            # Compute unoscillated expected spectrum per day per 22.5 kiloton
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

        # Default parameters
        self.param = {'SinT12': 0.319, 'T13': 8.57, 'M12': 7.54e-5}

    def __getitem__(self, param_update):
        """
        Compare the oscilated and unoscillated signal given updated parameters.

        Parameters:
            param_update (dict): Dictionary containing updated parameter values.

        Returns:
            
        """
        self.param.update(param_update)

        I_evolved, mass_weights = MSW(self.param, self.energy_nu)
        
        mean_I_evolved = np.zeros(((self.time_day.shape[0], mass_weights.shape[0], mass_weights.shape[1])))

        for i in range(mass_weights.shape[1]):
            mean_I_evolved[:,:,i] = (self.time_day[:,np.newaxis,2] * I_evolved[np.newaxis,:,i,0] 
                                + self.time_day[:,np.newaxis,3] * I_evolved[np.newaxis,:,i,1]) / (self.time_day[:,np.newaxis,2] + self.time_day[:,np.newaxis,3] )
        
        self.p_msw = np.sum(mean_I_evolved * mass_weights, axis=2)

        puldm = (1 - self.param["eps"] * np.cos(self.theta_p - self.param["alpha_eps"])**2) * np.sin(self.param["mdm"] * self.time_ev + self.param["alpha"])**2
        mu_couplings = np.array([self.param["mu1"], self.param["mu2"], self.param["mu3"]])
        self.p_uldm = puldm[:,np.newaxis] * mu_couplings
        
        self.pee = np.sum((1 - self.p_uldm)[:, np.newaxis, :] * mean_I_evolved * mass_weights, axis=2)
        self.pes = np.sum(self.p_uldm[:,np.newaxis,:] * mass_weights, axis=2)
        
        #p_uldm_electron = np.sum(mean_I_evolved * mass_weights * mu_couplings, axis=2)
        #p_uldm_muon = np.sum( (mean_I_evolved - 1) * mass_weights * mu_couplings, axis=2)

        # Initialize integral arrays
        num_recoil_bins = len(self.energy_recoil)
        integral_electron = np.zeros((self.time_day.shape[0], num_recoil_bins))
        integral_muon = np.zeros((self.time_day.shape[0], num_recoil_bins))
        
        #integral_electron_uldm = np.zeros((self.time_day.shape[0], num_recoil_bins))
        #integral_muon_uldm = np.zeros((self.time_day.shape[0], num_recoil_bins))
        # Compute the electron and muon integrals
        for i in range(len(self.weight)):
            indices = np.nonzero(self.weight[i])
            z = 0
            for k in indices[0]:      
                if k < num_recoil_bins - len(self.energy_nu) :
                    #integral_electron_uldm[:,k] = np.trapz(p_uldm_electron * self.spectrum_nu * self.cs_electron[k, :], self.energy_nu, axis=1)
                    #integral_muon_uldm[:,k] = np.trapz(-p_uldm_muon * self.spectrum_nu * self.cs_muon[k, :], self.energy_nu, axis=1)
                    integral_electron[:,k] = np.trapz(self.pee * self.spectrum_nu * self.cs_electron[k, :], self.energy_nu, axis=1)
                    integral_muon[:,k] = np.trapz( (1 - self.pee - self.pes) * self.spectrum_nu * self.cs_muon[k, :], self.energy_nu, axis=1)
                else:
                    z = k + len(self.energy_nu) - num_recoil_bins 
                    #integral_electron_uldm[:,k] = np.trapz(p_uldm_electron[:,z:] * self.spectrum_nu[z:] * self.cs_electron[k, z:], self.energy_nu[z:], axis=1)
                    #integral_muon_uldm[:,k] = np.trapz(-p_uldm_muon[:, z:] * self.spectrum_nu[z:] * self.cs_muon[k, z:], self.energy_nu[z:], axis=1)
                    integral_electron[:,k] = np.trapz(self.pee[:,z:] * self.spectrum_nu[z:] * self.cs_electron[k, z:], self.energy_nu[z:], axis=1)
                    integral_muon[:,k] = np.trapz((1 - self.pee[:, z:] - self.pes[:, z:]) * self.spectrum_nu[z:] * self.cs_muon[k, z:], self.energy_nu[z:], axis=1)
                    
        integral_electron_recoil = np.zeros((integral_electron.shape[0], len(self.weight)))
        integral_muon_recoil = np.zeros((integral_electron.shape[0], len(self.weight)))
        #integral_electron_recoil_uldm = np.zeros((integral_electron.shape[0], len(self.weight)))
        #integral_muon_recoil_uldm = np.zeros((integral_electron.shape[0], len(self.weight)))
        for i in range(len(self.weight)):
            #integral_electron_recoil_uldm[:,i] = np.trapz( self.weight[i] * integral_electron_uldm, self.energy_recoil, axis=1)
            #integral_muon_recoil_uldm[:,i] = np.trapz( self.weight[i] * integral_muon_uldm, self.energy_recoil, axis=1)
            integral_electron_recoil[:,i] = np.trapz( self.weight[i] * integral_electron, self.energy_recoil, axis=1)
            integral_muon_recoil[:,i] = np.trapz( self.weight[i] * integral_muon, self.energy_recoil, axis=1)

        if self.resolution_correction:
            self.pridection = np.mean((integral_electron_recoil + integral_muon_recoil) * (1 + 2 * ECCENTRICITY * np.cos(self.theta_p))[:,np.newaxis], axis=0)
            return np.mean((integral_electron_recoil + integral_muon_recoil) * (1 + 2 * ECCENTRICITY * np.cos(self.theta_p))[:,np.newaxis] / self.unoscillated_term , axis=0)
              
        elif self.efficiency_correction:
            return (integral_electron_recoil + integral_muon_recoil)[:,0] * 5.25 / (( 1 -  ECCENTRICITY * np.cos(self.theta_p) )**2 * self.unoscillated_term) 

    def _time_config(self, add):
        time_day = np.loadtxt(add)
        # 1 s \approx 1.519 10^{15} ev^{-1}
        time_ev = time_day[:,1] * 2.4 * 6. * 6. * 1.519 # in 1e18 eV^-1
    
        t_p = (pd.Timestamp('2009-01-03') - pd.Timestamp('2008-09-15')).days  # Days to perihelion
        theta_p = (2 * np.pi / 365.25) * (time_day[:,1] - t_p)

        return time_day, time_ev, theta_p

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
        
    def _response_function(self, energy_obs, energy_recoil):
        """Compute the detector's response function. It is used to peredict the Super-Kamiokande spectrum data """
        r   = np.zeros((len(energy_obs),len(energy_recoil)))
        for j in range (len(energy_obs)):
            e_nu = np.linspace(energy_obs[j,0],energy_obs[j,1])
            for i,t in enumerate(energy_recoil):
                sig  = -0.05525 + 0.3162 * np.sqrt( t + ELECTRON_MASS ) + 0.04572 * ( t + ELECTRON_MASS )
                a    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-e_nu)**2/sig**2)
                r[j,i] = np.trapz(a,e_nu)
        return r
    
    def _efficiency_function(self, energy_recoil, threshold):
        """
        Taken from PhysRevD.109.092001 (Solar neutrino measurements using the full data period of Super-Kamiokande-IV)
        """

        import pandas as pd

        superk_efficiency = np.array(pd.read_csv('./Data/superk_efficiency.csv'))
        superk_efficiency[0,0] = 3.5
        x = superk_efficiency[superk_efficiency[:,0]>= threshold,0]
        xp = superk_efficiency[superk_efficiency[:,0]>= threshold,1]
        right = superk_efficiency[-1,1]
        return np.interp(energy_recoil, x, xp, left=0, right=right)[np.newaxis,:]
    
    def _compute_unoscilated_signal(self, energy_recoil, energy_nu, spectrum_nu, cs_electron, weight):
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

    def _compute_cross_section(self, e_nu, t_e, i=1):
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
    
    def _pridection_function(self):
        if self.resolution_correction: 
            # spectrum events per day per 22.5 kiloton with oscillations
            return ( 
                self.total_volume
                * self.SNO_norm
                * self.target_number
                * self.pridection
                )
        elif self.efficiency_correction:
            return None
        else:
            return (
                32.5
                * self.SNO_norm
                * self.target_number
                * self.pridection
                )
        
    def _compute_day_lenght(self, lat):
        """
        Calculate daylight duration (hours) for a given day and latitude.
        
        Parameters:
        -----------
        days_since_start : float or array-like
            Days since reference date (2008-09-15).
        lat : float, optional
            Latitude in degrees (-90 to 90). Default: 36°.
        
        Returns:
        --------
        daylight_hour : float or array-like
            Hour of daylight (0 corresponds to whole day to 24 corresponds to no-day).
        
        """
        # Constants
        OBLIQUITY_RADIANS = np.radians(23.44)  # Earth's axial tilt (ε)
        DAYS_PER_YEAR = 365.25

        t_s = (pd.Timestamp('2008-12-21') - pd.Timestamp('2008-09-15')).days  # Days to solstice
                
        # Solar declination (δ)
        sin_delta = -np.sin(OBLIQUITY_RADIANS) * np.cos(2 * np.pi * (self.time_day[:,1] - t_s) / DAYS_PER_YEAR)
        cos_delta = np.sqrt(1 - sin_delta**2)
        
        cos_lam = np.cos(np.radians(lat))
        sin_lam = np.sin(np.radians(lat))

        t = np.linspace(0+1e-6, 0.5-1e-6, int(1e5))
        sin_t = np.sin(2 * np.pi * t)
        cos_eta = np.zeros((len(sin_delta),len(sin_t)))
        dt_dcos_eta = np.zeros((len(sin_delta),len(sin_t)))
        T_day = np.zeros(len(sin_delta))
        T_night = np.zeros(len(sin_delta))
        ind = np.zeros(len(sin_delta), dtype=np.int32)
        for i in range (len(sin_delta)):
            cos_eta[i,:] = cos_lam * cos_delta[i] * np.cos(2 * np.pi * t) - sin_lam * sin_delta[i]
            dt_dcos_eta[i,:] = 12 / ( np.pi * cos_lam * cos_delta[i] * sin_t )
            
            if len(t[cos_eta[i]>=0]) == len(t):
                T_day[i] = 0.
                T_night[i] = 24.
                ind[i] = -1
            
            elif  0 < len(t[cos_eta[i]>=0]) < len(t)  : 
                T_day[i] =  24 - 2*t[cos_eta[i]>=0][-1]*24
                T_night[i] = 2*t[cos_eta[i]>=0][-1]*24
                ind[i] = np.argwhere(cos_eta[i]>=0)[-1][0]
            else:
                T_day[i] = 24.
                T_night[i] = 0.
                ind[i] = 0
        return T_night, T_day, cos_eta, dt_dcos_eta, ind
        
    def _compute_I_mean(self, U_sq, W, T_night, cos_eta_grid, cos_eta_k0):
        """
        Compute I_i^night(k) for each k
        
        Parameters
        ----------
        U_sq : ndarray, shape (n,)
            Array of |U_{eα} U_{αi}|^2 sampled at cos(η) grid
        W : ndarray, shape (n, m)
            Array of W(η, k) values over cos(η) and k
        T_night : ndarray, shape (m,)
            Array of T_k^{night} for each k
        cos_eta_grid : ndarray, shape (n,)
            The grid of cos(η) values, assumed sorted ascending
        cos_eta_k0 : ndarray, shape (m,)
            Upper integration limits for each k
        
        Returns
        -------
        I_night : ndarray, shape (m,)
            Integral values I_i^{night}(k) for each k
        """
        n, m = W.shape
        I_night = np.zeros(m)
        
        for k in range(m):
            # mask cos(η) ∈ [0, cos_eta_k0[k]]
            upper_limit = cos_eta_k0[k]
            
            mask = (cos_eta_grid >= 0) & (cos_eta_grid <= upper_limit)
            
            cos_eta_sub = cos_eta_grid[mask]
            integrand_sub = W[mask, k] * U_sq[mask]
            
            if integrand_sub.size > 0:
                integral = np.trapz(integrand_sub, cos_eta_sub)
            else:
                integral = 0.0
            
            I_night[k] = (2 / T_night[k]) * integral
        
        return I_night
        
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
    

                