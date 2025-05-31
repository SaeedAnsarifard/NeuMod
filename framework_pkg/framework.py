import numpy as np
import pandas as pd
from numba import njit, prange

from datetime import datetime
from skyfield.api import load, utc

from framework_pkg.survival_probablity import SunEarthDistance, ParseDate

from framework_pkg.survival_probablity import MSW, PseudoDirac, ULDM

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
    - efficiency_correction: considering the Super-Kamiokande efficiency function. It should be True in case of total event rate (default is True).
    - resolution_correction: considering the Super-Kamiokande response function. It should be True in case of spectrum analsys (default is False).
    - first_day: Start date in the format 'year,month,day' (default is '2008,9,15').
    - last_day: End date in the format 'year,month,day' (default is '2018,5,30').
    """

    def __init__(self, threshold=3.5, efficiency_correction=True, resolution_correction=False):
        self.resolution_correction = resolution_correction
        self.efficiency_correction = efficiency_correction

        self.time_weights = np.loadtxt('./Data/time_exposures.txt')
        # 1 s \approx 1.519 10^{15} ev^{-1}
        self.time_ev = self.time_weights[:,1] * 2.4 * 6. * 6. * 1.519 # in 1e18 eV^-1
    
        self.t_p = (pd.Timestamp('2009-01-03') - pd.Timestamp('2008-09-15')).days  # Days to perihelion
        self.theta_p = (2 * np.pi / 365.25) * (self.time_weights[:,1] - self.t_p)


        #self.firstday = ParseDate(first_day)
        #self.lastday = ParseDate(last_day)
        #self.total_days = int(self.lastday - self.firstday)
        
        # Neutrino flux normalization from SNO
        self.SNO_norm = 1e-4 * 5.25  # x 10^10 cm^-2 s^-1
        self.target_number = (10/18) * (1/1.67) * 6. * 6. * 24. #per day per kilo ton 10^35
        self.total_volume = 22.5  # Total detector volume in kilotons

        # Load neutrino energy spectrum (B8 spectrum)
        spectrumB8 = np.loadtxt('./Spectrum/B8_spectrum.txt')
        self.spectrum_nu = spectrumB8[:, 1]
        self.energy_nu = spectrumB8[:, 0] 

        # Calculate recoil energy (in MeV)
        self.energy_recoil = spectrumB8[:, 0] / (1 + ELECTRON_MASS / (2 * spectrumB8[:, 0]))
        self.energy_recoil = np.concatenate((np.logspace(-5,-2,30), self.energy_recoil))

        #t0 = time_scale.utc(datetime(1970, 1, 1, 0, 0, 0, tzinfo=utc))
        #self.zeroday = self.firstday.tt - t0.tt

        # Geometric characteristic: Sun-Earth distance time step (in day, 1 is a fair choice)
        #self.time_step = 0.1
        #self.distance_list, self.day_list, self.angle_list = SunEarthDistance(self.firstday, self.total_days, self.time_step)
        #self.eta_list = self._eta_list_maker(self.day_list)

        #self.eta, self.theta, self.distance, self.day = self._variable_maker()        
        #self.extended_distance = self.distance[:, np.newaxis, np.newaxis] * np.ones((1,self.eta.shape[0], self.energy_nu.shape[0]))
        
        # neutrino electron/moun elastic scattering cross section
        self.cs_electron = self._compute_cross_section(self.energy_nu, self.energy_recoil, 1)
        self.cs_muon = self._compute_cross_section(self.energy_nu, self.energy_recoil, -1)


        # # Load modulation data from file
        # self.modulation_data = np.loadtxt('./Data/sksolartimevariation5804d.txt')
        # self.modulation_data[:, :3] /= (60. * 60. * 24.)  # Convert time columns to days
        # self.modulation_data = self.modulation_data[self.modulation_data[:, 0] - self.modulation_data[:, 1] >= self.zeroday]
        # self.modulation_data[:, 0] -= self.zeroday

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
            name (string) : name of survival probablity function (MSW or ULDM)

        Returns:
            
        """
        self.param.update(param_update)

        I_evolved, mass_weights = MSW(self.param, self.energy_nu)
        mean_I_evolved = np.zeros(((self.time_weights.shape[0], mass_weights.shape[0], 3)))
        for i in range(3):
            mean_I_evolved[:,:,i] = (self.time_weights[:,np.newaxis,2] * I_evolved[np.newaxis,:,i,0] 
                                + self.time_weights[:,np.newaxis,3] * I_evolved[np.newaxis,:,i,1]) / (self.time_weights[:,np.newaxis,2] + self.time_weights[:,np.newaxis,3] )
        p_msw = np.sum(mean_I_evolved * mass_weights, axis=2)
                 

        mu_couplings = np.array([self.param["mu1"],self.param["mu2"],self.param["mu3"]])
        p_uldm_electron = np.sum(mean_I_evolved * mass_weights * mu_couplings, axis=2)
        p_uldm_muon = np.sum( (mean_I_evolved - 1) * mass_weights * mu_couplings, axis=2)

        # Initialize integral arrays
        num_recoil_bins = len(self.energy_recoil)
        integral_electron = np.zeros((self.time_weights.shape[0], num_recoil_bins))
        integral_muon = np.zeros((self.time_weights.shape[0], num_recoil_bins))
        
        integral_electron_uldm = np.zeros((self.time_weights.shape[0], num_recoil_bins))
        integral_muon_uldm = np.zeros((self.time_weights.shape[0], num_recoil_bins))
        # Compute the electron and muon integrals
        z = 0
        for k in range(num_recoil_bins):            
            if k < num_recoil_bins - len(self.energy_nu) :
                integral_electron_uldm[:,k] = np.trapz(-p_uldm_electron * self.spectrum_nu * self.cs_electron[k, :], self.energy_nu, axis=1)
                integral_muon_uldm[:,k] = np.trapz( p_uldm_muon * self.spectrum_nu * self.cs_muon[k, :], self.energy_nu, axis=1)
                integral_electron[:,k] = np.trapz(p_msw * self.spectrum_nu * self.cs_electron[k, :], self.energy_nu, axis=1)
                integral_muon[:,k] = np.trapz( (1 - p_msw) * self.spectrum_nu * self.cs_muon[k, :], self.energy_nu, axis=1)
            else:
                integral_electron_uldm[:,k] = np.trapz(-p_uldm_electron[:,z:] * self.spectrum_nu[z:] * self.cs_electron[k, z:], self.energy_nu[z:], axis=1)
                integral_muon_uldm[:,k] = np.trapz(p_uldm_electron[:, z:] * self.spectrum_nu[z:] * self.cs_muon[k, z:], self.energy_nu[z:], axis=1)
                integral_electron[:,k] = np.trapz(p_msw[:,z:] * self.spectrum_nu[z:] * self.cs_electron[k, z:], self.energy_nu[z:], axis=1)
                integral_muon[:,k] = np.trapz((1 - p_msw[:, z:]) * self.spectrum_nu[z:] * self.cs_muon[k, z:], self.energy_nu[z:], axis=1)
                z = z + 1

        integral_electron_recoil = np.zeros((integral_electron.shape[0], len(self.weight)))
        integral_muon_recoil = np.zeros((integral_electron.shape[0], len(self.weight)))
        integral_electron_recoil_uldm = np.zeros((integral_electron.shape[0], len(self.weight)))
        integral_muon_recoil_uldm = np.zeros((integral_electron.shape[0], len(self.weight)))
        for i in range(len(self.weight)):
            integral_electron_recoil_uldm[:,i] = np.trapz( self.weight[i] * integral_electron_uldm, self.energy_recoil, axis=1)
            integral_muon_recoil_uldm[:,i] = np.trapz( self.weight[i] * integral_muon_uldm, self.energy_recoil, axis=1)
            integral_electron_recoil[:,i] = np.trapz( self.weight[i] * integral_electron, self.energy_recoil, axis=1)
            integral_muon_recoil[:,i] = np.trapz( self.weight[i] * integral_muon, self.energy_recoil, axis=1)

        if self.resolution_correction:
            self.pridection = np.mean((integral_electron_recoil + integral_muon_recoil) * (1 + 2 * ECCENTRICITY * np.cos(self.theta_p))[:,np.newaxis], axis=0)
            return np.mean((integral_electron_recoil + integral_muon_recoil) * (1 + 2 * ECCENTRICITY * np.cos(self.theta_p))[:,np.newaxis] / self.unoscillated_term , axis=0)
            #ULDM in this sector is not implemented       
        
        elif self.efficiency_correction:
            rbar_k = (integral_electron_recoil + integral_muon_recoil)[:,0]
            rbar_k_uldm = (integral_electron_recoil_uldm + integral_muon_recoil_uldm)[:,0]

            r_k = 5.25 * rbar_k * 2 * ECCENTRICITY * np.cos(self.theta_p) / self.unoscillated_term  
            
            if 0.1 <= param_update["mdm"] <= 1:
                r_k_uldm = - 5.25 * rbar_k_uldm *  self.param["eps"] * np.cos(2 * self.theta_p - self.param["alpha_eps"] )
            
            #10^21 ev^{-1} is in order of 10 days and more
            elif param_update["mdm"] <= 2e-3: 
                r_k_uldm = 5.25 * rbar_k_uldm *  ( - ( 1 - self.param["eps"] ) * np.cos(2 * self.param["mdm"] * self.time_ev + self.param["alpha"])
                                                  - self.param["eps"] * np.cos(2 * self.theta_p - self.param["alpha_eps"] )
                                                  + self.param["eps"] * np.cos(2 * self.theta_p - self.param["alpha_eps"] ) *  np.cos(2 * self.param["mdm"] * self.time_ev + self.param["alpha"])
                ) 
            else:
                return None
            return r_k + r_k_uldm

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
    

                