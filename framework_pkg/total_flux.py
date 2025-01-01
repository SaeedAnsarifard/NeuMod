import numpy as np

from framework_pkg.framework import FrameWork
from framework_pkg.survival_probablity import MSW , PseudoDirac

class SuperKFlux:
    """
    Class for computing and comparing the Super-Kamiokande event spectrum.
    """
    def __init__(self, first_day='2008,9,15', last_day='2018,5,30'):
        # Initialize the framework with necessary parameters
        resolution_correction = False
        masked_val = 0.1
        self.frame = FrameWork(resolution_correction, masked_val, first_day, last_day )
        
        #self.total_volume = 22.5  # Total detector volume in kilotons
        self.SNO_norm = self.frame.norm
        self.distance = self.frame.distance_list

        # Load modulation data from file
        self.modulation_data = np.loadtxt('./Data/sksolartimevariation5804d.txt')
        self.modulation_data[:, :3] /= (60. * 60. * 24.)  # Convert time columns to days


        #self.spectrum_data = np.loadtxt('./Data/B8_SuperK_Spectrum_2023.txt')
        #self.energy_obs = self.spectrum_data[:, :2]
        zeroday = self.frame.zeroday
        self.modulation_data = self.modulation_data[self.modulation_data[:, 0] - self.modulation_data[:, 1] >= zeroday]
        self.modulation_data[:, 0] -= zeroday
        
        # if do_binning:
        #     self.data = self._bin_data(self.data)
        # self.distance = np.sqrt(self.data[:, 0])

        # Compute response function and unoscillated flux
        self.energy_obs = np.array([[4.5, 19.5]])
        self.response_function = self.frame._response_function(self.energy_obs, self.frame.energy_recoil)
        self.unoscillated_flux = self.frame._compute_unoscilated_signal(
            self.frame.energy_recoil,
            self.frame.energy_nu,
            self.frame.spectrum_nu,
            self.energy_obs,
            self.frame.cs_electron,
            self.response_function,
        )


        # Compute unoscillated events per day
        # self.unoscillated_event = (
        #     self.total_volume
        #     * self.SNO_norm
        #     * self.frame.target_number
        #     * self.unoscillated_spectrum
        # )

    def __getitem__(self, param_update, name="MSW"):
        """
        Compare the oscilated and unoscillated spectra given updated parameters.

        Parameters:
            param_update (dict): Dictionary containing updated parameter values.

        Returns:
            np.ndarray with shape [l,m]: Spectrum of events per day after applying oscillation effects.
            l is number of distance bins and m is number of energy bins
        """

        # Compute survival probability using the specified method
        if name == "MSW":
            self.frame.param.update(param_update)
            survival_probability = MSW(self.frame.param, self.frame.energy_nu)
            appearance = (survival_probability)[np.newaxis]
            disappearance = (1 - survival_probability)[np.newaxis]
        elif name == "PseudoDirac":
            if "mum1" not in param_update:
                param_update["mum1"] = 0  # Set a default value for param3
            if "mum2" not in param_update:
                param_update["mum2"] = 1.5  # Set a default value for param3
            if "mum3" not in param_update:
                param_update["mum3"] = 0  # Set a default value for param3

            self.frame.param.update(param_update)
            survival_probability, sterile_probability = PseudoDirac(self.frame.param, self.distance, self.frame.energy_nu)
            appearance = survival_probability
            disappearance = 1 - survival_probability - sterile_probability
            appearance = survival_probability / self.distance[:,np.newaxis]**2
            disappearance = (1 - survival_probability - sterile_probability)/self.distance[:,np.newaxis]**2
        else:
            raise ValueError(f"Unsupported survival probability method: {name}")

        # Initialize integral arrays
        num_recoil_bins = len(self.frame.energy_recoil)
        integral_electron = np.zeros((appearance.shape[0], num_recoil_bins))
        integral_muon = np.zeros((appearance.shape[0],num_recoil_bins))

        # Compute the electron and muon integrals
        for k in range(num_recoil_bins):
            integral_electron[:,k] = np.trapz(
                self.frame.spectrum_nu[k:] * self.frame.cs_electron[k, k:] * appearance[:,k:],
                self.frame.energy_nu[k:],
            )
            integral_muon[:,k] = np.trapz(
                self.frame.spectrum_nu[k:] * self.frame.cs_muon[k, k:] * disappearance[:,k:],
                self.frame.energy_nu[k:],
            )

        # Map integrals to observed energy bins
        num_obs_bins = len(self.energy_obs)
        integral_electron_recoil = np.zeros((appearance.shape[0], num_obs_bins))
        integral_muon_recoil = np.zeros((appearance.shape[0], num_obs_bins))

        for i in range(num_obs_bins):
            integral_electron_recoil[:,i] = np.trapz( integral_electron, self.frame.energy_recoil)
            
            integral_muon_recoil[:,i] = np.trapz( integral_muon, self.frame.energy_recoil)
            

        # Compute spectrum events per day with oscillations
        oscilated_flux = ((integral_electron_recoil + integral_muon_recoil)/self.unoscillated_flux )
        return self.SNO_norm * oscilated_flux