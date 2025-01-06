import numpy as np

from framework_pkg.framework import FrameWork
from framework_pkg.survival_probablity import MSW , PseudoDirac

class SuperKFlux:
    """
    Class for computing and comparing the Super-Kamiokande event spectrum.
    """
    def __init__(self, threshold=3.5,  first_day='2008,9,15', last_day='2018,5,30'):
        # Initialize the framework with necessary parameters
        resolution_correction = False
        self.frame = FrameWork(resolution_correction, first_day, last_day)
        
        # Total detector volume in kilotons
        self.SNO_norm = self.frame.norm
        self.distance = self.frame.distance_list

        # Load modulation data from file
        self.modulation_data = np.loadtxt('./Data/sksolartimevariation5804d.txt')
        self.modulation_data[:, :3] /= (60. * 60. * 24.)  # Convert time columns to days

        zeroday = self.frame.zeroday
        self.modulation_data = self.modulation_data[self.modulation_data[:, 0] - self.modulation_data[:, 1] >= zeroday]
        self.modulation_data[:, 0] -= zeroday
        
        # Compute response function and unoscillated flux
        self.energy_obs = np.array([[threshold, 19.5]])
        self.response_function = self.frame._response_function(self.energy_obs, self.frame.energy_recoil)
        self.unoscillated_flux = self.frame._compute_unoscilated_signal(
            self.frame.energy_recoil,
            self.frame.energy_nu,
            self.frame.spectrum_nu,
            self.energy_obs,
            self.frame.cs_electron,
            self.response_function,
        )
        
        # Default parameters
        self.param = {'SinT12': 0.319, 'T13': 8.57, 'M12': 7.54e-5}

    def __getitem__(self, param_update, name="MSW"):
        """
        Compare the oscilated and unoscillated total events given updated parameters.

        Parameters:
            param_update (dict): Dictionary containing updated parameter values.

        Returns:
            flux in cm^-2 s^-1 times 10^6 for electron recoil larger than masked_val 
            in the period of first_day and last_day
        """

        # Compute survival probability using the specified method
        if name == "MSW":
            self.param.update(param_update)
            survival_probability = MSW(self.param, self.frame.energy_nu)
            appearance = survival_probability * np.mean(1 / self.distance**2)
            disappearance = (1 - survival_probability) * np.mean(1 / self.distance**2)
        elif name == "PseudoDirac":
            if "mum1" not in param_update:
                param_update["mum1"] = 0  # Set a default value for param3
            if "mum2" not in param_update:
                param_update["mum2"] = 1.5  # Set a default value for param3
            if "mum3" not in param_update:
                param_update["mum3"] = 0  # Set a default value for param3

            self.param.update(param_update)
            survival_probability, sterile_probability = PseudoDirac(self.param, self.distance, self.frame.energy_nu)
            appearance = np.mean(survival_probability / self.distance[:,np.newaxis]**2, axis=0)
            disappearance = np.mean((1 - survival_probability - sterile_probability) / self.distance[:,np.newaxis]**2, axis= 0 )
        else:
            raise ValueError(f"Unsupported survival probability method: {name}")

        # Initialize integral arrays
        num_recoil_bins = len(self.frame.energy_recoil)
        integral_electron = np.zeros((num_recoil_bins))
        integral_muon = np.zeros((num_recoil_bins))

        # Compute the electron and muon integrals
        for k in range(num_recoil_bins):
            integral_electron[k] = np.trapz(
                self.frame.spectrum_nu[k:] * self.frame.cs_electron[k, k:] * appearance[k:],
                self.frame.energy_nu[k:],
            )
            integral_muon[k] = np.trapz(
                self.frame.spectrum_nu[k:] * self.frame.cs_muon[k, k:] * disappearance[k:],
                self.frame.energy_nu[k:],
            )


        # Map integrals to observed energy bins
        num_obs_bins = len(self.energy_obs)
        integral_electron_recoil = np.zeros((num_obs_bins))
        integral_muon_recoil = np.zeros((num_obs_bins))

        for i in range(num_obs_bins):
            integral_electron_recoil[i] = np.trapz(
                self.response_function[i] * integral_electron , self.frame.energy_recoil
            )
            integral_muon_recoil[i] = np.trapz(
                self.response_function[i] * integral_muon, self.frame.energy_recoil
            )

        # integral_electron_recoil = np.trapz( integral_electron * , self.frame.energy_recoil)
        # integral_muon_recoil = np.trapz( integral_muon, self.frame.energy_recoil)
            
        # Compute total events
        oscilated_flux = ((integral_electron_recoil + integral_muon_recoil)/self.unoscillated_flux )
        return self.SNO_norm * oscilated_flux