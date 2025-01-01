import numpy as np

from framework_pkg.framework import FrameWork
from framework_pkg.survival_probablity import MSW

def MSWSpectrum(param = {'SinT12': 0.308, 'M12': 6.9e-5}):
    """
    Comparsion between MSW and unoscilation spectrum.
    """
    frame = FrameWork(resolution_correction=True, masked_val=2.5)
    spectrum_data = np.loadtxt('./Data/B8_SuperK_Spectrum_2023.txt')
    energy_obs  = spectrum_data[:,0:2]
    resp_func   = frame._response_function(energy_obs, frame.energy_recoil)

    borom_unoscilated_spectrum = frame._compute_unoscilated_signal(frame.energy_recoil, frame.energy_nu, frame.spectrum_nu, energy_obs, frame.cs_electron, resp_func)

    total_volume = 22.5
    SNO_norm = 1e-4 * frame.norm

    b_un_per_day = total_volume * SNO_norm * frame.target_number * borom_unoscilated_spectrum

    frame.param.update(param)
    survival_probablity = MSW(frame.param,frame.energy_nu)

    integral_electron = np.zeros(len(frame.energy_recoil))
    integral_muon  = np.zeros(len(frame.energy_recoil))

    for k in range (len(frame.energy_recoil)):
        integral_electron[k] = np.trapz(frame.spectrum_nu[k:] * frame.cs_electron[k,k:] * survival_probablity[k:] ,frame.energy_nu[k:])
        integral_muon[k] = np.trapz(frame.spectrum_nu[k:] * frame.cs_muon[k,k:] * (1-survival_probablity[k:]), frame.energy_nu[k:])

    integral_electron_recoil = np.zeros(len(energy_obs))
    integral_muon_recoil = np.zeros(len(energy_obs))
    for i in range (len(energy_obs)):
        integral_electron_recoil[i] = np.trapz(integral_electron * resp_func[i], frame.energy_recoil)
        integral_muon_recoil[i] = np.trapz(integral_muon * resp_func[i], frame.energy_recoil)
    
    spectrum_event_per_day = total_volume * SNO_norm * frame.target_number * (integral_electron_recoil + integral_muon_recoil)
    return  spectrum_event_per_day , b_un_per_day