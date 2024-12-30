import numpy as np

from datetime import datetime
from skyfield.api import load, utc

from framework_pkg.survival_probablity import PseudoDirac

# Global Constants
FERMI_CONSTANT = 1.166  # e-11 MeV^-2
HBAR_C = 1.97  # e-11 MeV cm
HBARC_FERMI_CONSTANT = 1.97 * 1.166  #x 1e-11 MeV.cm x 1e-11 MeV^-2
ELECTRON_MASS = 0.511  # MeV
WEAK_MIXING_ANGLE = 0.2315
RHO = 1.0126
ALPHA = 1 / (137 * np.pi)  # Fine-structure constant (radiative correction term)

time_scale = load.timescale()  # Create a timescale object

class FrameWork:
    """
    Computes B8 prediction in unit of [10^6 cm^-2 s^-1] at each day from initial to final date,
    according to the Super-Kamiokande experiment response function.
    
    Parameters:
    - resolution_correction: considering the Super-Kamiokande response function. It should be True in case of spectrum analsys (default is False).
    - do_binning: Whether to bin the data (default is True).
    - masked_val: Mask neutrino energy less than masked_val (default is 2 Mev).
    - first_day: Start date in the format 'year,month,day' (default is '2008,9,15').
    - last_day: End date in the format 'year,month,day' (default is '2018,5,30').
    """

    def __init__(self, resolution_correction=False, do_binning=True, masked_val=2.5, first_day='2008,9,15', last_day='2018,5,30'):
        self.resolution_correction = resolution_correction
        
        self.firstday = self._parse_date(first_day)
        self.lastday = self._parse_date(last_day)
        self.total_days = int(self.lastday - self.firstday)
        
        # Neutrino flux normalization from SNO
        self.norm = 5.25  # x 10^6 cm^-2 s^-1
        self.target_number = 3.3 # x 10^32 per kilo ton
        
        
        # Load neutrino energy spectrum (B8 spectrum)
        spectrumB8 = np.loadtxt('./Spectrum/B8_spectrum.txt')
        
        mask = spectrumB8[:, 0] >= masked_val  # Only consider energies >= 2 MeV
        self.spectrum_nu = spectrumB8[mask, 1]
        self.energy_nu = spectrumB8[mask, 0]

        
        # Calculate recoil energy (in MeV)
        self.energy_recoil = self.energy_nu / (1 + ELECTRON_MASS / (2 * self.energy_nu))
        
        # Load Super-Kamiokande Data event
        self.data = np.loadtxt('./Data/sksolartimevariation5804d.txt')
        self.data[:, :3] /= (60. * 60. * 24.)  # Convert time columns to days
        
        t0 = time_scale.utc(datetime(1970, 1, 1, 0, 0, 0, tzinfo=utc))
        zeroday = self.firstday.tt - t0.tt
        self.data = self.data[self.data[:, 0] - self.data[:, 1] >= zeroday]
        self.data[:, 0] -= zeroday
        
        if do_binning:
            self.data = self._bin_data(self.data)
        self.distance = np.sqrt(self.data[:, 0])
        
        # Geometric characteristic: Sun-Earth distance resolution (1 day)
        self.resolution = 1
        self.distance_high_resolution, self.day_high_resolution = self._sun_earth_distance(self.firstday, self.total_days, self.resolution)
        
        # Super-Kamiokande detector response function
        self.energy_obs = np.array([[4.5, 19.5]])
        self.resp_func = self._response_function(self.energy_obs, self.energy_recoil)
        
        # neutrino electron elastic scattering cross section
        self.cs_electron = self._compute_cross_section(self.energy_nu,self.energy_recoil,1)
        self.cs_muon = self._compute_cross_section(self.energy_nu,self.energy_recoil,-1)
        
        # Unoscillated signal calculation
        self.borom_unoscilated_total = self._compute_unoscilated_signal(
            self.energy_recoil, self.energy_nu, self.spectrum_nu, self.energy_obs, self.cs_electron, self.resp_func
        )
        
        # Default parameters
        #self.param = {'T12': 34, 'T13': 8.57, 'mum1': 0., 'mum2': 0, 'mum3': 0., 'M12': 7.54e-5}
        self.param = {'SinT12': 0.319, 'T13': 8.57, 'mum1': 0., 'mum2': 0, 'mum3': 0., 'M12': 7.54e-5}
                
    def __getitem__(self, param_update):
        """
        Updates parameters and computes survival probabilities.
        
        Parameters:
        - param_update: Tuple containing updates for T12, mum2, and M12.
        
        Returns:
        - Computed results based on updated parameters.
        """
        t12, mum2, m12 = param_update
        
        self.param.update({'SinT12': t12, 'mum2': mum2, 'M12': m12})
        
        survival_prob, sterile_prob = PseudoDirac(self.param, self.distance, self.energy_nu)
        
        r = np.zeros((len(self.distance),self.energy_recoil.shape[0]))
        
        for z,ts in enumerate(self.energy_recoil):
            if (len(self.energy_nu) - len(self.energy_nu[z:]))/len(self.energy_nu) >= 0.8 :
                r[:,z] = np.trapz(self.spectrum_nu * (self.cs_electron[z,:] * survival_prob + self.cs_muon[z,:] * (1 - survival_prob - sterile_prob)), self.energy_nu, axis=1) - np.trapz(self.spectrum_nu[:z] * (self.cs_electron[z,:z] * survival_prob[:,:z] + self.cs_muon[z,:z] * (1 - survival_prob[:,:z] - sterile_prob[:,:z])), self.energy_nu[:z], axis=1)
            else:
                r[:,z] = np.trapz(self.spectrum_nu[z:] * (self.cs_electron[z,z:] * survival_prob[:,z:] + self.cs_muon[z,z:] * (1 - survival_prob[:,z:] - sterile_prob[:,z:])), self.energy_nu[z:],axis=1)
            
        if self.resolution_correction:
            self.flux_fraction_prediction = (self.norm/self.borom_unoscilated_total) * np.trapz( r * self.resp_func,self.energy_recoil, axis=1)/self.distance**2
        else:
            self.flux_fraction_prediction = (self.norm/self.borom_unoscilated_total) * np.trapz( r , self.energy_recoil, axis=1)/self.distance**2
        return self.flux_fraction_prediction
    
    def _parse_date(self, date_str):
        """Parse a date string in 'year,month,day' format and return the Skyfield utc date."""
        year, month, day = map(int, date_str.split(','))
        date = datetime(year, month, day, 0, 0, 0, tzinfo=utc)
        return time_scale.utc(date)
    
    def _bin_data(self, data):
        """Bins the data based on some criteria."""
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
    
    def _sun_earth_distance(self, start_date, total_days, resolution):
        """Calculate Sun-Earth distance over a period."""
        
        """Load the JPL ephemeris DE421 (covers 1900-2050).
        https://ui.adsabs.harvard.edu/abs/2019ascl.soft07024R """
    
        planets     = load('./JPL_ephemeris/de421.bsp')
        sun,earth   = planets['sun'],planets['earth']
        t_array     = np.arange(0,total_days,resolution)
        dtheory_sun = np.zeros(len(t_array))
        day_sun     = np.zeros(len(t_array))
     
        for i,dt in enumerate(t_array):
            tstep = start_date + dt
            
            astrometric_sun    = earth.at(tstep).observe(sun)
            lat, lon, distance = astrometric_sun.radec()
            dtheory_sun[i]     = distance.au
            
            day_sun[i]         = np.mod(dt,365.25)/365.25   # :)

        day_sun -= day_sun[dtheory_sun==np.min(dtheory_sun)]
        day_sun[day_sun<0] += 1
        return dtheory_sun, day_sun
    
    def _response_function(self, energy_obs, energy_recoil):
        """Compute the detector's response function."""
        r   = np.zeros((len(energy_obs),len(energy_recoil)))
        for j in range (len(energy_obs)):
            e_nu = np.linspace(energy_obs[j,0],energy_obs[j,1])
            for i,t in enumerate(energy_recoil):
                sig  = -0.05525 + 0.3162 * np.sqrt( t + ELECTRON_MASS ) + 0.04572 * ( t + ELECTRON_MASS )
                a    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-e_nu)**2/sig**2)
                r[j,i] = np.trapz(a,e_nu)
        return r
    
    def _compute_unoscilated_signal(self, energy_recoil, energy_nu, spectrum_nu, energy_obs, cs_electron, resp_func):
        from scipy import interpolate
        """Compute the unoscillated signal. The is unit of 10^{-45} cm^2"""
        r         = np.zeros(energy_recoil.shape)
        num_event = np.zeros(len(energy_obs))
        
        for z, ts in enumerate(energy_recoil):
            if (len(energy_nu) - len(energy_nu[z:]))/len(energy_nu) >= 0.8 :
                r[z] = np.trapz(spectrum_nu*cs_electron[z,:],energy_nu) - np.trapz(spectrum_nu[:z]*cs_electron[z,:z],energy_nu[:z])
            else:
                r[z] = np.trapz(spectrum_nu[z:] * cs_electron[z,z:], energy_nu[z:])
                
        if self.resolution_correction:
            for i in range(len(energy_obs)):
                num_event[i] = np.trapz( r * resp_func[i], energy_recoil)
        else:
            for i in range(len(energy_obs)):
                num_event[i] = np.trapz( r , energy_recoil)
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
