import numpy as np

from datetime import datetime
from skyfield.api import load, utc


global g , m_e , time_scale

""" h_{bar}c : 1.97 10^{-11} Mev.cm , Fermi constant :  1.166 \times 10^{-11}/Mev^2 """
g  = 1.97 * 1.166

"""Electron mass : 0.511 Mev"""
m_e = 0.511

"""Create a timescale."""
time_scale  = load.timescale()


class FrameWork(object):
    """Computing B8 prediction in unit of [10^-6 cm^-2 s^-1] at each day from day initial to day final in counts per day per kilo ton, according to the Super-Kamiokande experiment response function.
    inputs are date for initial and final days, in the format of 'year,month,day' """
    
    def __init__(self, first_day='2008,9,15', last_day= '2018,5,30'):
        firstdate = first_day.split(',')
        lastdate  = last_day.split(',')
        
        firstdate = datetime(int(firstdate[0]),int(firstdate[1]),int(firstdate[2]),0,0,0,tzinfo=utc)
        lastdate  = datetime(int(lastdate[0]),int(lastdate[1]),int(lastdate[2]),0,0,0,tzinfo=utc)
        
        self.firstday   = time_scale.utc(firstdate)
        self.lastday    = time_scale.utc(lastdate)
        self.total_days = int(self.lastday-self.firstday)
        
        """ Nutrino Flux normalization :    from SNO"""
        self.norm  = 5.25   #\times 10^{6}
                
        """ Neutrino energy spectrum : http://www.sns.ias.edu/~jnb/"""
        spectrumB8       = np.loadtxt('./Spectrum/B8_spectrum.txt')
        self.spectrum_nu = spectrumB8[:,1]
        
        """ Neutrino energy in Mev"""
        self.energy_nu   = spectrumB8[:,0]
        
        """ Electron recoil energy in Mev"""
        self.uppt          = 100
        self.energy_recoil = IntegralLimit(spectrumB8[:,0],uppt=self.uppt)
        
        """ Super-Kamiokande Data event (PhysRevLett.132.241803) :"""
        self.data      = np.loadtxt('./Data/sksolartimevariation5804d.txt')
        self.data[:,0] = self.data[:,0]/(60.*60.*24.)
        self.data[:,1] = self.data[:,1]/(60.*60.*24.)
        self.data[:,2] = self.data[:,2]/(60.*60.*24.)
        
        t0        = time_scale.utc(datetime(1970,1,1,0,0,0,tzinfo=utc))
        zeroday   = self.firstday.tt - t0.tt
        self.data = self.data[self.data[:,0]-self.data[:,1]>=zeroday]
        self.data[:,0] = self.data[:,0] - zeroday
        
        """ geometric characteristic: resolution of d, the distance between sun and earth is considered to be one day """
        self.resolution = 1
        self.distance, self.day  = SunEarthDistance(self.firstday,self.total_days,self.resolution)
        
        #self.survival_probablity = np.zeros((self.total_days,len(self.energy_nu)))
        #self.sterile_probablity  = np.zeros((self.total_days,len(self.energy_nu)))
        
        """ Super-Kamiokande detector response function: PhysRevD.109.092001 """
        self.energy_obs = np.array([[4.5,19.5]])
        self.resp_func = ResSu(self.energy_obs,self.energy_recoil)
        
        """ Unoscilated signal is produced to compare with the Super-Kamiokande results. the unit is [10^-45 cm^2]. For more info see their papers! """
        self.borom_unoscilated_total = BoromUnoscilated(self.energy_recoil,self.energy_nu,self.spectrum_nu,g,self.uppt,self.energy_obs,self.resp_func)
        

    def __getitem__(self,getitem_pos):
        """ input is an array of survival probability. Its shape is (d,e) d is the number of days from initial day to the final day and e is the number of neutrino spectrum energy bin """
        
        survival_probablity, sterile_probablity, distance = getitem_pos
    
        r = np.zeros((distance.shape[0],self.energy_recoil.shape[0]))
        k = 0
        for z,ts in enumerate(self.energy_recoil):
            if z<=self.uppt:
                cse    = DCS(self.energy_nu,ts,1)
                csmu   = DCS(self.energy_nu,ts,-1)
                r[:,z] = np.trapz(self.spectrum_nu * (cse * survival_probablity + csmu * (1 - survival_probablity - sterile_probablity)), self.energy_nu,axis=1)
            else:
                cse    = DCS(self.energy_nu[k:],ts,1)
                csmu   = DCS(self.energy_nu[k:],ts,-1)
                r[:,z] = np.trapz(self.spectrum_nu[k:] * (cse * survival_probablity[:,k:] + csmu * (1 - survival_probablity[:,k:] - sterile_probablity[:,k:])), self.energy_nu[k:],axis=1)
                k      = k + 1
        
        self.flux_fraction_prediction = np.trapz((1./distance**2)[:,np.newaxis] * r * self.resp_func,self.energy_recoil,axis=1)
        return (self.norm/self.borom_unoscilated_total) * self.flux_fraction_prediction

def SunEarthDistance(t_initial,t_total,resolution):
    """Load the JPL ephemeris DE421 (covers 1900-2050).
    https://ui.adsabs.harvard.edu/abs/2019ascl.soft07024R """
    
    planets     = load('de421.bsp')
    sun,earth   = planets['sun'],planets['earth']
    t_array     = np.arange(0,t_total,resolution)
    dtheory_sun = np.zeros(t_array.shape[0])
    day_sun     = np.zeros(t_array.shape[0])
 
    for i,dt in enumerate(t_array):
        tstep = t_initial + dt
        
        astrometric_sun    = earth.at(tstep).observe(sun)
        lat, lon, distance = astrometric_sun.radec()
        dtheory_sun[i]     = distance.au
        
        day_sun[i]         = np.mod(dt,365.25)/365.25   # :)

    day_sun = day_sun - day_sun[dtheory_sun==np.min(dtheory_sun)]
    day_sun[day_sun<0] = day_sun[day_sun<0] + 1
    return dtheory_sun, day_sun

def IntegralLimit(e, lowt=-4, uppt=100):
    mint = np.min(e)
    maxt = np.max(e)
    mint = np.log10(mint/(1+m_e/(2*mint)))
    return np.concatenate((np.logspace(lowt,mint,uppt),e[1:]/(1+m_e/(2*e[1:]))))
        
def DCS(e_nu, t_e, i=1):
    #dsigma/dT_e as function of T_e and E_nu (electron recoil and neutrino energy)

    #weak mixing angle = 0.22342 : https://pdg.lbl.gov/2019/reviews/rpp2019-rev-standard-model.pdf
    sw    = 0.2315

    #Bahcall, John N., Marc Kamionkowski, and Alberto Sirlin. 
    #"Solar neutrinos: Radiative corrections in neutrino-electron scattering experiments." 
    #Physical Review D 51.11 (1995): 6146.
    rho   = 1.0126
    x     = np.sqrt(1 + 2*m_e/t_e)
    it    = (1/6) * ((1/3) + (3 - x**2) * ((x/2) * (np.log(x+1) - np.log(x-1)) - 1))
    if i == 1:
        kappa = 0.9791 + 0.0097 * it
        gl    = rho * (0.5 - kappa * sw) - 1
    if i == -1:
        kappa = 0.9970 - 0.00037 * it
        gl    = rho * (0.5 - kappa * sw)
    gr    = -rho * kappa * sw
    
    z     = t_e/e_nu
    #radiative correction we dont consider it currently
    ap  = 1/(137*np.pi)
    fm  = 0
    fp  = 0
    fmp = 0
    
    a1  = gl**2 * (1 + ap * fm)
    a2  = gr**2 * (1 + ap * fp) * (1-z)**2
    a3  = gr * gl * (1 + ap * fmp) * (m_e/e_nu) * z
    
    return  2 * g**2 * (m_e/np.pi) * (a1 + a2 - a3) * 10 #\times 10^{-45} in cm^2

def ResSu(energy_obs, energy_recoil):
    r   = np.zeros((energy_obs.shape[0],energy_recoil.shape[0]))
    for j in range (energy_obs.shape[0]):
        e_nu = np.linspace(energy_obs[j,0],energy_obs[j,1])
        for i,t in enumerate(energy_recoil):
            sig  = (-0.05525+0.3162*np.sqrt(t)+0.04572*t)
            a    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-e_nu)**2/sig**2)
            r[j,i] = np.trapz(a,e_nu)
    return r
    
def BoromUnoscilated(t, e, sp, g, uppt, data_su, res):
    r         = np.zeros(t.shape)
    num_event = np.zeros(len(data_su))
    k         = 0
    for z,ts in enumerate(t):
        if z<=uppt:
            cse  = DCS(e,ts,1)
            r[z] = np.trapz(sp*cse,e)
        else:
            cse  = DCS(e[k:],ts,1)
            r[z] = np.trapz(sp[k:]*cse,e[k:])
            k    = k + 1
            
    for i in range(len(data_su)):
        num_event[i] = np.trapz(r*res[i],t)
    return num_event

def SuperkPrediction(data,total_days,prediction,distance):
    day_array    = np.arange(0,total_days,1)
    bin_predict  = np.zeros(len(data))
    dist_predict = np.zeros(len(data))
    day_predict  = np.zeros(len(data))

    for i,day in enumerate (data[:,0]):
        condition = (day_array >= day - data[i,1]) & (day_array <= day + data[i,2])
        bin_predict[i]  = np.mean(prediction[condition])
        dist_predict[i] = np.mean(distance[condition])
        day_predict[i]  = np.mod(day,365.25)/365.25

    day_predict = day_predict - day_predict[dist_predict==np.min(dist_predict)]
    day_predict[day_predict<0] = day_predict[day_predict<0] + 1
    return bin_predict,dist_predict,day_predict


