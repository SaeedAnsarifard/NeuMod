import numpy as np

from datetime import datetime
from skyfield.api import load, utc

"""Create a timescale."""
ts     = load.timescale()
"""Fermi constant :  1.166 \times 10^{-11}/Mev^2"""
f_c    = 1.166
"""Electron mass : 0.511 Mev"""
m_e    = 0.511
"""h_{bar}c : 1.97 10^{-11} Mev.cm"""
hbarc  = 1.97
"""proton mass :  1.67 \times 10^{-27} kg"""
m_p    = 1.67
g      = hbarc * f_c
    
class FrameWork(object):
    """Computing B8 total event rate at each day from day initial to day final in counts per day per kilo ton, according to the Super-Kamiokande experiment response function.
    inputs are date for initial and final days, in the format of 'year,month,day' """
    
    def __init__(self, first_day='2008,9,15', last_day= '2018,5,30'):
        firstdate = first_day.split(',')
        lastdate  = last_day.split(',')
        
        firstdate = datetime(int(firstdate[0]),int(firstdate[1]),int(firstdate[2]),0,0,0,tzinfo=utc)
        lastdate  = datetime(int(lastdate[0]),int(lastdate[1]),int(lastdate[2]),0,0,0,tzinfo=utc)
        
        self.firstday= ts.utc(firstdate)
        self.lastday = ts.utc(lastdate)
        self.total_days = int(self.lastday-self.firstday)
        
        self.survival_probablity = np.zeros(())

        
        """ Nutrino Flux normalization :    from SNO"""
        self.norm  = {'B8' : 5.25e-4}   #\times 10^{10}
        
        """ Neutrino production point weight function : http://www.sns.ias.edu/~jnb/"""
        load_phi   = np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack = True)
        self.phi   = {'B8' : load_phi[6,:]}
        
        """ Electron density inside sun 10^{23}/cm^{3}"""
        self.n_e  = 6*10**load_phi[2,:]
        
        """ Neutrino energy spectrum : http://www.sns.ias.edu/~jnb/"""
        spectrumB8    = np.loadtxt('./Spectrum/B8_spectrum.txt')
        self.spectrum = {'B8' : spectrumB8[:,1]}
        
        """ Neutrino energy in Mev"""
        self.energy_nu       = {'B8' : spectrumB8[:,0]}
        
        """ Electron recoil energy in Mev"""
        self.uppt          = 100
        self.energy_recoil = {'B8'  : IntegralLimit(spectrumB8[:,0],m_e,uppt=self.uppt)}
        
#        """ Super-Kamiokande Data event (count per year per  kilo ton) :"""
#        self.su_nbin  = su_nbin
#        self.data_su  = np.loadtxt('./Data/B8_Data_2020.txt')[:self.su_nbin,:]
        
        """ geometric characteristic: resolution of d, the distance between sun and earth is considered to be one day """
        self.resolution = 1
        self.distance   = SunEarthDistance(self.firstday,self.total_days,self.resolution)
        
        self.survival_probablity = np.zeros((self.total_days,len(self.e_nu['B8'])))
        self.sterile_probablity  = np.zeros((self.total_days,len(self.e_nu['B8'])))
        
        """ Super-Kamiokande detector response function: PhysRevD.109.092001 """
        self.energy_obs = np.array([[3.5,20.0]])
        self.resp_func = ResSu(self.energy_obs,self.t_e['B8'])
        

        
#        """ Unoscilated signal is produced to compare with the Super-Kamiokande results. For more info see their papers!
#        number of target per kilo ton per year   :  365.25 * 24. * 6.0 * 6.0 * (10/18) \times 10^{6}/m_p -> 0.33 \times 10^{27} """
#        self.detector = 365.25 * 24. * 6. * 6. * (10/18) * 1/m_p
#        borom_spec,borom_total = BoromUnoscilated(self.t_e['B8'][0],self.e_nu['B8'][0],self.spec['B8'][0],g,m_e,self.uppt,self.data_su,self.res)
#        self.borom_unoscilated_spectrum = self.det_su * (2 * np.pi/self.year) * 5.25e-4 * (self.a**2/self.h) * borom_spec
#        self.borom_unoscilated_total    = self.det_su * (2 * np.pi/self.year) * 5.25e-4 * (self.a**2/self.h) * borom_total
#
    def __getitem__(self,getitem_pos):
        """ input is an array of survival probability. Its shape is (d,e) d is the number of days from initial day to the final day and e is the number of neutrino spectrum energy bin """
        survival_probablity_update,sterile_probablity_update = getitem_pos
        
        if (survival_probablity_update.shape != self.survival_probablity.shape):
            raise Exception("Survival Probablity shape not match")
        else :
            self.survival_probablity = survival_probablity_update
            
        if sterile_probablity_update == 0:
            sterile_probablity = 0
        else :
            if (sterlie_probablity_update.shape != self.sterile_probablity.shape):
                raise Exception("Sterile Probablity shape not match")
            else :
                self.sterile_probablity = sterile_probablity_update
        
        r = np.zeros((self.d.shape[0],t.shape[0]))
        k = 0
        for z,ts in enumerate(self.t_e['B8']):
            if z<=self.uppt:
                cse    = DCS(g,m_e,e,ts,1)
                csmu   = DCS(g,m_e,e,ts,-1)
                r[:,z] = np.trapz(self.spec['B8'] * (cse*self.survival_probablity + csmu * (1-self.survival_probablity - self.sterile_probablity)),self.e_nu['B8'],axis=1)
            else:
                cse    = DCS(g,m_e,e[k:],ts,1)
                csmu   = DCS(g,m_e,e[k:],ts,-1)
                r[:,z] = np.trapz(self.spec['B8'][k:] * (cse*self.survival_probablity[:,k:] + csmu * (1 - self.survival_probablity[:,k:] - self.sterile_probablity[:,k:])),self.e_nu['B8'][k:],axis=1)
                k      = k + 1
        
        self.dr_dldt = (self.norm['B8']/self.d)[:,np.newaxis] * r #number of event per each delta theta per electron recoil times 10^{-35}
        return self.dr_dldt

def SunEarthDistance(t_initial,t_total,resolution):
    """Load the JPL ephemeris DE421 (covers 1900-2050).
    https://ui.adsabs.harvard.edu/abs/2019ascl.soft07024R """
    
    planets     = load('de421.bsp')
    sun,earth   = planets['sun'],planets['earth']
    t_array     = np.arange(0,t_total,resolution)
    dtheory_sun = np.zeros(t_array.shape[0])
    
    for i,dt in enumerate(t_array):
        tstep = t_initial + dt
        
        astrometric_sun    = earth.at(tstep).observe(sun)
        lat, lon, distance = astrometric_sun.radec()
        dtheory_sun[i]     = distance.au
    
    return dtheory_sun

def IntegralLimit(e,m_e,lowt=-4,uppt=100):
    mint = np.min(e)
    maxt = np.max(e)
    mint = np.log10(mint/(1+m_e/(2*mint)))
    return np.concatenate((np.logspace(lowt,mint,uppt),e[1:]/(1+m_e/(2*e[1:]))))
        
def DCS(g, m_e, e_nu, t_e, i=1):
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
    r   = np.zeros((data.shape[0],t_e.shape[0]))
    for j in range (data.shape[0]):
        e_nu = np.linspace(data[j,0],data[j,1])
        for i,t in enumerate(t_e):
            sig  = (-0.05525+0.3162*np.sqrt(t)+0.04572*t)
            a    = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-0.5*(t-e_nu)**2/sig**2)
            r[j,i] = np.trapz(a,e_nu)
    return r
    
def BoromUnoscilated(t,e,sp,g,m_e,uppt,data_su,res):
    r         = np.zeros(t.shape)
    num_event = np.zeros(len(data_su))
    k         = 0
    for z,ts in enumerate(t):
        if z<=uppt:
            cse  = DCS(g,m_e,e,ts,1)
            r[z] = np.trapz(sp*cse,e)
        else:
            cse  = DCS(g,m_e,e[k:],ts,1)
            r[z] = np.trapz(sp[k:]*cse,e[k:])
            k    = k + 1
            
    for i in range(len(data_su)):
        num_event[i] = np.trapz(r*res[i],t)
        
    res_tot = ResSu(np.array([[data_su[0,0],data_su[-1,1]]]), t)
    return num_event,np.trapz(r*res_tot[0],t)

def SuperkTotalEventPrediction(dr_dldt,frame):
    year     = frame.year
    theta    = frame.theta
    detector = frame.det_su
    len_m12  = len(dr_dldt)
    res = ResSu(np.array([[frame.data_su[0,0],frame.data_su[-1,1]]]), frame.t_e['B8'][0])
    num_event= np.zeros((len_m12))
    for i in range(len_m12):
        dr_dl        =  detector * np.trapz(dr_dldt[i]['B8'][0]*res[0],frame.t_e['B8'][0],axis=1)
        num_event[i] = np.trapz(dr_dl,theta,axis=0)/year
    return num_event
    
def SuperkSpectrumEventPrediction(dr_dldt,t,year,theta,detector,b8_un,res):
    num_event = np.zeros((theta.shape[0],b8_un.shape[0]))
    for i in range(b8_un.shape[0]):
        num_event[:,i] = (detector/b8_un[i]) * np.trapz(dr_dldt*res[i],t,axis=1)
    return np.trapz(num_event,theta,axis=0)/year

def AveragedPerdiction(dr_dldt,frame):
    t_e        = frame.t_e
    year       = frame.year
    theta      = frame.theta
    det_su     = frame.det_su
    b8_un      = frame.borom_unoscilated_spectrum
    res        = frame.res
    components = frame.components
    len_m12    = len(dr_dldt)
    pred_bo    = np.zeros((len_m12,3))
    pred_su    = np.zeros((len_m12,b8_un.shape[0]))
    for i in range(len_m12):
        #Borexino
        for k,c in enumerate (components[:-1]):
            pred_bo[i,k] = BorexinoTotalEventPrediction(dr_dldt[i][c],t_e[c],year,theta)
        #SuperKamiokande
        pred_su[i] = SuperkSpectrumEventPrediction(dr_dldt[i]['B8'][0],t_e['B8'][0],year,theta,det_su,b8_un,res)
    return pred_bo,pred_su
