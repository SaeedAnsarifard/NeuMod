import numpy as np

global f_c , m_e, hbarc, phi, n_e

""" Fermi constant :  1.166 \times 10^{-11}/Mev^2 """
f_c    = 1.166

""" Electron mass : 0.511 Mev """
m_e    = 0.511

""" h_{bar}c : 1.97 10^{-11} Mev.cm """
hbarc  = 1.97

""" Neutrino production point weight function : http://www.sns.ias.edu/~jnb/"""
load_phi = np.loadtxt('./Solar_Standard_Model/bs2005agsopflux1.txt', unpack = True)
phi      = load_phi[6,:]

""" Electron number density inside sun 10^{23}/cm^{3}"""
n_e  = 6*10**load_phi[2,:]

def PseudoDirac(param,enu):
    ls_au = np.linspace(0.983,1.017,75)
    ls   = 1.496e11 * ls_au
    #enu = np.logspace(-1,1.3,100)
    pel = np.zeros((ls.shape[0],enu.shape[0]))
    psl = np.zeros((ls.shape[0],enu.shape[0]))
    
    util= np.ones((n_e.shape[0],enu.shape[0]))
    ne  = np.reshape(n_e ,(n_e.shape[0],1))*util
    e   = np.reshape(enu ,(1,enu.shape[0]))*util

    ve  = 2 * np.sqrt(2) * f_c * ne * hbarc**3 * 1e-9 * e
    den = np.sqrt((param['M12'] * np.cos(2*(np.pi/180) * param['T12'])- ve)**2 + (param['M12'] * np.sin(2*(np.pi/180) * param['T12']))**2)         
    nom = param['M12'] * np.cos(2*(np.pi/180) * param['T12']) - ve
    tm  = 0.5*np.arccos(nom/den)

    sin = np.sin((np.pi/180) * param['T12'])**2 * np.cos((np.pi/180) * param['T13'])**4
    cos = np.cos((np.pi/180) * param['T12'])**2 * np.cos((np.pi/180) * param['T13'])**4

    for j,l in enumerate(ls):
        ae1 = cos * np.cos(tm)**2  * np.cos(10*param['mum1']*l/(hbarc*2*e))**2
        ae2 = sin * np.sin(tm)**2  * np.cos(10*param['mum2']*l/(hbarc*2*e))**2
        ae3 = np.sin((np.pi/180)*param['T13'])**4 * np.cos(10*param['mum3']*l/(hbarc*2*e))**2

        pee = ae1 + ae2 + ae3
        pel[j]  = np.sum(np.reshape(phi,(n_e.shape[0],1))*pee,axis=0)

        as1 = np.cos((np.pi/180) * param['T13'])**2 * np.cos(tm)**2  * np.sin(10*param['mum1']*l/(hbarc*2*e))**2
        as2 = np.cos((np.pi/180) * param['T13'])**2 * np.sin(tm)**2  * np.sin(10*param['mum2']*l/(hbarc*2*e))**2
        as3 = np.sin((np.pi/180) * param['T13'])**2 * np.sin(10*param['mum3']*l/(hbarc*2*e))**2

        pes = as1 + as2 + as3
        psl[j]  = np.sum(np.reshape(phi,(n_e.shape[0],1))*pes,axis=0)

    return pel, psl, ls_au
