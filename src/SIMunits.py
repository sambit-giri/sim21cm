import numpy as np
import importlib
from astropy import units as u
from astropy import cosmology

class TIPSYunits:
    def __init__(self, Lbox, cosmo='WMAP7', h_unit=True):
        '''
        Parameters:
            * Lbox (float): Give the boxsize of your simulation. If astropy units are not used, then the code assumes Mpc.
            * cosmo (str or astropy.cosmology): The option for string input is 'WMAP5', 'WMAP7', 'WMAP9', 'Planck13' and 'Planck15'. 
                        If you want to give custom build cosmology, see https://docs.astropy.org/en/stable/cosmology/
        '''
        cosmo_str = ['WMAP5', 'WMAP7', 'WMAP9', 'Planck13', 'Planck15']
        cosmo_all = [cosmology.WMAP5, cosmology.WMAP7, cosmology.WMAP9, cosmology.Planck13, cosmology.Planck15]
        cosmo_arg = None
        for i,cc in enumerate(cosmo_str):
            if cosmo.lower()==cc.lower(): cosmo_arg = i
        if cosmo_arg is None:
            print('Please choose from the cosmologies below:')
            print(cosmo_str)
        
        self.cosmo = cosmo_all[cosmo_arg]
        
        if h_unit:
            self.Lbox_hunits = Lbox*u.Mpc if isinstance(Lbox, (float,int)) else Lbox
            self.Lbox = self.Lbox_hunits/self.cosmo.h
        else:
            self.Lbox = Lbox*u.Mpc if isinstance(Lbox, (float,int)) else Lbox
            self.Lbox_hunits = self.Lbox*self.cosmo.h
        
        self.m_unit()
        self.v_unit()
        
    def m_unit(self):
        self.MUNIT = (self.cosmo.critical_density0*self.Lbox**3).to('solMass')
        self.MUNIT_hunits = self.MUNIT*self.cosmo.h
            
    def v_unit(self):
        self.VUNIT = self.Lbox*100/np.sqrt(8*np.pi/3)
        self.VUNIT_hunits = self.VUNIT*self.cosmo.h

