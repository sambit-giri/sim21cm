import numpy as np
import pandas as pd
from sys import argv,exit

class Tipsy:
    def __init__(self, filename):
        self.filename = filename
        self.tipsy = self.open()

        self.header_type = np.dtype([('time', '>f8'),('N', '>i4'), ('Dims', '>i4'), ('Ngas', '>i4'), ('Ndark', '>i4'), ('Nstar', '>i4'), ('pad', '>i4')])
        self.gas_type  = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'), ('rho','>f4'), ('temp','>f4'), ('hsmooth','>f4'), ('metals','>f4'), ('phi','>f4')])
        self.dark_type = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'), ('eps','>f4'), ('phi','>f4')])
        self.star_type  = np.dtype([('mass','>f4'), ('x', '>f4'),('y', '>f4'),('z', '>f4'), ('vx', '>f4'),('vy', '>f4'),('vz', '>f4'), ('metals','>f4'), ('tform','>f4'), ('eps','>f4'), ('phi','>f4')])
        self.header = self.Header()
    
    def Header(self):
        header = np.fromfile(self.tipsy,dtype=self.header_type,count=1)
        header = dict(zip(self.header_type.names,header[0]))
        return header

    def Gas(self):
        header = self.header
        gas  = np.fromfile(self.tipsy,dtype=self.gas_type,count=header['Ngas'])
        gas  = pd.DataFrame(gas,columns=gas.dtype.names)
        return gas

    def Dark(self):
        header = self.header
        dark = np.fromfile(self.tipsy,dtype=self.dark_type,count=header['Ndark'])
        dark = pd.DataFrame(dark,columns=dark.dtype.names)
        return dark

    def Star(self):
        header = self.header
        star = np.fromfile(self.tipsy,dtype=self.star_type,count=header['Nstar'])
        star = pd.DataFrame(star,columns=star.dtype.names)
        return star

    def close(self):
        self.tipsy.close()

    def open(self):
        return open(self.filename,'rb')

    def mass_conversion_factor(Lbox, rho_c=None, cosmo=None):
        import astropy.units as u
        if rho_c is None:
            if cosmo is None: from astropy.cosmology import Planck15 as cosmo
            rho_c = cosmo.critical_density0.to('solMass/Mpc**3').value
        if type(Lbox) == u.quantity.Quantity: 
            Lbox = Lbox.to('Mpc').value
        MUNIT = rho_c*Lbox**3.0
        print('Factor for converting to Msun.')
        return MUNIT

    def velocity_conversion_factor(Lbox, cosmo=None):
        import astropy.units as u
        if type(Lbox) == u.quantity.Quantity: 
            Lbox = Lbox.to('Mpc').value
        VUNIT = Lbox*100/(8*np.pi/3)**0.5
        print('Factor for converting to km/s.')
        return VUNIT

if __name__ == '__main__':
    filename = argv[1]
    tipsy  = Tipsy(filename)
    header = tipsy.Header()
    gas    = tipsy.Gas()
    dark   = tipsy.Dark()
    star   = tipsy.Star()
    print(header)
    print(gas)
    print(dark)
    print(star)

