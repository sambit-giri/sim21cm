import numpy as np
import tools21cm as t2c
from astropy import units as u
from scipy.integrate import simps

class InsideOut:
    def __init__(self, sim_params=None, file_type='npy', out_dir='./'):
        self.Lbox   = t2c.conv.LB if sim_params is None else sim_params["Box_Len"] 
        #self.nGrid  = nGrid if sim_params is None else sim_params["nGrid"]
        self.dens_files = {}
        self.file_type  = file_type
        self.out_dir    = out_dir

    def define_density_file(self, dens_files, dens_file_reader=None, zs=None):
        if zs is None: assert type(dens_files) == dict
        #else: assert len(zs)==len(dens_files)
        self.dens_file_reader = dens_file_reader
        if type(dens_files) == dict: 
            self.dens_files = dens_files
        else: 
            for i,z0 in enumerate(zs): self.dens_files['%.3f'%z0] = dens_files[i]
        self.dens_z = np.array([kk for kk in self.dens_files.keys()]).astype(float)

    def define_eor_hist(self, zs=None, xs=None, fn_zs=None, calc_fn=False):
        if calc_fn or fn_zs is None:
            from scipy.interpolate import interp1d
            self.fn_zs = interp1d(zs, xs, kind='linear', fill_value='extrapolate')
        self.sim_z = zs
        self.sim_x = self.fn_zs(zs) if xs is None else xs

    def read_density(self, dens_file, contrast=False):
        if self.dens_file_reader is not None: dens = self.dens_file_reader(dens_file)
        else:
            t2c.set_sim_constants(int(self.Lbox*0.7))
            dens = t2c.DensityFile(dens_file).raw_density#cgs_density*u.g/u.cm**3
        self.nGrid = dens.shape[0]
        if contrast: dens = dens/dens.mean() - 1.
        return dens

    def write_file(self, filename, data, file_type='npy'):
        ## More file types to be added later.
        #print(file_type)
        assert file_type in ['npy', 'fits']
        filename = self.out_dir+filename+'.'+file_type
        if file_type is 'npy': np.save(filename, data)
        elif file_type is 'npy': 
            from astropy.io import fits
            fits.writeto(filename, data)

    def xi_mass_average(self, dens, xi):
        dens_flatn = dens.flatten()
        if np.any(dens<0): dens_flatn - dens.min() + 1
        arg_ascend = np.argsort(dens_flatn)
        thres = dens_flatn.sum()*(1-xi)
        out = np.ones_like(dens_flatn)
        mass  = 0
        print('Running a brute force method for mass averaged ionization.')
        for i in arg_ascend:
            mass = mass+dens_flatn[i]
            if mass>thres: break
            out[i] = 0
        return out.reshape(dens.shape)

    def xi_volume_average(self, dens, xi):
        dens_flatn = dens.flatten() 
        arg_ascend = np.argsort(dens_flatn)
        out = np.zeros_like(dens_flatn)
        out[arg_ascend>=xi*dens_flatn.size] = 1
        return out.reshape(dens.shape)

    def run_sim(self, zs=None, volume_averaged=False, mass_averaged=True):
        if zs is not None: 
            self.sim_z = np.array([zs]) if type(zs) in [float, int] else zs
            self.sim_x = self.fn_zs(xs)
        for z0,x0 in zip(self.sim_z,self.sim_x):
            print('Simalating at z=%.3f for xi=%.2f.'%(z0,x0))
            dens = self.read_density(self.dens_files['%.3f'%z0])
            if volume_averaged:
                print('Volume averaged.')
                data = self.xi_volume_average(dens, x0)
                filename = 'xhii_z%.3f_xv%.3f'%(z0,x0)
                self.write_file(filename, data, file_type=self.file_type)
            if mass_averaged:
                print('Mass averaged.')
                data = self.xi_mass_average(dens, x0)
                filename = 'xhii_z%.3f_xm%.3f'%(z0,x0)
                self.write_file(filename, data, file_type=self.file_type)
        print('Simulations saved in')
        print(self.out_dir)

    def run_instant(self, dens, x0, volume_averaged=False, mass_averaged=True, write=False):
        from time import time
        print('Creating ionization field with xi=%.2f.'%(x0))
        stamp = time()
        if volume_averaged:
            xhii_v = self.xi_volume_average(dens, x0)
            if write:
                filename1 = 'xhii_xv%.3f_%d'%(x0,stamp)
                self.write_file(filename1, xhii_v, file_type=self.file_type)
                print('The volume averaged xhii data saved in')
                print(self.out_dir+filename1)
        if mass_averaged:
            xhii_m = self.xi_mass_average(dens, x0)
            if write:
                filename2 = 'xhii_z%.3f_xm%.3f'%(x0,stamp)
                self.write_file(filename2, xhii_m, file_type=self.file_type)
                print('The mass averaged xhii data saved in')
                print(self.out_dir+filename2)
        return {'v': xhii_v, 'm': xhii_m}
        

class OutsideIn:
    def __init__(self, sim_params=None, file_type='npy', out_dir='./'):
        self.Lbox   = t2c.conv.LB if sim_params is None else sim_params["Box_Len"] 
        #self.nGrid  = nGrid if sim_params is None else sim_params["nGrid"]
        self.dens_files = {}
        self.file_type  = file_type
        self.out_dir    = out_dir

    def define_density_file(self, dens_files, dens_file_reader=None, zs=None):
        if zs is None: assert type(dens_files) == dict
        #else: assert len(zs)==len(dens_files)
        self.dens_file_reader = dens_file_reader
        if type(dens_files) == dict: 
            self.dens_files = dens_files
        else: 
            for i,z0 in enumerate(zs): self.dens_files['%.3f'%z0] = dens_files[i]
        self.dens_z = np.array([kk for kk in self.dens_files.keys()]).astype(float)

    def define_eor_hist(self, zs=None, xs=None, fn_zs=None, calc_fn=False):
        if calc_fn or fn_zs is None:
            from scipy.interpolate import interp1d
            self.fn_zs = interp1d(zs, xs, kind='linear', fill_value='extrapolate')
        self.sim_z = zs
        self.sim_x = self.fn_zs(zs) if xs is None else xs

    def read_density(self, dens_file, contrast=False):
        if self.dens_file_reader is not None: dens = self.dens_file_reader(dens_file)
        else:
            t2c.set_sim_constants(int(self.Lbox*0.7))
            dens = t2c.DensityFile(dens_file).raw_density#cgs_density*u.g/u.cm**3
        self.nGrid = dens.shape[0]
        if contrast: dens = dens/dens.mean() - 1.
        return dens

    def write_file(self, filename, data, file_type='npy'):
        ## More file types to be added later.
        #print(file_type)
        assert file_type in ['npy', 'fits']
        filename = self.out_dir+filename+'.'+file_type
        if file_type is 'npy': np.save(filename, data)
        elif file_type is 'npy': 
            from astropy.io import fits
            fits.writeto(filename, data)

    def xi_mass_average(self, dens, xi):
        dens_flatn = dens.flatten()
        if np.any(dens<0): dens_flatn - dens.min() + 1
        arg_ascend = np.argsort(-dens_flatn)
        thres = dens_flatn.sum()*(1-xi)
        out = np.ones_like(dens_flatn)
        mass  = 0
        print('Running a brute force method for mass averaged ionization.')
        for i in arg_ascend:
            mass = mass+dens_flatn[i]
            if mass>thres: break
            out[i] = 0
        return out.reshape(dens.shape)

    def xi_volume_average(self, dens, xi):
        dens_flatn = dens.flatten() 
        arg_ascend = np.argsort(-dens_flatn)
        out = np.zeros_like(dens_flatn)
        out[arg_ascend>=xi*dens_flatn.size] = 1
        return out.reshape(dens.shape)

    def run_sim(self, zs=None, volume_averaged=False, mass_averaged=True):
        if zs is not None: 
            self.sim_z = np.array([zs]) if type(zs) in [float, int] else zs
            self.sim_x = self.fn_zs(xs)
        for z0,x0 in zip(self.sim_z,self.sim_x):
            print('Simalating at z=%.3f for xi=%.2f.'%(z0,x0))
            dens = self.read_density(self.dens_files['%.3f'%z0])
            if volume_averaged:
                print('Volume averaged.')
                data = self.xi_volume_average(dens, x0)
                filename = 'xhii_z%.3f_xv%.3f'%(z0,x0)
                self.write_file(filename, data, file_type=self.file_type)
            if mass_averaged:
                print('Mass averaged.')
                data = self.xi_mass_average(dens, x0)
                filename = 'xhii_z%.3f_xm%.3f'%(z0,x0)
                self.write_file(filename, data, file_type=self.file_type)
        print('Simulations saved in')
        print(self.out_dir)

    def run_instant(self, dens, x0, volume_averaged=False, mass_averaged=True, write=False):
        from time import time
        print('Creating ionization field with xi=%.2f.'%(x0))
        stamp = time()
        if volume_averaged:
            xhii_v = self.xi_volume_average(dens, x0)
            if write:
                filename1 = 'xhii_xv%.3f_%d'%(x0,stamp)
                self.write_file(filename1, xhii_v, file_type=self.file_type)
                print('The volume averaged xhii data saved in')
                print(self.out_dir+filename1)
        else: xhii_v = None
        if mass_averaged:
            xhii_m = self.xi_mass_average(dens, x0)
            if write:
                filename2 = 'xhii_z%.3f_xm%.3f'%(x0,stamp)
                self.write_file(filename2, xhii_m, file_type=self.file_type)
                print('The mass averaged xhii data saved in')
                print(self.out_dir+filename2)
        else: xhii_m = None
        return {'v': xhii_v, 'm': xhii_m}
