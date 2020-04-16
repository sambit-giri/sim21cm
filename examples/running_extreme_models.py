'''
Created on 16 Apr 2020
@author: Sambit Giri
Setup script
'''
import numpy as np
import matplotlib.pyplot as plt

from sim21cm import extreme_model
import tools21cm as t2c

t2c.set_sim_constants(500)

# Instantaneous simulation
### In this mode, we take one density field and ionize it based on a 
### given average ionization fraction

dens_file = '/disk/dawn-1/garrelt/Reionization/C2Ray_WMAP7/500Mpc/coarser_densities/nc300/8.064n_all.dat'
dens = t2c.DensityFile(dens_file).raw_density
xi   = [0.3, 0.7]

simoi = extreme_model.OutsideIn()
xhii0 = simoi.run_instant(dens, xi[0], volume_averaged=True, mass_averaged=True, write=False)
xhii1 = simoi.run_instant(dens, xi[1], volume_averaged=True, mass_averaged=True, write=False)

plt.figure()
plt.suptitle('Outside-In')
plt.subplot(221); plt.title('$x_\mathrm{v}$=%.2f'%xi[0])
plt.imshow(xhii0['v'][:,:,100], cmap='jet')
plt.subplot(222); plt.title('$x_\mathrm{m}$=%.2f'%xi[0])
plt.imshow(xhii0['m'][:,:,100], cmap='jet')
plt.subplot(223); plt.title('$x_\mathrm{v}$=%.2f'%xi[1])
plt.imshow(xhii1['v'][:,:,100], cmap='jet')
plt.subplot(224); plt.title('$x_\mathrm{m}$=%.2f'%xi[1])
plt.imshow(xhii1['m'][:,:,100], cmap='jet')

simio = extreme_model.InsideOut()
xhii2 = simio.run_instant(dens, xi[0], volume_averaged=True, mass_averaged=True, write=False)
xhii3 = simio.run_instant(dens, xi[1], volume_averaged=True, mass_averaged=True, write=False)

plt.figure()
plt.suptitle('Inside-Out')
plt.subplot(221); plt.title('$x_\mathrm{v}$=%.2f'%xi[0])
plt.imshow(xhii2['v'][:,:,100], cmap='jet')
plt.subplot(222); plt.title('$x_\mathrm{m}$=%.2f'%xi[0])
plt.imshow(xhii2['m'][:,:,100], cmap='jet')
plt.subplot(223); plt.title('$x_\mathrm{v}$=%.2f'%xi[1])
plt.imshow(xhii3['v'][:,:,100], cmap='jet')
plt.subplot(224); plt.title('$x_\mathrm{m}$=%.2f'%xi[1])
plt.imshow(xhii3['m'][:,:,100], cmap='jet')

plt.show()


# Full simulation over a history
### There are methods to do so in the InsideOut and OutsideIn classes.


