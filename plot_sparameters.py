import pathlib
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

touchstone_path = 'example.s2p'
output_dir = pathlib.Path('plots')
output_dir.mkdir(exist_ok=True)

ntw = rf.Network(touchstone_path)
freq = ntw.frequency.f/1e9
return_loss_db = -20*np.log10(np.abs(ntw.s[:,0,0]))
insertion_loss_db = -20*np.log10(np.abs(ntw.s[:,1,0]))
impedance = ntw.z[:,0,0]

plt.figure()
plt.plot(freq, return_loss_db)
plt.title('Return Loss (S11)')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.savefig(output_dir/'return_loss.png')

plt.figure()
plt.plot(freq, insertion_loss_db)
plt.title('Insertion Loss (S21)')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.savefig(output_dir/'insertion_loss.png')

plt.figure()
plt.plot(freq, impedance.real, label='Real')
plt.plot(freq, impedance.imag, label='Imag')
plt.title('Input Impedance (Z11)')
plt.xlabel('Frequency [GHz]')
plt.ylabel('Impedance [Ohms]')
plt.legend()
plt.grid(True)
plt.savefig(output_dir/'input_impedance.png')

import nbformat
from nbconvert import ScriptExporter

nb = nbformat.read('plot_sparameters.ipynb', as_version=4)
script, _ = ScriptExporter().from_notebook_node(nb)
with open('plot_sparameters.py', 'w') as f:
    f.write(script)
print('Exported notebook to plot_sparameters.py')
