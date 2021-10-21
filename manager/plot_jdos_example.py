# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:29:40 2021

@author: NSing
"""
import json
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import DosPlotter
from pymatgen.electronic_structure import dos
from pymatgen.electronic_structure.core import Spin

from dos_helper import set_rc_params, get_plot


# parameters
dos_to_plot = [('Total','up'), ('Total','down'), ('Fe_2', 'd', 'up'), ('Fe_2', 'd', 'down'),
               ('Mo_12', 'd', 'up'), ('Mo_12', 'd', 'down')]
smear = 0.8
file = 'jpdos.json'
colors = ['grey','grey', 'r', 'r', 'b', 'b']
elim = [-4, 2]
alpha = [0.4,0.4, 0.7, 0.7, 0.5, 0.5]


if __name__ == '__main__':
    # setup plotting and get jdos dict
    set_rc_params()
    with open(file,'r') as f:
        jdos = json.load(f)
    
    # setup plotted and add dos objects 
    plotter = DosPlotter(sigma = smear)  
    names = []
    for pdos in dos_to_plot:
        spin = pdos[-1]
        x = jdos['up']['Total']
        xd = jdos['down']['Total']
        y = jdos['up']['Energy']
        
        if pdos[0] == 'Total':
            dens = jdos[spin]['Total']
            name = 'Total '+spin
        else:
            dens = jdos[spin][pdos[0]][pdos[1]] # Atom_# + orbital
            name = pdos[0].split('_')[0]+'('+pdos[0].split('_')[1]+') '+pdos[1] +' '+spin
            
        pmg_jdos = dos.Dos(0.0, jdos[spin]['Energy'], # centered @ 0 (Efermi))
                           {Spin.up if spin == 'up' else Spin.down: dens})
        plotter.add_dos(name, pmg_jdos)
        names.append(name)
    
    # reorder colors and alpha lists to match dos keys
    old_names = names
    names, colors = zip(*sorted(zip(names, colors)))
    names = old_names
    names, alpha = zip(*sorted(zip(names, alpha)))
    alpha = list(alpha)
    
    plot = get_plot(plotter, energy_lim=elim, colors=colors, alpha = alpha,
                    normalize_density=False) 
    plt.show()
    