# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:34:17 2021

@author: NSing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcdefaults()
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.plotting import pretty_plot

def set_rc_params():
    """
    Args:
        
    Returns:
        dictionary of settings for mpl.rcParams
    """
    params = {'axes.linewidth' : 1.5,'axes.unicode_minus' : False,
              'figure.dpi' : 100,
              'font.size' : 16,'font.family': 'sans-serif','font.sans-serif': 'Verdana',
              'legend.frameon' : False,'legend.handletextpad' : 0.2,
              'legend.handlelength' : 0.6,'legend.fontsize' : 16,
              'legend.columnspacing': 0.8,
              'mathtext.default' : 'regular','savefig.bbox' : 'tight',
              'xtick.labelsize' : 16,'ytick.labelsize' : 16,
              'xtick.major.size' : 6,'ytick.major.size' : 6,
              'xtick.major.width' : 1.5,'ytick.major.width' : 1.5,
              'xtick.top' : False,'xtick.bottom' : True,'ytick.right' : True,'ytick.left' : True,
              'xtick.direction': 'out','ytick.direction': 'out','axes.edgecolor' : 'black'}
    for p in params:
        mpl.rcParams[p] = params[p]
    return params

def get_plot(plotter, energy_lim=[-5, 5], density_lim=None, flip_axes = True, colors = None,
             normalize_density = True, fill = True, alpha = 1):
    """
    Taken from pymatgen.electronic_structure.plotter, DosPlotter.get_plot()
    Needed to rewrite to allow for axes flipping
    """
    if colors is None:
        import palettable
        ncolors = max(3, len(plotter._doses))
        ncolors = min(9, ncolors)
        # pylint: disable=E1101
        colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    else:
        ncolors = len(colors)
        
    y = None
    max_density = 0
    min_density = 0
    alldensities = []
    allenergies = []
    plot = pretty_plot(8, 12)

    for key, idos in plotter._doses.items():
        energies = idos["energies"]
        densities = idos["densities"]
        allenergies.append(energies)
        alldensities.append(densities)

    keys = list(plotter._doses.keys())
    keys.reverse()
    alldensities.reverse()
    allenergies.reverse()
    allpts = []
    for i, key in enumerate(keys):
        x = []
        y = []
        for spin in [Spin.up, Spin.down]:
            if spin in alldensities[i]:
                densities = list(int(spin) * alldensities[i][spin])
                energies = list(allenergies[i])
                if spin == Spin.down:
                    energies.reverse()
                    densities.reverse()
                x.extend(energies)
                y.extend(densities)
        allpts.extend(list(zip(x, y)))
        
        maxy = max([yi for ii,yi in enumerate(y) if x[ii] <= max(energy_lim) and x[ii] >= min(energy_lim)])
        miny = min([yi for ii,yi in enumerate(y) if x[ii] <= max(energy_lim) and x[ii] >= min(energy_lim)])
        if normalize_density:
            y = [yi / maxy for yi in y]
            density_lim = [0, 1.1]
        if maxy > max_density:
            max_density = maxy
        if miny < min_density:
            min_density = miny
            
        if flip_axes:
            tmp = x
            x = y
            y = tmp
        
        if fill and flip_axes:
            plt.fill_betweenx(y, np.zeros_like(y), x, color=colors[i % ncolors], label=str(key),
                              alpha = alpha[i % ncolors] if type(alpha) == list else alpha)
        elif fill:
            plot.fill_between(x, np.zeros_like(x), y, color=colors[i % ncolors], label=str(key),
                              alpha = alpha[i % ncolors] if type(alpha) == list else alpha)
        else:
            plot.plot(x, y, color=colors[i % ncolors], label=str(key), linewidth=2,
                      alpha = alpha[i % ncolors] if type(alpha) == list else alpha)
        
        if not plotter.zero_at_efermi:
            ylim = plot.ylim()
            plot.plot([plotter._doses[key]["efermi"], plotter._doses[key]["efermi"]],ylim,
                      color=colors[i % ncolors],linestyle="--",linewidth=2,)

    if not flip_axes:
        if energy_lim:
            plot.xlim(energy_lim)
        if density_lim:
            plot.ylim(density_lim)
        else:
            xlim = plot.xlim()
            relevanty = [p[1] for p in allpts if xlim[0] < p[0] < xlim[1]]
            plot.ylim((min(relevanty), max(relevanty)))
    else:
        if density_lim:
            plot.xlim(density_lim)
        else:
            plot.xlim((1.1 * min_density, max_density * 1.1))
        if energy_lim:
            plot.ylim(energy_lim)

    if plotter.zero_at_efermi:
        if not flip_axes:
            ylim = plot.ylim()
            plot.plot([0, 0], ylim, "k--", linewidth=2)
        else:
            plot.plot(plot.xlim(), [0, 0], "k--", linewidth=2)

    plot.xlabel("Energies (eV)")
    plot.ylabel("Density of states")

    plot.axhline(y=0, color="k", linestyle="--", linewidth=2)
    plot.legend()
    leg = plot.gca().get_legend()
    ltext = leg.get_texts()  
    plot.setp(ltext, fontsize=30)
    plot.tight_layout()
    return plot