#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:59:29 2021

@author: NSing
"""

import os
import argparse
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
from jdft_helper import helper 
h = helper()
opj = os.path.join


pseudoMap = {'SG15' : 'SG15/$ID_ONCV_PBE.upf', # readable format
             'GBRV' : 'GBRV/$ID_pbe.uspp', # not readable
             'GBRV_v1.5' : 'GBRV_v1.5/$ID_pbe_v1.uspp.F.UPF', # readable format
             'dojo': 'dojo/$ID.upf', # readable format
             } 

def set_elec_n_bands(root, file, psd, band_scaling, kpoint_density):
    
    try:
        st = Structure.from_file(opj(root, file))
    except:
        assert False, 'Error reading '+file+' file (check existence).'
    psdir = os.environ['JDFTx_pseudos']
    
    if psd != 'None':
        try:
            tags = h.read_inputs(root)
            psd = tags['pseudos'] if 'pseudos' in tags else 'None'
            print('Pseudopotential set to '+psd+' from inputs file')
        except:
            pass
    assert psd != 'None', 'No pseudopotential specified!'
        
    ps_key = opj(psdir, pseudoMap[psd])
    els = [s.species_string for s in st.sites]
    el_dic = {el: els.count(el) for el in els}
    # get electrons from psd files
    nelec = 0
    for el, count in el_dic.items():
        file = ps_key.replace('$ID', el.lower())
        with open(file, 'r') as f:
            ps_txt = f.read()
        zval = [line for line in ps_txt.split('\n') if 'Z valence' in line # GBRV
                or 'z_valence' in line][0] # SG15
        electrons = int(float(zval.split()[0])) if 'Z' in zval else (
                    int(float(zval.split()[1].replace('"',''))))
        nelec += electrons * count
    nbands_add = int(nelec / 2) + 10
    nbands_mult = int((nelec / 2) * band_scaling)
    
    
    print('Suggested Input Files Based On '+file+':')
    
    if nbands_add >= nbands_mult:
        print('elec-n-bands = '+str(nbands_add)+' (by nelec/2 + 10)')
    else:
        print('elec-n-bands = '+str(nbands_mult)+' (by nelec/2 * %.1f)'%(band_scaling))
    
    
#    kpoint_density = 1000
    kpts = Kpoints.automatic_density(st, kpoint_density).as_dict()
    kpt_str = ' '.join([str(k) for k in kpts['kpoints'][0]])
    print('kpoints = '+kpt_str+' (density = '+str(kpoint_density)+')')
    
    return max([nbands_add, nbands_mult])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pseudo', help='Pseudopotential to read (default to read inputs file)',
                        type=str, default='None')
    parser.add_argument('-f', '--file', help='File to read (default POSCAR)',
                        type=str, default='POSCAR')
    parser.add_argument('-s', '--scale', help='Scaling multiplier for bands, nelec/2 * s (default 1.2)',
                        type=float, default=1.2)
    parser.add_argument('-k', '--kptd', help='Kpoint grid density (default 1000)',
                        type=int, default=1000)
    parser.add_argument('-r', '--root', help='Root folder (default ./)',
                        type=str, default='./')
    
    args = parser.parse_args()
    
    nbands = set_elec_n_bands(args.root, args.file, args.pseudo, args.scale, args.kptd)
    
    