#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:27:07 2021

@author: NSing
"""

import json
import numpy as np
hartree_to_ev = 27.2114

fileup = 'dosUp'
filedown = 'dosDn'

remove_small_vals = False
zero_fermi = True

with open(fileup, 'r', errors='ignore') as f:
    txt_up = f.read()
with open(filedown, 'r', errors='ignore') as f:
    txt_down = f.read()
with open('out', 'r') as f:
    txt_out = f.read()

def read_EF(txt):
    target = 'mu: '
    ef = None
    for line in txt.split('\n'):
        if target in line:
            ef = float(line.split()[2])
    assert ef is not None, 'ERROR: no mu/EF found.'
    return ef * hartree_to_ev

efermi = read_EF(txt_out)
dos = {'up': {}, 'down': {}, 'Efermi': efermi, 'zeroed_fermi': zero_fermi}
edic = {'up': {}, 'down': {}}
for ispin,spin in enumerate(['up','down']):
    # target format: {'up': {Fe_1: {'s': [], 'd': []}, 'Energy': [], 'Total': []}}
    txt = [txt_up, txt_down][ispin]
    for i,line in enumerate(txt.split('\n')):
        if line in ['', ' ']:
            continue
        if i == 0:
            # add keys
            keys = [k for k in line.split('"') if k not in ['', ' ', '\t']]
            continue
        
        values = [float(v) for v in line.split()]
        if len(values) != len(keys):
            print('ERROR: Failed to read line: '+str(i))
            print(line)
            continue
        
        # get energy in eV (centered @ efermi)
        energy = values[keys.index('Energy')] * hartree_to_ev - (efermi if zero_fermi else 0.0)
        estr = '%.3f'%energy
        if estr in edic[spin]:
            edic[spin][estr].append(values)
        else:
            edic[spin][estr] = [values]
        
    for estr, value_list in edic[spin].items():
        for ii, v in enumerate(value_list[0]):
            key = keys[ii]
            
            # add energy to dic
            if key in ['Energy']:
                en = float(estr)
                if key not in dos[spin]:
                    dos[spin][key] = [en]
                else:
                    dos[spin][key].append(en)
                continue
            
            # get dos average at energy
            dos_eav = np.average([vs[ii] for vs in value_list])
            if remove_small_vals and dos_eav < 0.01:
                dos_eav = 0.0
            
            # add total dos based on average value
            if key in ['Total']:
                if key not in dos[spin]:
                    dos[spin][key] = [dos_eav]
                else:
                    dos[spin][key].append(dos_eav)
                continue
            
            # key is an orbital string
            el = key.split()[-2]
            n = key.split()[-1].replace('#','')
            orbital = key.split()[0]
            
            el_key = el+'_'+n
            if el_key not in dos[spin]:
                dos[spin][el_key] = {}
            if orbital not in dos[spin][el_key]:
                dos[spin][el_key][orbital] = []
            dos[spin][el_key][orbital].append(dos_eav) 

with open('jpdos.json','w') as f:
    json.dump(dos, f)