# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:32:44 2021

@author: NSing
"""
import os
import json
import numpy as np
from pymatgen.core.structure import Structure


hartree_to_ev = 27.2114


class helper():
    
    def read_inputs(self, folder):
        # returns a dictionary of inputs
        with open(os.path.join(folder, 'inputs'), 'r') as f:
            input_text = f.read()
        tags = {}
        for line in input_text.split('\n'):
            if line == '':
                continue
            if line[0] == '#':
                continue
            words = line.split(' ')
            if words[0] in tags:
                if type(tags[words[0]]) != list:
                    tags[words[0]] = [tags[words[0]]]
                tags[words[0]].append(' '.join(words[1:]))
            else:
                tags[words[0]] = ' '.join(words[1:])
        return tags
    
    def write_inputs(self, inputs_dic, folder):
        # expects a dictionary of inputs with either strings or lists of strings as vals
        text = ''
        for k,v in inputs_dic.items():
            if type(v) == list:
                for vi in v:
                    text += k + ' ' + vi + '\n'
            else:
                text += k + ' ' + v + '\n'
        with open(os.path.join(folder, 'inputs'), 'w') as f:
            f.write(text)

    def read_optlog(self, folder, file = 'opt.log', verbose = True):
        try:
            with open(os.path.join(folder, file), 'r') as f:
                opt_text = f.read()
        except:
            print('ERROR: Cannot read opt.log file')
            return False
        if len(opt_text) == 0:
            # no data yet
            return False #[{'energy': 'None', 'force': 'None', 'step': 0}]
        steps = []
        not_warned = True
        ionic_step = 0
        for i, line in enumerate(opt_text.split('\n')):
            if ' Step ' in line or '*Force-consistent' in line or 'Convergence Step' in line:
                continue
            if line == '':
                continue
            var = line.split()
            step_dic = {'opt_method': var[0][:-1],
                        'energy': float(var[3][:-1]) / hartree_to_ev,
                        'force': float(var[4]),
                        'step': ionic_step}
            if float(var[4]) > 10 and not_warned and verbose:
                print('**** WARNING: High forces present in convergence. Check CONTCAR. ****')
                not_warned = False
            ionic_step += 1
            steps.append(step_dic)
        return steps
    
    def read_Ecomponents(self, folder):
        try:
            with open(os.path.join(folder, 'Ecomponents'), 'r') as f:
                ecomp_text = f.read()
        except:
            print('ERROR: Cannot read Ecomponents file')
            return False
        ecomp = {}
        for line in ecomp_text.split('\n'):
            if '----' in line or len(line) == 0:
                continue
            ecomp[line.split()[0]] = float(line.split()[-1])
        return ecomp
    
    def read_outfile(self, folder, contcar = 'None'):
        try:
            with open(os.path.join(folder, 'out'), 'r', errors='ignore') as f:
                out_text = f.read()
        except:
            print('Error reading out file.')
            return False
        site_data = {}
        record_forces, record_ions = False, False
        el_counter = 0
        net_oxidation = 0
        net_mag = 0
        initial_electrons = None
        final_electrons = None
        
        if contcar != 'None':
            ct_els = list(set([site['label'] for site in contcar['sites']]))
            ct_counter = {el: 0 for el in ct_els}
        
        for li,line in enumerate(out_text.split('\n')):
            if 'nElectrons' in line and initial_electrons is None:
                initial_electrons = float(line.split()[1])
            if 'FillingsUpdate' in line:
                final_electrons = float(line.split()[4])
            
            if 'Ionic positions in cartesian coordinates' in line:
                record_ions = True
                # reset net oxidation states and magnetization for new group
                net_oxidation = 0
                net_mag = 0
                continue
            if record_ions:
                if 'ion' in line:
                    atom, xpos, ypos, zpos = line.split()[1:5]
                    site_data[str(el_counter)] = {'atom': atom}
                    
                    if contcar != 'None':
                        if el_counter >= len(contcar['sites']):
                            print('Error: different number of sites in contcar and out file.')
                            return False
                        try:
                            el_index = [i for i, site in enumerate(contcar['sites']) if atom == site['label']]
                            ct_index = el_index[ct_counter[atom]]
                            ct_site = contcar['sites'][ct_index]
                            ct_counter[atom] += 1
                        except:
                            print('Error reading out file sites.')
                            return False
                        site_data[str(el_counter)]['positions'] = ct_site['xyz'] 
                        site_data[str(el_counter)]['contcar_index'] = ct_index
                        
                        if 'properties' in ct_site and 'selective_dynamics' in ct_site['properties']:
                            site_data[str(el_counter)]['selective_dynamics'] = (
                                        ct_site['properties']['selective_dynamics'])
                    el_counter += 1
                else: #if line == '' or 'ion' not in line:
                    record_ions = False
                    el_counter = 0
                    if contcar != 'None':
                        ct_counter = {el: 0 for el in ct_els}
            
            if 'Forces in Cartesian coordinates' in line:
                record_forces = True
                continue
            if record_forces:
                if 'force' in line:
                    atom, xfor, yfor, zfor = line.split()[1:5]
                    if atom != site_data[str(el_counter)]['atom']:
                        print('Error reading out file.')
                        return False
                    site_data[str(el_counter)]['forces'] =  [float(xfor), float(yfor), float(zfor)]
                    el_counter += 1
                else: #if line == '' or 'force' not in line:
                    record_forces = False
                    el_counter = 0
            
            if 'magnetic-moments' in line:
                el = line.split()[2]
                mags = line.split()[3:]
                mag_counter = 0
                for site,sv in site_data.items():
                    if sv['atom'] == el:
                        try:
                            site_data[site]['mag_mom'] = float(mags[mag_counter])
                            net_mag += float(mags[mag_counter])
                            mag_counter += 1
                        except:
                            print('Error reading magnetic moments')
                            return False
            if 'oxidation-state' in line:
                el = line.split()[2]
                oxis = line.split()[3:]
                oxi_counter = 0
                for site,sv in site_data.items():
                    if sv['atom'] == el:
                        try:
                            site_data[site]['oxi_state'] = float(oxis[oxi_counter])
                            net_oxidation += float(oxis[oxi_counter])
                            oxi_counter += 1
                        except:
                            print('Error reading oxidation states')
                            return False
        return site_data, net_oxidation, net_mag, final_electrons, initial_electrons

    def read_contcar(self, folder):
        st = Structure.from_file(os.path.join(folder, 'CONTCAR'))
        return st.as_dict()
            
    def get_force(self, steps):
        return steps[-1]['force']
    
    def check_convergence(self, inputs, steps):
        force = self.get_force(steps)
        fmax = float(inputs['fmax'])
        if force <= fmax:
            return True
        return False
    
    def get_bias(self, bias_str):
        if bias_str in ['No_bias']:
            return 'No_bias'
        return float(bias_str[:-1])
    
    def get_bias_str(self, bias):
        if bias == 'No_bias':
            return 'No_bias'
        return '%.2f'%bias + 'V'
    
    def read_data(self, folder):
        # currently reads inputs, opt_log for energies, and CONTCAR. Also checks convergence based on forces
        # reads out file for oxidation states and magentic moments
        inputs = self.read_inputs(folder)
        opt_steps = self.read_optlog(folder)
        ecomp = self.read_Ecomponents(folder)
        if opt_steps == False or ecomp == False:
            return {'opt': 'None', 'inputs': inputs, 'converged': False,
                    'final_energy': 'None', 'contcar': 'None', 'current_force': 'None'}
        # check if calc has high forces
        if opt_steps[-1]['force'] > 10:
            print('**** WARNING: High forces (> 10) in current step! May be divergent. ****')
        # check for convergence
        conv = self.check_convergence(inputs, opt_steps)
        contcar = 'None'
        if 'CONTCAR' in os.listdir(folder):
            contcar = self.read_contcar(folder)
        out_sites = self.read_outfile(folder, contcar)
        if out_sites == False:
            sites = {}
            net_oxi = 'None'
            net_mag = 'None'
            nfinal = 'None'
        else:
            sites = out_sites[0]
            net_oxi = out_sites[1]
            net_mag = out_sites[2]
            nfinal = out_sites[3]
        return {'opt': opt_steps, 'current_force': opt_steps[-1]['force'],
                'inputs': inputs, 'Ecomponents': ecomp, 
                'Ecomp_energy': ecomp['F'] if 'F' in ecomp else (ecomp['G'] if 'G' in ecomp else 'None'),
                'converged': conv, 'contcar': contcar, 'nfinal': nfinal,
                'final_energy': 'None' if not conv else opt_steps[-1]['energy'],
                'site_data': sites, 'net_oxidation': net_oxi, 'net_magmom': net_mag}
        
    def get_neb_data(self, folder, bias):
        # reads neb folder and returns data as a dictionary
        inputs = self.read_inputs(folder)
        opt_steps = self.read_optlog(folder, 'neb.log')
        if opt_steps == False:
            return {'opt': 'None', 'inputs': inputs, 'converged': False,
                    'final_energy': 'None', 'images': {}}
        # check if calc has high forces
        if opt_steps[-1]['force'] > 10:
            print('**** WARNING: High forces (> 10) in current step! May be divergent. ****')
        # check for convergence
        conv = self.check_convergence(inputs, opt_steps)
        images = {}
        for root, folders, files in os.walk(folder):
            # look at subfolders
            if 'CONTCAR' not in files or 'opt.log' not in files:
                continue
            image_num = root.split(os.sep)[-1]
            contcar = self.read_contcar(root)
            ecomp = self.read_Ecomponents(root)
            energy = ecomp['F'] if 'F' in ecomp else (ecomp['G'] if 'G' in ecomp else 'None')
            out_sites = self.read_outfile(root, contcar)
            if out_sites == False:
                sites = {}
                net_oxi = 'None'
                net_mag = 'None'
                nfinal = 'None'
            else:
                sites = out_sites[0]
                net_oxi = out_sites[1]
                net_mag = out_sites[2]
                nfinal = out_sites[3]
            images[image_num] = {'contcar': contcar, 'energy': energy, 'Ecomponents': ecomp, 'nfinal': nfinal,
                                 'site_data': sites, 'net_oxidation': net_oxi, 'net_magmom': net_mag}
        return {'opt': opt_steps,
                'inputs': inputs,
                'converged': conv,
                'final_energy': 'None' if not conv else opt_steps[-1]['energy'],
                'images': images, 'path_energy': {k:v['energy'] for k,v in images.items()}}
        
    def get_sites(self, line):
        sites = line.split('[')[-1].replace(']','').split(', ')
        for i,s in enumerate(sites):
            # sites can be a type of site (ie hollow), 'all', 'center', int, element symbol, or (x,y,z) tuple
            try:
                sites[i] = int(s)
                continue
            except:
                pass
            if '(' in s and ')' in s:
                try:
                    xyz = s.replace(')','').replace('(','').split(',')
                    xyz = tuple([float(x) for x in xyz])
                    sites[i] = xyz
                except:
                    pass
            # strings are already treated as themselves in initial split
#            if s in ['All', 'Hollow', 'Ontop', 'Bridge', 'all', 'hollow', 'ontop', 'bridge', 'center']:
#                sites[i] = s
#                continue
        return sites

    def check_surface(self, file, dist = 8):
        # check that surface seems reasonable for manager to handle
        st = Structure.from_file(file)
        # check that z-direction is long distance
        if not (st.lattice.c > st.lattice.a and st.lattice.c > st.lattice.b):
            print('Bad Surface: z-direction is not longest side, skipping: '+file)
            return False
        # check that 10A (dist) of space is available in the vacuum above surface
        coords = st.cart_coords
        max_lens = [np.max(coords[:,0]), np.max(coords[:,1]), np.max(coords[:,2])]
#        if max_lens[2] <= max_lens[0] or max_lens[2] <= max_lens[1]:
#            print('Bad Surface: surface is wider than it is tall, skipping: '+file)
#            return False
        if st.lattice.c - max_lens[2] < dist:
            print('Bad Surface: Needs at least '+str(dist)+'A of vacuum space above surface, skipping: '+file)
            return False
        return True

    def read_bias(self, line):
        bias_str = line.split('[')[-1].replace(']','').split(', ')
        return ['No_bias' if x in ['None','none','No_bias'] else float(x) for x in bias_str]
    
    def update_run_new(self, run_new):
        for root in run_new:
            inputs = self.read_inputs(root)
            inputs['restart'] = 'False'
            self.write_inputs(inputs, root)

    def analyze_data(self, data, ref_mols):
        '''
        Main function for analyzing converged data from scan_calcs function
        Functions:
            1) Creates analyzed.json file containing:
                - binding energies mapped over biases for each system
                - NEB barriers mapped over biases for each NEB system
        '''
        print('Data analysis not yet available. Please contact Nick to add.')
        return None #data_analysis(data)
    
    @property
    def reference_molecules():
        refs = {'H': {'refs': ['H2'], 'coeffs': [0.5]},
                'H2': {'refs': ['H2'], 'coeffs': [1]},
                'H2O': {'refs': ['H2O'], 'coeffs': [1]},
                'H3O':{'refs': ['H2O', 'H'], 'coeffs': [1, 1]}, 
                'H_H2O':{'refs': ['H2O', 'H'], 'coeffs': [1, 1]},
                'H_H3O':{'refs': ['H2O', 'H2'], 'coeffs': [1, 1]},
                'H2_H2O':{'refs': ['H2O', 'H2'], 'coeffs': [1, 1]},
                'O': {'refs': ['H2O','H2'], 'coeffs': [1,-1]},
                'CO2': {'refs': ['CO2'], 'coeffs': [1]},
                'CO': {'refs': ['CO2','O'], 'coeffs': [1,-1]},
                'CHO': {'refs': ['CO','H'], 'coeffs': [1,1]},
                'COH': {'refs': ['CHO'], 'coeffs': [1]},
                'OCH': {'refs': ['CHO'], 'coeffs': [1]},
                'N': {'refs': ['N2'], 'coeffs': [0.5]},
                'N2': {'refs': ['N2'], 'coeffs': [1]},
                'N2H': {'refs': ['N2','H'], 'coeffs': [1,1]},
                'NNH2': {'refs': ['N2','H'], 'coeffs': [1,2]},
                'NNH3': {'refs': ['N2','H'], 'coeffs': [1,3]},
                'NH': {'refs': ['N2','H'], 'coeffs': [0.5,1]},
                'NH2': {'refs': ['N2','H'], 'coeffs': [0.5,2]},
                'NH3': {'refs': ['N2','H'], 'coeffs': [0.5,3]},
                'OC': {'refs': ['CO'], 'coeffs': [1,]},
                'OCO': {'refs': ['CO2'], 'coeffs': [1]},
                'OH': {'refs': ['O','H'], 'coeffs': [1,1]},
                'OOH': {'refs': ['O','H'], 'coeffs': [2,1]},
                'S2': {'refs': ['S8'], 'coeffs': [0.25]},
                'S4': {'refs': ['S8'], 'coeffs': [0.5]},
                'S6': {'refs': ['S8'], 'coeffs': [0.75]},
                'S8': {'refs': ['S8'], 'coeffs': [1]},
                }
        
        if os.path.exists('./molecules/refs.json'):
            with open('./molecules/refs.json', 'r') as f:
                refs2 = json.load(f)
            
            for k,v in refs2.items():
                refs[k] = v
        
        return refs
    
    
