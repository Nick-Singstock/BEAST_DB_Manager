#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:32:44 2021

@author: NSing
"""
import os
import json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core import Lattice

opj = os.path.join
ope = os.path.exists
hartree_to_ev = 27.2114
ang_to_bohr = 1.88973

class helper():
    
    def read_inputs(self, folder, file = 'inputs'):
        # returns a dictionary of inputs
        with open(os.path.join(folder, file), 'r') as f:
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
        steps = {'1': []}
        step = 1
        not_warned = True
        ionic_step = 0
        for i, line in enumerate(opt_text.split('\n')):
            if 'Convergence Step' in line:
                step = int(line.split()[-1])
                if str(step) not in steps:
                    steps[str(step)] = []
                continue
            if ' Step ' in line or '*Force-consistent' in line:
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
            steps[str(step)].append(step_dic)
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
    
    def read_out_struct(self, folder, site_data = False):
        try:
            with open(opj(folder, 'out'), 'r', errors='ignore') as f:
                out_text = f.read()
        except:
            assert False, 'Error reading out file.'
                
        record_ions = False
        record_lattice = False
        lattice = np.zeros((3,3))
        no_atoms = True
        no_lattice = True
        if site_data:
            spdata = {'selective_dynamics': []}
        for line in out_text.split('\n'):
            # get lattice
            if '# Lattice vectors:' in line:
                record_lattice = True
                lattice_track = 0
                continue
            if record_lattice:
                if 'unit cell volume' in line:
                    record_lattice = False
                    lattice = lattice.T
                    continue
                if 'R =' in line:
                    continue
                no_lattice = False
                lattice[lattice_track] = np.array([float(x)/ang_to_bohr for x in line.split()[1:-1]])
                lattice_track += 1
            
            # get ions
            if 'Ionic positions in cartesian coordinates' in line:
                record_ions = True
                coords = []
                species = []
                continue
            if record_ions:
                if 'ion' not in line:
                    record_ions = False
                    continue
                species.append(line.split()[1])
                coords.append([float(x)/(ang_to_bohr) for i,x in enumerate(line.split()[2:5])])
                no_atoms = False
                if site_data:
                    spdata['selective_dynamics'] += int(line.split()[-1])
        
        if no_lattice or no_atoms:
            assert False, 'No lattice or no atoms. Out file structure not read correctly.'
        if site_data:
            return Structure(lattice, species, coords, coords_are_cartesian=True, site_properties=spdata)
        return Structure(lattice, species, coords, coords_are_cartesian=True)
                
        

    def read_contcar(self, folder):
        st = Structure.from_file(os.path.join(folder, 'CONTCAR'))
        return st.as_dict()
            
    def get_force(self, steps):
        return steps[-1]['force']
    
    def get_energies(self, steps):
        return [s['energy'] for s in steps]
    
    def check_convergence(self, folder, inputs, steps):
        current_step = list(steps.keys())[-1]
        fmax = 'None'
        econv = 'None'
        max_steps = 'None'
        if ope(opj(folder, 'convergence')):
            conv = self.read_convergence(folder)
            last_step = list(conv.keys())[-1]
            if current_step != last_step: # convergence is on last step
                return False
            if 'fmax' in conv[current_step]:
                fmax = float(conv[current_step]['fmax'])
            if 'econv' in conv[current_step]:
                econv = float(conv[current_step]['econv'])
            if 'max_steps' in conv[current_step]:
                max_steps = int(conv[current_step]['max_steps'])
        
        if fmax == 'None':
            fmax = float(inputs['fmax'])
        if econv == 'None':
            econv = float(inputs['econv']) if 'econv' in inputs else 'None'
        if max_steps == 'None':
            max_steps = int(inputs['max_steps'])
        
        # check if convergence is complete based on force or energy convergence criteria (may change per step)
        force = self.get_force(steps[current_step]) if len(steps[current_step]) > 0 else 1e6
        energies = self.get_energies(steps[current_step]) if len(steps[current_step]) > 0 else 1e6
        nsteps = len(steps[current_step])
        
        if force <= fmax:
            return True # force based convergence
        elif econv != 'None' and len(energies) > 2 and np.abs(energies[-1]-energies[-2]) < econv:
            return True # energy based convergence
        elif nsteps > 0 and max_steps <= 1:
            return True # convergence based on single point calc (no conv for hitting max_steps otherwise)
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
        current_step = list(opt_steps.keys())[-1]
        current_force = opt_steps[current_step][-1]['force'] if len(opt_steps[current_step]) > 0 else 'None'
        current_energy = opt_steps[current_step][-1]['energy'] if len(opt_steps[current_step]) > 0 else 'None'
        if current_force != 'None' and current_force > 10:
            print('**** WARNING: High forces (> 10) in current step! May be divergent. ****')
        # check for convergence
        conv = self.check_convergence(folder, inputs, opt_steps)
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
        return {'opt': opt_steps, 'current_force': current_force, 'current_step': current_step,
                'inputs': inputs, 'Ecomponents': ecomp, 'current_energy': current_energy,
                'Ecomp_energy': ecomp['F'] if 'F' in ecomp else (ecomp['G'] if 'G' in ecomp else 'None'),
                'converged': conv, 'contcar': contcar, 'nfinal': nfinal,
                'final_energy': 'None' if not conv else current_energy,
                'site_data': sites, 'net_oxidation': net_oxi, 'net_magmom': net_mag}
        
    def get_neb_data(self, folder, bias):
        # reads neb folder and returns data as a dictionary
        inputs = self.read_inputs(folder)
        opt_steps = self.read_optlog(folder, 'neb.log')
        if opt_steps == False:
            return {'opt': 'None', 'inputs': inputs, 'converged': False,
                    'final_energy': 'None', 'images': {}}
        # check if calc has high forces
        current_step = list(opt_steps.keys())[-1]
        current_force = opt_steps[current_step][-1]['force'] if len(opt_steps[current_step]) > 0 else 'None'
        current_energy = opt_steps[current_step][-1]['energy'] if len(opt_steps[current_step]) > 0 else 'None'
        if current_force != 'None' and current_force > 10:
            print('**** WARNING: High forces (> 10) in current step! May be divergent. ****')
        # check for convergence
        conv = self.check_convergence(folder, inputs, opt_steps)
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
        return {'opt': opt_steps, 'current_force': current_force, 'current_step': current_step,
                'inputs': inputs, 'current_energy': current_energy,
                'converged': conv,
                'final_energy': 'None' if not conv else current_energy,
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

    def check_surface(self, file, dist = 6):
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
    
#    @property
    def reference_molecules(self):
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
    
    def read_convergence(self, root):
        '''
        convergence example:
        step 1
        kpoints 1 1 1
        
        step 2
        kpoints 3 3 3
        '''
        with open(opj(root, 'convergence'),'r') as f:
            conv_txt = f.read()
        step = '1'
        add_step = False
        conv_dic = {}
        for line in conv_txt.split('\n'):
            if line == '' or line == ' ': 
                continue
#            print(line)
            if any(x in line for x in ['step ','Step ']):
                step = line.split()[1]
                if step == '0':
                    add_step = True
                if add_step:
                    step  = str(int(step)+1) # update so steps are always indexed to 1
                if step not in conv_dic:
                    conv_dic[step] = {}
            else:
                cmd, val = line.split()[0], ' '.join(line.split()[1:])
                if cmd in conv_dic[step]:
                    # multiple versions of same command (i.e. pdos)
                    if type(conv_dic[step][cmd]) == list:
                        conv_dic[step][cmd].append(val)
                    else:
                        conv_dic[step][cmd] = [conv_dic[step][cmd], val]
                else:
                    # single version of command
                    conv_dic[step][cmd] = val
        return conv_dic
    
    def write_convergence(self, root, conv_dic):
        txt = ''
        assert '1' in conv_dic, 'METAERROR: Value Step 1 not in convergence dictionary, cannot write file.'
        for step, step_dic in conv_dic.items():
            if step == '1':
                txt += 'step 1\n'
            else:
                txt += '\nstep '+step + '\n'
            
            for cmd, val in step_dic.items():
                if type(val) == list:
                    for v in val:
                        txt += cmd + ' ' + v + '\n'
                else:
                    txt += cmd + ' ' + val + '\n'
        
        # write convergence file
        with open(opj(root, 'convergence'), 'w') as f:
            f.write(txt)
        
        
        

    
    
