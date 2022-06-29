#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-throughput manager class for JDFTx calculations. Created alongside BEAST database. 

@author: Nick_Singstock
"""

#import warnings
#warnings.filterwarnings("ignore")   # skip pymatgen yaml warnings 

import os
import argparse
import subprocess
import json
from time import sleep
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints
import numpy as np
from adsorbate_helper import (save_structures, add_adsorbates, write_parallel, 
                              minimum_movement_strs, assign_selective_dynamics,
                              write_parallel_bundle)
from parallel_manager import sub_parallel
from jdft_helper import helper 
h = helper()
opj = os.path.join
ope = os.path.exists

# Eagle defaults
calc_folder = 'calcs/'
molecule_folder = 'molecules/'
results_folder = 'results/'
inputs_folder = 'inputs/'
backup_folder = 'backup/'

hartree_to_ev = 27.2114

class jdft_manager():
    '''
    Author: Nick Singstock
    Date: Sept 30th 2021
    
    Class jdft_manager allows for management of a large number of JDFTx surface calculations 
    in a properly setup folder using the manager_control.txt file. Please run the "-s True" 
    command in a new folder to setup a managed folder and obtain the important subdirectories
    and files. More information is available in the manager_control.txt file.
    
    Please email Nick with questions about proper use and to alert about bugs.
    '''
    
    def __init__(self):
        # initialize parameters
        self.cwd = os.getcwd()
        self.__get_user_inputs__()
        self.__get_run_cmd__()
        self.initialize_vars()
        self.proper_setup = self.check_setup()
        
        if self.args.setup == 'True':
            print('Setting up initial directory for management.')
            self.__setup__()
        
        self.pseudoMap = {'SG15' : 'SG15/$ID_ONCV_PBE.upf', # readable format
                          'GBRV' : 'GBRV/$ID_pbe.uspp', # not readable
                          'GBRV_v1.5' : 'GBRV_v1.5/$ID_pbe_v1.uspp.F.UPF', # readable format
                          'dojo': 'dojo/$ID.upf', # readable format
                          } 

    def __setup__(self, verbose = True, overwrite = False):
        # make necessary head-folders
        for dir_to_make in [calc_folder, molecule_folder, inputs_folder, results_folder]:
            if not os.path.exists(dir_to_make):
                os.mkdir(dir_to_make)
            else:
                print('Directory: '+dir_to_make+' already exists. No changes made.')
        # make calc sub-folders
        for cs in self.calc_subfolders:
            folder = os.path.join(calc_folder, cs)
            if not os.path.exists(folder):
                os.mkdir(folder)
            else:
                print('Directory: '+dir_to_make+' already exists. No changes made.')
        if verbose: print('Head directories created.')
        
        # copy default files to molecules and inputs
        def_inputs_folder = os.path.join(defaults_folder, 'inputs')
        for file in os.listdir(def_inputs_folder):
            file_loc = os.path.join(def_inputs_folder, file)
            self.run('cp ' + file_loc + ' ' + os.path.join(inputs_folder, file))
        
        def_mols_folder = os.path.join(defaults_folder, 'molecules')
        for folder in os.listdir(def_mols_folder):
            if folder in os.listdir(molecule_folder):
                print('Molecule: '+folder+' already exists. No changes made.')
                continue
            folder_loc = os.path.join(def_mols_folder, folder)
            self.run('cp -r ' + folder_loc + ' ' + os.path.join(molecule_folder, folder))
        if verbose: print('Default input files and molecules added.')
        
        if overwrite or not os.path.exists('./manager_control.txt'):
            self.run('cp ' + os.path.join(defaults_folder, 'manager_control.txt') + ' ./manager_control.txt')
        if overwrite or not os.path.exists('./readme.txt'):
            self.run('cp ' + os.path.join(defaults_folder, 'readme.txt') + ' ./readme.txt')
        print('\nSuccessfully setup directory! Please see manager_control.txt for help.\n')
    
    def check_setup(self):
        if not all([os.path.exists(f) for f in [calc_folder, molecule_folder, 
                    inputs_folder, results_folder]]):
            return False
        if any([x not in os.listdir(calc_folder) for x in self.calc_subfolders]):
            return False
        return True
    
    def run(self, cmd):
        subprocess.call(cmd, shell=True)
    
    def initialize_vars(self):
        self.calc_subfolders = ['surfs', 'molecules', 'adsorbed', 'desorbed', 'neb', 'bulks']
        self.data_file = os.path.join(results_folder, 'all_data.json')
        self.default_adsorbate_distance = 2.0
        self.default_desorbed_distance = 5.0

    def __get_user_inputs__(self):
        # get all user inputs from command line
        parser = argparse.ArgumentParser()
        
        parser.add_argument('-s', '--setup', help='Setup downstream folders for management. (-s True)',
                            type=str, default='False')
        parser.add_argument('-t', '--run_time', help='Time to run jobs. Default 12 (hours).',
                            type=int, default=12)
        parser.add_argument('-u', '--rerun_unconverged', help='Rerun all unconverged calculations being managed, '+
                            'requires "check_calcs". Default True.',type=str, default='True')
        parser.add_argument('-g', '--gpu', help='If True, run all calculations on gpu nodes. Default False.',
                            type=str, default='False')
        parser.add_argument('-m', '--make_new', help='Make new calculations based on requested calcs.'+
                            ' Default True.',type=str, default='True')
        parser.add_argument('-b', '--backup', help='Whether to backup calcs folder. Default False.',
                            type=str, default='False')
        parser.add_argument('-dos', '--save_dos', help='Whether to save DOS data to all_data. Default False.',
                            type=str, default='False')
        parser.add_argument('-ra', '--read_all', help='Read all folders for new data. Does not use convergence '+
                            'file to speed up reading. Default False.', type=str, default='False')
        parser.add_argument('-a', '--analyze', help='Runs analysis on converged calcs, requires "save".'+
                            ' Default False. INCOMPLETE.',type=str, default='False')
        parser.add_argument('-cc', '--check_calcs', help='Check convergence of all managed calculations. '+
                            'Default True.',type=str, default='True')
        parser.add_argument('-sd', '--selective_dynamics', 
                            help='Whether to add selective dynamics to surface / adsorbate calcs. '+
                            'Set to False for 2D materials. Default True.',type=str, default='True')
        parser.add_argument('-v', '--save', help='Save all newly processed data, requires "check_calcs".'+
                            ' Default True.',type=str, default='True')
        parser.add_argument('-rn', '--run_new', help='Run all newly setup calculations, requires "make_new".'+
                            ' Default True.',type=str, default='True')
        parser.add_argument('-ads', '--add_adsorbed', help='Add all requested adsorbates to converged surfs, '
                            +'requires "make_new". Default True.',type=str, default='True')
#        parser.add_argument('-dist', '--adsorbate_distance', help='Standard distance from surface to adsorbate,'
#                            +' requires "add_adsorbed". Default 2.0',type=float, default=2.0)
        parser.add_argument('-des', '--add_desorbed', help='Add all requested desorbed calcs to converged surfs, '
                            +'requires "make_new" and "add_adsorbed". Needed for NEB. Default True.',
                            type=str, default='True')
        parser.add_argument('-mol', '--add_molecules', help='Add all requested molecules, requires "make_new".'+
                            ' Default True.',type=str, default='True')
        parser.add_argument('-neb', '--make_neb', help='Makes NEB calculations from manager_control file, '+
                            'requires "make_new". Default True.',type=str, default='True')
        parser.add_argument('-nebc', '--neb_climbing', help='If True, uses NEB climbing image. Requires'+
                            ' "make_neb". Default True.',type=str, default='True')
        parser.add_argument('-cf', '--current_force', help='If True, displays calc forces. Default True.',
                            type=str, default='True')
        parser.add_argument('-n', '--nodes', help='Nodes per job (total nodes for parallel). Default 1.',
                            type=int, default=1)
        parser.add_argument('-c', '--cores', help='Cores per node.',
                            type=int, default=core_architecture)
        parser.add_argument('-r', '--short_recursive', help='Run jobs recursively on short queue until complete.'+
                            ' Very helpful when queue is busy. Default False.',type=str, default='False')
        parser.add_argument('-rhe', '--rhe_zeroed', help='If True, converts all biases to be zeroed '+
                            'at 0V vs. RHE rather than 0V vs. SHE (if False).', type=str, default='False')
        parser.add_argument('-q', '--qos', help='Whether qos should be high (True) or standard. Default False.',
                            type=str, default='False')
        parser.add_argument('-p', '--parallel', help='Runs multiple calcs together on a single node. Input'+
                            ' should be max number (int) of calcs to run together per node. Default 1.',
                            type=int, default=1)
        parser.add_argument('-conv', '--use_convergence', help='If True (default), copy convergence '+
                           'file to new calc folders and update.',
                            type=str, default='True')
        parser.add_argument('-kptd', '--kpoint_density', help='Kpoint grid density (default 350)',
                            type=int, default=350)
        parser.add_argument('-kptdb', '--kpoint_density_bulk', help='Bulk Kpoint grid density (default 1000)',
                            type=int, default=1000)
        parser.add_argument('-elec', '--copy_electronic', help='If True, copy electronic state files '+
                           ' to new bias folders based on converged biases (default False).',
                            type=str, default='False')
        parser.add_argument('-sp', '--smart_procs', help='Whether to use smart system for setting number '+
                            'of processes (default True)',type=str, default='True')
        parser.add_argument('-fix', '--calc_fixer', help='Whether to use the smart error fixer for '+
                            'failed calcs (default False)',type=str, default='False')
        parser.add_argument('-fr', '--full_rerun', help='Rerun ALL calculations. Be careful '+
                            'with this. Calcs start at current state. (default False)',type=str, default='False')
        parser.add_argument('--clean_wfns', help='Delete converged wfns  to reduce mem. Be careful '+
                            'with this. (default False)',type=str, default='False')
        parser.add_argument('-bundle', '--bundle_jobs', help='Bundle all jobs together. Useful for Cori / Perl.'+
                            ' (default False)',type=str, default='False')
        parser.add_argument('-use_nb', '--use_no_bias_structure', help='Whether surface and ads calcs should'+
                            ' be upgraded from no_bias or 0V converged calcs (default False).'+
                            ' False = calcs are independently setup and run. Should be used with -elec tag.',
                            type=str, default='False')
        self.args = parser.parse_args()

    def __get_run_cmd__(self):
        self.run_cmd = run_command
        # add user inputs
        self.run_cmd += ' -t '+str(self.args.run_time)
        self.run_cmd += ' -n '+str(self.args.nodes)
        self.run_cmd += ' -c '+str(self.args.cores)
        if self.args.qos == 'True':
            self.run_cmd += ' -q high'
        if self.args.short_recursive == 'True':
            self.run_cmd += ' -r True'
        if self.args.gpu == 'True':
            self.run_cmd += ' -g True'
            
    def scan_calcs(self, all_data, running_dirs, verbose = True, force_limit = 50):
        '''
        Main function for scanning through all previously-created sub-directories
        Functions independently from manager_control
        Functions:
            1) Scans through "calcs" directory 
            2) Saves data for all converged calculations
            3) Lists unconverged directories for rerunning
            4) Lists directories without inputs or CONTCARs to be started
        '''
        # look through all calc folders for converged calcs, unconverged calcs, and calcs to setup
        ncalcs = 0
        nnewcalcs = 0
        ndos = 0
        dos_per_file = 40
        add_inputs = []
        run_new = []
        rerun = []
        failed_calcs = []
        running_parallel = self.get_parallel_running() # TODO: setup function and rules
        
        # add multi-file dos writing and dos file tracker
        if self.args.save_dos == 'True':
            print('\n*** Reading DOS files ***\n')
            if ope(opj(results_folder, 'dos_tracker.json')):
                with open(opj(results_folder, 'dos_tracker.json'),'r') as f:
                    dos_tracker = json.load(f)
            else:
                dos_tracker = {}
            # set dos file
            if not ope(opj(results_folder, 'dos')):
                os.mkdir(opj(results_folder, 'dos'))
            if len(dos_tracker) == 0:
                dos_file_count = 1
                all_dos = {}
            else:
                # some dos are already saved 
                dos_file_list = [v['n'] for k,v in dos_tracker.items()]
                dos_file_count = int(np.max(dos_file_list))
                nmax_dos = len([n for n in dos_file_list if n == dos_file_count])
                #print('Stats:', dos_file_count, nmax_dos, dos_per_file)
                if nmax_dos == dos_per_file:
                    # already fully saved to this file
                    dos_file_count += 1
                    all_dos = {}
                else:
#                    print('Reading DOS file: '+ opj('dos', 
#                                  'all_dos_'+str(dos_file_count)+'.json'))
                    with open(opj(results_folder, 'dos', 
                                  'all_dos_'+str(dos_file_count)+'.json'),'r') as f:
                        all_dos = json.load(f) 
                print('dos file counter initiated at:', dos_file_count,
                      'with',len(all_dos),'current dos.')
           
        if 'converged' not in all_data:
            all_data['converged'] = []
        if verbose: print('\n----- Scanning Through Calculations -----')
        
        for subf in self.calc_subfolders:
            calc_subfolder = opj(calc_folder, subf)
            print('Scanning: '+ calc_subfolder)
            
            for root, folders, files in os.walk(calc_subfolder):
                if 'POSCAR' not in files and 'inputs' not in files:
                    continue
                if 'neb' in root and len(root.split(os.sep)) >= 7:
                    # ignore neb subdirs
                    continue
                if '__' in root:
                    continue
                if verbose: 
                    print('\nFolder found at:', root)
                
                ncalcs += 1
                full_root = os.path.join(self.cwd, root)
                if full_root in running_dirs:
                    if verbose: print('Currently Running.')
                    continue
                
                # full rerun for all calcs 
                if self.args.full_rerun == 'True' and 'inputs' in files and 'CONTCAR' in files:
                    print('Added to rerun (full rerun)')
                    rerun.append(root)
                    continue
                
                if (self.args.save_dos == 'True' and root in all_data['converged'] 
                    and root not in dos_tracker and 'opt.log' in files):
                    print('Previously Converged. Adding DOS.')
                
                elif root in all_data['converged'] and 'opt.log' in files: # NEW: delete opt.log to force rerun 
                    if verbose: print('Previously Converged.')
                    continue
                
                nnewcalcs += 1
                if nnewcalcs % 100 == 0:
                    print('*** Temp convergence save ***')
                    with open(self.data_file, 'w') as f:
                        json.dump(all_data, f)
                
                # get type of calculation
                calc_type = None
                for subf in self.calc_subfolders:
                    tag = os.path.join(calc_folder, subf)
                    if tag in root:
                        calc_type = subf
                        continue
                if calc_type is None:
                    print('Error: No calc_type found for root: '+root+' Skipping.')
                    continue
    #            sub_dirs = root.split(tag)[-1].split(os.sep)
                sub_dirs = root.split(os.sep)
                
                if len(sub_dirs) < 4 and calc_type != 'bulks':
                    if verbose: print('Not a calculation directory.')
                    continue
                if calc_type == 'neb' and len(sub_dirs) > 5:
                    print('Skipping NEB sub-directory.')
                    continue
                if 'inputs' not in files:
                    add_inputs.append(root)
                    if verbose: print('Adding inputs.')
                    continue
                if 'CONTCAR' not in files and calc_type not in ['neb']:
                    run_new.append(root)
                    if verbose and self.args.run_new == 'True': print('Running unstarted job.')
                    continue
                
                # read calc data at root
                if calc_type not in ['neb']:
                    data = h.read_data(root)
                    cf = '%.3f'%data['current_force'] if data['current_force'] != 'None' else 'None'
                    skip_high_forces = (False if (data['current_force'] == 'None' or 
                                                  data['current_force'] < force_limit) else True)
                    
                    # add dos files 
                    if data['converged'] and self.args.save_dos == 'True':
                        if root not in dos_tracker:
                            dos_data = h.get_jdos(root)
                            all_dos[root] = dos_data
                            dos_tracker[root] = {'n': dos_file_count, 
                                                 'file': 'all_dos_'+str(dos_file_count)+'.json'}
                            ndos += 1
                            if ndos % dos_per_file == 0: # added intermittent saving of all_dos file 
                                with open(opj(results_folder, 'dos', 
                                              'all_dos_'+str(dos_file_count)+'.json'),'w') as f:
                                    json.dump(all_dos, f)
                                with open(opj(results_folder, 'dos_tracker.json'),'w') as f:
                                    json.dump(dos_tracker, f)
                                print('Int. DOS save at', ndos, 
                                      '- all_dos reset - dos counter =', dos_file_count)
                                dos_file_count += 1
                                all_dos = {}
                
                # save molecule data
                if calc_type == 'molecules':
                    if verbose: 
                        if self.args.current_force == 'True': 
                            print('Molecule calc read (force='+cf+')')
                        else:
                            print('Molecule calc read.')
                    mol_name = sub_dirs[2]
                    bias_str = sub_dirs[3]
                    bias = h.get_bias(bias_str)
                    if mol_name not in all_data:
                        all_data[mol_name] = {}
                    data['bias'] = bias
                    all_data[mol_name][bias_str] = data
                    if data['converged']:
                        all_data['converged'].append(root)
                        if verbose: print('Molecule calc converged.')
                    else:
                        if not skip_high_forces: 
                            rerun.append(root)
                            if verbose: print('Molecule calc not converged. Adding to rerun.')
                        else:
                            failed_calcs.append(root)
                    continue
                
                # save bulk data
                elif calc_type == 'bulks':
                    if verbose: 
                        if self.args.current_force == 'True': 
                            print('Bulk calc read (force='+cf+')')
                        else:
                            print('Bulk calc read.')
                    bulk_name = sub_dirs[2]
                    if bulk_name not in all_data:
                        all_data[bulk_name] = {}
                    if 'bulk' not in all_data[bulk_name]:
                        all_data[bulk_name]['bulk'] = {}
                    all_data[bulk_name]['bulk'] = data
                    if data['converged']:
                        all_data['converged'].append(root)
                        if verbose: print('Bulk calc converged.')
                    else:
                        if not skip_high_forces: 
                            rerun.append(root)
                            if verbose: print('Bulk calc not converged. Adding to rerun.')
                        else:
                            failed_calcs.append(root)
                    continue
                
                # save surface data
                elif calc_type == 'surfs':
                    if verbose: 
                        if self.args.current_force == 'True': 
                            print('Surface calc read (force='+cf+')')
                        else:
                            print('Surface calc read.')
                    surf_name = sub_dirs[2]
                    bias_str = sub_dirs[3]
                    bias = h.get_bias(bias_str)
                    if surf_name not in all_data:
                        all_data[surf_name] = {}
                    if 'surf' not in all_data[surf_name]:
                        all_data[surf_name]['surf'] = {}
                    data['bias'] = bias
                    all_data[surf_name]['surf'][bias_str] = data
                    if data['converged']:
                        all_data['converged'].append(root)
                        if verbose: print('Surface calc converged.')
                    else:
                        if not skip_high_forces: 
                            rerun.append(root)
                            if verbose: print('Surface calc not converged. Adding to rerun.')
                        else:
                            failed_calcs.append(root)
                    continue
                
                # save adsorbate and desorbed state calcs
                elif calc_type in ['adsorbed', 'desorbed']:
                    # dic = surf: calc_type(s): mol: biases: configs: data. no configs for desorbed
                    if verbose: 
                        if self.args.current_force == 'True': 
                            print('Adsorbed/Desorbed calc read (force='+cf+')')
                        else:
                            print('Adsorbed/Desorbed calc read.')
                    surf_name = sub_dirs[2]
                    mol_name = sub_dirs[3]
                    bias_str = sub_dirs[4]
                    mol_config = None
                    if calc_type == 'adsorbed': 
                        mol_config = sub_dirs[5]
                    bias = h.get_bias(bias_str)
                    if surf_name not in all_data or 'surf' not in all_data[surf_name]:
    #                    print('Surface: '+surf_name+' must be created before adsorbed/desorbed calcs can be read!')
                        continue
    #                print(surf_name)
    #                if surf_name not in all_data:
    #                    all_data[surf_name] = {}
                    if calc_type not in all_data[surf_name]:
                        all_data[surf_name][calc_type] = {}
                    if mol_name not in all_data[surf_name][calc_type]:
                        all_data[surf_name][calc_type][mol_name] = {}
                    if bias_str not in all_data[surf_name][calc_type][mol_name]:
                        all_data[surf_name][calc_type][mol_name][bias_str] = {}
                        
                    if bias_str in all_data[surf_name]['surf'] and all_data[surf_name]['surf'][bias_str]['converged']:
                        data['bias'] = bias
                        if calc_type == 'adsorbed':
                            all_data[surf_name][calc_type][mol_name][bias_str][mol_config] = data
                        elif calc_type == 'desorbed':
                            all_data[surf_name][calc_type][mol_name][bias_str] = data
                        if data['converged']:
                            all_data['converged'].append(root)
                            if verbose: print('Adsorbed/Desorbed calc converged.')
                        else:
                            if not skip_high_forces: 
                                rerun.append(root)
                                if verbose: print('Adsorbed/Desorbed calc not converged. Adding to rerun.')
                            else:
                                failed_calcs.append(root)
                    else:
                        print('Surface: '+surf_name+' at bias '+bias_str+
                              ' must be converged before adsorbed/desorbed can be saved.')
                        if not data['converged']:
                            if not skip_high_forces: 
                                rerun.append(root)
                                if verbose: print('Adsorbed/Desorbed calc not converged. Adding to rerun.')
                            else:
                                failed_calcs.append(root)
                    continue
                        
                elif calc_type == 'neb':
                    # calc/neb/surf/path_name(Path_#-start-finish)/bias/data
                    if verbose: print('NEB calc read.')
    #                continue
                    # dic = surf: 'neb': mol: bias: path:
                    surf_name = sub_dirs[2]
                    path_name = sub_dirs[3]
                    bias_str = sub_dirs[4]
                    bias = h.get_bias(bias_str)
                    if surf_name not in all_data or 'surf' not in all_data[surf_name]:
                        print('WARNING: Surface does not exist with same name, nowhere to save neb calc.')
                        continue
                    if calc_type not in all_data[surf_name]:
                        all_data[surf_name][calc_type] = {}
                    if path_name not in all_data[surf_name][calc_type]:
                        all_data[surf_name][calc_type][path_name] = {}
                    if bias_str not in all_data[surf_name][calc_type][path_name]:
                        all_data[surf_name][calc_type][path_name][bias_str] = {}
                    neb_data = h.get_neb_data(root, bias)
                    all_data[surf_name][calc_type][path_name][bias_str] = neb_data
                    if neb_data['converged']:
                        print('NEB path '+path_name+' for '+surf_name+' at '+bias_str+' converged.')
                    else:
    #                    print('NEB path '+path_name+' for '+surf_name+' at '+bias_str+' not converged.'
    #                          +' Added to rerun.')
                        continue # remove after testing
                        if neb_data['opt'] != 'None' and neb_data['opt'][-1]['force'] < force_limit:
                            print('NEB path '+path_name+' for '+surf_name+' at '+bias_str+' not converged.'
                                  +' Added to rerun.')
                            rerun.append(root)
                        else:
                            neb_force = '%.3f'%(neb_data['opt'][-1]['force']) if neb_data['opt'] != 'None' else 'None'
                            print('NEB path '+path_name+' for '+surf_name+' at '+bias_str+' not converged.'
                                  +' Skipping due to high forces ('+neb_force+')')
                    continue
        
        # save remaining dos 
        if self.args.save_dos == 'True':
            with open(opj(results_folder, 'dos', 'all_dos_'+str(dos_file_count)+'.json'),'w') as f:
                json.dump(all_dos, f)
            with open(opj(results_folder, 'dos_tracker.json'),'w') as f:
                json.dump(dos_tracker, f)
            print('\n***** Saved All DOS *****\n')
        
        return all_data, add_inputs, rerun, run_new, failed_calcs, ncalcs

    def rerun_calcs(self, rerun):
        print('\n----- Rerunning unconverged calcs -----\n')
        
        for ii,root in enumerate(rerun):
            if ii%100 == 99:
                print('\n...\n')
                sleep(5) # pause for 5 seconds before submitting more jobs
            calc_type = root.split(os.sep)[1]
            os.chdir(root)
            print('Rerunning: '+self.get_job_name(root))
            if self.args.smart_procs == 'True':
#                print('debug: smart_procs = '+self.args.smart_procs, 'calc type:', calc_type)
                if calc_type in ['adsorbed','surfs','desorbed','neb']:
#                    print('-procs 2')
                    self.run(self.run_cmd + ' -o '+self.get_job_name(root)+ ' -m 2') 
                else:
                    self.run(self.run_cmd + ' -o '+self.get_job_name(root)+ ' -m 8') 
            else:
                self.run(self.run_cmd + ' -o '+self.get_job_name(root))
            os.chdir(self.cwd)

    def update_rerun(self, rerun):
        for root in rerun:
            os.chdir(root)
            self.failed_rerun_fixer(root, 
                        auto_delete = True if self.args.calc_fixer == 'True' else False)
            inputs = h.read_inputs('./')
            inputs['restart'] = 'True'
            h.write_inputs(inputs, './')
            os.chdir(self.cwd)

    def failed_rerun_fixer(self, folder, auto_delete = False):
        '''
        Tries to fix errors that show up when rerunning a calc that previously failed.
        Current fixes:
            1) Length of state files (e.g., wfns) is incorrect
                Fix: delete state files 
        '''
        try:
            with open('out', 'r', errors='ignore') as f:
                outf = f.read()
            end_lines = outf.split('\n')[-10:]
            if 'Failed.' not in end_lines:
                return True
            for line in end_lines:
                # fillings is wrong size, NEED TO RERUN WITHOUT SYMMETRY (kpoints changes during opt)
                if "Length of '" in line:
                    if not auto_delete:
                        print('"State" files are incorrect size, job may fail: '+folder)
                    else:
                        print('"State" files are incorrect size, files removed.')
                        cwd = os.getcwd()
                        os.chdir(folder)
                        for file in ['fillings','wfns','eigenvals','eigenStats','fluidState','nbound']:
                            self.run('rm '+file)
                        os.chdir(cwd)
                        break
        except:
            pass
        return True

    def get_job_name(self, root):
        folders = root.split(os.sep)[1:]
        folders[0] = folders[0][:3]
        return '-'.join(folders)

    def make_new_calcs(self, converged):
        '''
        Main management function for creating and upgrading calculations based on manager_control.txt
        Main functionalities:
            1) Upgrades surf calcs from No_bias -> 0V and 0V to other biases
            2) Adds requested molecules as adsorbates on converged surfaces at same bias
                2.1) Creates single point calculations of molecules above converged surface
                     at same bias. These are desorbed calculations and are needed for NEB.
                2.2) Runs molecules at same bias and solvent to allow for binding energy analysis
            3) Sets up NEB calculations from converged adsorbed+desorbed calculations at same bias
        '''
        # read manager_control.txt
        with open('manager_control.txt', 'r') as f:
            mc = f.read()
        print('\n----- Manager Control -----')
        # get dictionary of managed calcs
        managed_calcs = self.read_manager_control(mc)
        if managed_calcs == False:
            return False
        print('Manager control file successfully read.')
        # setup sub dirs based on converged and managed calcs
        new_folders = self.setup_managed_calcs(managed_calcs, converged)
        return new_folders

    def setup_managed_calcs(self, managed, converged, ads_distance = None, 
                            desorbed_single_point = True): #surf_selective_dyn = True
        '''
        Sets up all new calculations based on inputs from manager_control.txt
        '''
        sd = True if self.args.selective_dynamics == 'True' else False
        use_no_bias = True if self.args.use_no_bias_structure == 'True' else False
        
        # Done: remove dependence upon No_bias calculation
        
        if ads_distance is None:
            ads_distance = self.default_adsorbate_distance
        # creates new calc folders with POSCARs and inputs 
        # depends on args: add_adsorbed, add_desorbed, add_molecules, make_neb
        new_roots = []
        managed_mols = []
        # setup molecule folders in calc_folder
        if self.args.add_molecules == 'True': # add convergence file to mols?
            for mol, molv in managed['molecules'].items():
                # ref_mols used for binding analysis, mol used for desorb SP calcs
#                try:
                ref_mols = self.get_ref_mols(mol)
#                except:
#                    ref_mols = []
#                    print('WARNING: No reference molecules found for: '+mol+', add manually if needed.')
                if 'desorb' in molv:
                    ref_mols += [mol]
                else:
                    managed_mols.append(mol)
                for ref_mol in ref_mols:
#                    if ref_mol in managed_mols:
#                        continue # It seems like this is taken care of by os.path.exists(bias_dir)
                    biases = list(set(molv['biases']))
                    mol_location = self.get_mol_loc(ref_mol)
                    if mol_location == False:
                        continue
                    if not os.path.exists(os.path.join(calc_folder, 'molecules', ref_mol)):
                        os.mkdir(os.path.join(calc_folder, 'molecules', ref_mol))
                    for bias in biases:
                        bias_dir = os.path.join(calc_folder, 'molecules', ref_mol, h.get_bias_str(bias))
                        if bias_dir in converged:
                            continue
                        if os.path.exists(bias_dir):
                            # do not setup folders that exist, they are already setup
                            continue
                        os.mkdir(bias_dir)
                        # add POSCAR to bias folder
                        self.run('cp '+mol_location+' '+os.path.join(bias_dir, 'POSCAR'))
                        # add inputs to bias folder
                        self.run('cp '+os.path.join(inputs_folder, 'molecules_inputs')+' '
                                 +os.path.join(bias_dir, 'inputs'))
                        tags = ['target-mu '+ ('None' if bias in ['None','none','No_bias'] 
                                else '%.4f' % self.get_mu(bias, h.read_inputs(bias_dir)))]
                        self.add_tags(bias_dir, tags)
                        good_setup = self.set_input_system_params(bias_dir, 'molecules')
                        if not good_setup:
                            print('Setup failed: '+bias_dir)
                            continue
                        new_roots.append(bias_dir)
                        print('Reference molecule '+ref_mol+' at bias '+h.get_bias_str(bias)+
                              ' properly setup for mol '+mol)
                    managed_mols.append(ref_mol)
        
        # setup managed surfs, adsorbates, desorbed states, and NEB jobs
        for surf,v in managed.items():
            # 1) add new surfs and perform surf upgrading
            if surf in ['molecules'] or 'biases' not in v:
                continue
            print('\nManagement of surface: ' + surf)
            surf_roots = [os.path.join(calc_folder, 'surfs', surf, h.get_bias_str(bias)) 
                          for bias in v['biases']]
            for i, root in enumerate(surf_roots):
                # check if root is converged, if so it can be skipped
                if root in converged:
                    continue
                # check if root exists with POSCAR, if so it is being managed by rerun_calcs or add_inputs
                if os.path.exists(os.path.join(root, 'POSCAR')) or os.path.exists(os.path.join(root, 'inputs')):
                    continue
                # root does not exist yet, check on bias dependency
                bias = v['biases'][i]
                if type(bias) == float:
                    # bias is zero, ensure no-mu is converged
                    nomu_root = os.path.join(calc_folder, 'surfs', surf, 'No_bias') 
                    zero_root = os.path.join(calc_folder, 'surfs', surf, '0.00V')
                    if zero_root in converged:
                        # upgrade from no_bias (which exists)
                        self.upgrade_calc(root, zero_root, bias, v['tags'] if 'tags' in v else [])
                        new_roots.append(root)
                    elif nomu_root in converged:
                        # upgrade from no_bias (which exists)
                        self.upgrade_calc(root, nomu_root, bias, v['tags'] if 'tags' in v else [])
                        new_roots.append(root)
                    elif 'No_bias' not in v['biases'] and not use_no_bias:
                        # No bias not requested, run directly 
                        good_setup = self.make_calc(calc_folder, surf, root, v, bias, sd = sd)
                        if good_setup:
                            new_roots.append(root)
                        continue
                    else:
                        # waiting for No_bias to converge
                        continue
                elif bias == 'No_bias':
                    good_setup = self.make_calc(calc_folder, surf, root, v, bias, sd = sd)
                    if good_setup:
                        new_roots.append(root)
                    continue
                
            # 2) add adsorbates to converged surfaces at same bias
            if self.args.add_adsorbed == 'True':
                for mol, mv in v.items():
                    if mol in ['biases','tags','conv-tags','NEB']:
                        continue
                    if mol not in managed_mols:
                        print('Cannot add adsorbate, molecule '+mol+' not setup correctly.')
                        continue
                    # add each molecule requested at each bias on surface of corresponding bias
                    # mol = name of molecule to add, mv is dict with 'sites', 'biases' and 'tags' (optional)
                    if 'ads_dist' in mv:
                        # if listed, change ads_distance
                        ads_dist = mv['ads_dist']
                    else:
                        ads_dist = ads_distance
                    # create all adsorbate roots to make
                    if 'biases' not in mv:
                        print('Error: No biases found for mol '+mol+' for surf '+surf)
#                    ads_roots = [os.path.join(calc_folder, 'adsorbed', surf, mol, h.get_bias_str(bias)) 
#                          for bias in mv['biases']]
                    if not os.path.exists(os.path.join(calc_folder, 'adsorbed', surf)):
                        os.mkdir(os.path.join(calc_folder, 'adsorbed', surf))
                    if not os.path.exists(os.path.join(calc_folder, 'adsorbed', surf, mol)):
                        os.mkdir(os.path.join(calc_folder, 'adsorbed', surf, mol))
                        
                    if 'conv-tags' in v or 'conv-tags' in mv:
                        convtags = mv['conv-tags'] if 'conv-tags' in mv else {}
                        if 'conv-tags' in v: # add all tags from surface
                            for step, vals in v['conv-tags'].items():
                                if step in convtags:
                                    convtags[step] += vals
                                else:
                                    convtags[step] = vals
                        
                    for bias in mv['biases']:
                        # new option to upgrade ads calcs from no_bias or 0V case (not default)
                        bias_str = h.get_bias_str(bias)
                        if use_no_bias and bias_str not in ['0.00V', 'No_bias']: 
                            nomu_headroot = os.path.join(calc_folder, 'adsorbed', surf, mol, 'No_bias') 
                            zero_headroot = os.path.join(calc_folder, 'adsorbed', surf, mol, '0.00V')
                            # scan through subdirs (sites) of headroot dirs
                            for headroot in [zero_headroot, nomu_headroot]:
                                if not os.path.exists(headroot):
                                    continue # skip dir if it doesn't exist
                                site_dirs = [opj(headroot, f) for f in os.listdir(headroot)
                                             if os.path.isdir(opj(headroot, f))]
                                for sitedir in site_dirs:
                                    # look through site directories at current bias case (0V or nomu)
                                    if '__' in sitedir:
                                        continue
                                    if sitedir in converged:
                                        site = sitedir.split(os.sep)[-1]
                                        newroot = opj(calc_folder, 'adsorbed', surf, mol, bias_str, site)
                                        if os.path.exists(newroot):
                                            # skip existing dirs, including those just made at other headroot
                                            continue
                                        os.mkdir(newroot) # make new calc dir at bias from conv. nomu or 0V
                                        # upgrade calc copies conv CONTCAR and makes inputs/convergence files
                                        if self.args.copy_electronic != 'True':
                                            print('WARNING: -use_no_bias should be used with -elec to reduce'
                                                  +' calc initialization runtime!')
                                        self.upgrade_calc(newroot, sitedir, bias, convtags) 
                                        new_roots.append(newroot)
                            continue # do not proceed to make more calcs for biases w/o conv str. if use_no_bias
                        
                        surf_root = os.path.join(calc_folder, 'surfs', surf, h.get_bias_str(bias))
                        if surf_root not in converged:
                            continue
                        # surf at same bias exists
                        st_surf = Structure.from_file(os.path.join(surf_root, 'CONTCAR'))
                        ads_dic = {mol: mv['sites']}
                        head_folder = os.path.join(calc_folder, 'adsorbed', surf, mol, h.get_bias_str(bias))
                        
                        # add molecule as adsorbate at all requested destinations
                        surf_ads = add_adsorbates(st_surf, ads_dic, ads_distance = ads_dist,
                                                  freeze_depth=2.0 if sd else -1.0)
                        # save any new folders created, *** ANY EXISITING FOLDERS ARE IGNORED *** 
                        # does not use converged but has same functionality
                        # skipping earlier will miss new sites
                        save_locs = save_structures(surf_ads[mol], head_folder, skip_existing = True)
                        # add surfs to setup_new
                        if len(save_locs) == 0:
                            continue
                        for sl in save_locs:
                            # check if inputs is already in folder
                            if os.path.exists(os.path.join(sl, 'inputs')):
                                continue
                            print('Added adsorbate folder: '+sl)
                            self.run('cp '+os.path.join(inputs_folder, 'adsorbed_inputs')
                                     +' '+os.path.join(sl, 'inputs'))
                            
                            # add convergence file
                            if self.args.use_convergence == 'True':
                                if os.path.exists(os.path.join(inputs_folder, 'convergence')):
                                    self.run('cp '+os.path.join(inputs_folder, 'convergence')+
                                         ' '+os.path.join(sl, 'convergence'))
                                else:
                                    print('WARNING: "convergence" file not found in '+inputs_folder)
                                
#                                if 'conv-tags' in v or 'conv-tags' in mv:
#                                    convtags = mv['conv-tags'] if 'conv-tags' in mv else {}
#                                    if 'conv-tags' in v: # add all tags from surface
#                                        for step, vals in v['conv-tags'].items():
#                                            if step in convtags:
#                                                convtags[step] += vals
#                                            else:
#                                                convtags[step] = vals
                                    
                                    # update convergence file
                                    self.set_conv_tags(sl, convtags)
                            
                            # tags is a list
                            tags = mv['tags'] if 'tags' in mv else []
                            if 'tags' in v: # add all tags from surface
                                tags += v['tags']
#                            tags += 'target-mu '+ ('None' if bias in ['None','none','No_bias'] else '%.2f'%bias)
                            tags += ['target-mu '+ ('None' if bias in ['None','none','No_bias'] 
                                     else '%.4f' % self.get_mu(bias, h.read_inputs(sl), tags))]
                            self.add_tags(sl, tags)
                            
                            # set nbands and kpts from surf
                            good_setup = self.set_input_system_params(sl, 'adsorbed')
                            if not good_setup:
                                print('Setup failed: '+sl)
                                continue
                            new_roots.append(sl)
                
                    # setup single point calcs of molecules (at bias) above surface (at bias)
                    # is single point a good estimate?
                    if self.args.add_desorbed == 'True' and 'desorb' in mv:
                        desorb_biases = mv['desorb']
                        if 'desorb_dist' in mv:
                            des_dist = mv['desorb_dist']
                        else:
                            des_dist = self.default_desorbed_distance
                        if not os.path.exists(os.path.join(calc_folder, 'desorbed', surf)):
                            os.mkdir(os.path.join(calc_folder, 'desorbed', surf))
                        if not os.path.exists(os.path.join(calc_folder, 'desorbed', surf, mol)):
                            os.mkdir(os.path.join(calc_folder, 'desorbed', surf, mol))
                        for bias in desorb_biases:
                            surf_root = os.path.join(calc_folder, 'surfs', surf, h.get_bias_str(bias))
                            mol_root = os.path.join(calc_folder, 'molecules', mol, h.get_bias_str(bias))
                            if surf_root not in converged or mol_root not in converged:
#                                print('Desorbed waiting on converged surface and molecule.')
                                continue
                            # surf and mol at same bias exists
                            st_surf = Structure.from_file(os.path.join(surf_root, 'CONTCAR'))
                            ads_dic = {mol: ['center']}
                            head_folder = os.path.join(calc_folder, 'desorbed', surf, mol, h.get_bias_str(bias))
                            surf_ads = add_adsorbates(st_surf, ads_dic, ads_distance = des_dist,
                                                      molecules_loc = os.path.join(mol_root,'CONTCAR'),
                                                      freeze_depth=2.0 if sd else -1.0)
                            save_locs = save_structures(surf_ads[mol], head_folder, skip_existing = True,
                                                        single_loc=True)
                            if len(save_locs) == 0:
                                continue
                            for sl in save_locs:
                                # check if inputs is already in folder
                                if os.path.exists(os.path.join(sl, 'inputs')):
                                    continue
                                print('Added desorbed folder: '+sl)
                                self.run('cp '+os.path.join(inputs_folder, 'desorbed_inputs')
                                         +' '+os.path.join(sl, 'inputs'))
                                
                                # add convergence file
                                if self.args.use_convergence == 'True':
                                    if os.path.exists(os.path.join(inputs_folder, 'convergence')):
                                        self.run('cp '+os.path.join(inputs_folder, 'convergence')+
                                             ' '+os.path.join(sl, 'convergence'))
                                    else:
                                        print('WARNING: "convergence" file not found in '+inputs_folder)
                                    
                                    if 'conv-tags' in v or 'conv-tags' in mv:
                                        convtags = mv['conv-tags'] if 'conv-tags' in mv else {}
                                        if 'conv-tags' in v: # add all tags from surface
                                            for step, vals in v['conv-tags'].items():
                                                if step in convtags:
                                                    convtags[step] += vals
                                                else:
                                                    convtags[step] = vals
                                        # update convergence file
                                        self.set_conv_tags(sl, convtags)
                                
                                # tags is a list
                                tags = mv['tags'] if 'tags' in mv else []
                                if 'tags' in v: tags += v['tags']
                                tags += ['target-mu '+ ('None' if bias in ['None','none','No_bias'] 
                                         else '%.4f' % self.get_mu(bias, h.read_inputs(sl), tags))]
                                if desorbed_single_point:
                                    tags += ['max_steps 0']
                                self.add_tags(sl, tags)
                                
                                # set nbands and kpts from surf
                                good_setup = self.set_input_system_params(sl, 'desorbed')
                                if not good_setup:
                                    print('Setup failed: '+sl)
                                    continue
                                new_roots.append(sl)
                        
            # Create NEB calcs from converged ads. and des. calcs
            if self.args.make_neb == 'True' and 'NEB' in v:
                # TODO: set up neb calc from already converged path if available!  ************
                # managed[surf]['NEB']=[
                #       {'init': init, 'final': final, 'biases': biases, 'images': images, 
                #       'path_name': path_name} ]
                for neb in v['NEB']:
                    # add NEB folders: calcs/neb/surf/path_number/bias/
                    if not os.path.exists(os.path.join(calc_folder, 'neb', surf)):
                        os.mkdir(os.path.join(calc_folder, 'neb', surf))
                    # get and check dependent dirs
                    bias_strs = [h.get_bias_str(b) for b in neb['biases']]
                    
                    init_path = neb['init']
                    final_path = neb['final']
                    images = neb['images']
                    path_name = neb['path_name']
                    
                    path_folder = os.path.join(calc_folder, 'neb', surf, path_name)
                    if not os.path.exists(path_folder):
                        os.mkdir(path_folder)
                        
                    for bias_str in bias_strs:
                        neb_dir = os.path.join(path_folder, bias_str)
                        if os.path.exists(neb_dir):
                            # managed by rerun, already set up
                            continue
                        init_bias_path = init_path.replace('BIAS',bias_str)
                        final_bias_path = final_path.replace('BIAS',bias_str)
                        if init_bias_path not in converged or final_bias_path not in converged:
                            # surfaces not yet converged.
                            continue 
                        init_st = Structure.from_file(os.path.join(init_bias_path,'CONTCAR'))
                        final_st = Structure.from_file(os.path.join(final_bias_path,'CONTCAR'))
                        # fix issue with structures not having same order
                        init_st_sort = init_st.get_sorted_structure()
                        final_st_sort = final_st.get_sorted_structure()
                        initst, finalst = minimum_movement_strs(init_st_sort, final_st_sort)
                        
                        # does not yet exist
                        os.mkdir(neb_dir)
                        new_init_folder = os.path.join(neb_dir, '00')
                        os.mkdir(new_init_folder)
                        new_final_folder = os.path.join(neb_dir, str(images+1).zfill(2))
                        os.mkdir(new_final_folder)
                        
                        # copy over files
                        for i,folder in enumerate([init_bias_path,final_bias_path]):
                            to_folder = [new_init_folder, new_final_folder][i]
                            for file in ['inputs','Ecomponents','opt.log','out']:
                                cmd = 'cp '+os.path.join(folder, file)+' '+os.path.join(to_folder, file)
                                self.run(cmd)
                            st = [initst, finalst][i]
                            st.to('POSCAR', os.path.join(to_folder, 'CONTCAR'))
                            
                        # both final and initial folders are setup
                        # 1) check for paths with forces < criteria, if so, copy files
                        data_folder = None
                        force_criteria = 3.0
                        for p_root, p_folders, p_files in os.walk(path_folder):
                            if 'neb.log' in p_files:
                                opt = h.ead_optlog(p_root, 'neb.log', verbose = False)
                                if opt == False:
                                    continue
                                if opt[-1]['force'] < force_criteria:
                                    data_folder = p_root
                                    break
                        if data_folder is not None:
                            # copy files from other neb folder
                            print('Copying files for NEB path from '+data_folder+' to '+neb_dir+
                                  '. Copying wfns may take a few minutes.')
                            for image_folder in [str(i+1).zfill(2) for i in range(images)]:
                                from_folder = os.path.join(data_folder, image_folder)
                                to_folder = os.path.join(neb_dir, image_folder)
                                if not os.path.exists(to_folder):
                                    os.mkdir(to_folder)
                                for file in ['CONTCAR', 'wfns', 'fillings', 'fluidState', 'eigenvals',
                                             'nbound', 'd_tot']:
                                    self.run('cp '+os.path.join(from_folder, file)+' '+
                                             os.path.join(to_folder, file))
                            # copy inputs
                            cwd = os.getcwd()
                            os.chdir(neb_dir)
                            self.run('cp 00/inputs ./inputs')
                            os.chdir(cwd)
                            tags = ['restart True']
                            
                            # update initial and final structures to match copied CONTCAR path
                            init_st = Structure.from_file(os.path.join(neb_dir,'00','CONTCAR'))
                            neighbor_st = Structure.from_file(os.path.join(neb_dir,'01','CONTCAR'))
                            nst, ist = minimum_movement_strs(neighbor_st, init_st)
                            ist.to('POSCAR', os.path.join(neb_dir,'00','CONTCAR'))
                            final_st = Structure.from_file(os.path.join(neb_dir,
                                                           str(images+1).zfill(2),'CONTCAR'))
                            neighbor_st = Structure.from_file(os.path.join(neb_dir,
                                                              str(images).zfill(2),'CONTCAR'))
                            nst, fst = minimum_movement_strs(neighbor_st, final_st)
                            fst.to('POSCAR', os.path.join(neb_dir,str(images+1).zfill(2),'CONTCAR'))
                            
                        else:
                            # 2) setup idpp folders from scratch and add inputs for neb
                            cwd = os.getcwd()
                            os.chdir(neb_dir)
                            cmd = ('idpp_nebmake.py 00/CONTCAR '+str(images+1).zfill(2)+
                                   '/CONTCAR '+str(images)+' -idpp')
                            self.run(cmd)
                            self.run('cp 00/inputs ./inputs')
                            os.chdir(cwd)
                            tags = ['restart False']
                        
                        tags += ['nimages '+str(images), 'fmax 0.05', 'latt-move-scale 0 0 0']
                        self.add_tags(neb_dir, tags)
                        new_roots.append(neb_dir)
                        
                        # add convergence file
                        if self.args.use_convergence == 'True':
                            if os.path.exists(os.path.join(inputs_folder, 'convergence')):
                                self.run('cp '+os.path.join(inputs_folder, 'convergence')+
                                     ' '+os.path.join(neb_dir, 'convergence'))
                            
                        
                        print('Setup NEB folder: '+neb_dir)

        return new_roots

    def make_calc(self, calc_folder, surf, root, v, bias, sd = True, sd_dist = 1.5):
        head_root = os.path.join(calc_folder, 'surfs', surf)
        if not os.path.exists(os.path.join(head_root, 'POSCAR')):
            print('POSCAR must be added to folder: '+head_root)
            return False
        else:
            if not os.path.exists(root):
                os.mkdir(root)
            self.run('cp '+os.path.join(head_root, 'POSCAR')+' '+os.path.join(root, 'POSCAR'))
        # check surface 
        if not h.check_surface(os.path.join(root, 'POSCAR')):
            return False
        
        # assign selective dynamics if requested
        if sd:
            st = Structure.from_file(opj(root, 'POSCAR'))
            new_st = assign_selective_dynamics(st, sd_dist)
            new_st.to('POSCAR', opj(root, 'POSCAR'))
        
        # copy inputs from inputs_folder and update based on tags
        self.run('cp '+os.path.join(inputs_folder, 'surfs_inputs')+' '+os.path.join(root, 'inputs'))
        
        # add convergence file
        if self.args.use_convergence == 'True':
            if os.path.exists(os.path.join(inputs_folder, 'convergence')):
                self.run('cp '+os.path.join(inputs_folder, 'convergence')+
                     ' '+os.path.join(root, 'convergence'))
            else:
                print('WARNING: "convergence" file not found in '+inputs_folder)
            
            if 'conv-tags' in v:
                self.set_conv_tags(root, v['conv-tags'])
        
        if 'tags' in v:
            tags = v['tags']
        else:
            tags = []
        tags += ['target-mu '+ ('None' if bias in ['None','none','No_bias'] 
                 else '%.4f' % self.get_mu(bias, h.read_inputs(root), tags))]
        self.add_tags(root, tags)
        
        good_setup = self.set_input_system_params(root, 'surfs')
        if not good_setup:
            print('Setup failed: '+calc_folder)
            return False
        return True
    
    def set_conv_tags(self, root, conv_tags, remove = False):        
        if not ope(opj(root, 'convergence')):
            return
        conv_dic = h.read_convergence(root)
#        print(conv_dic)
        
        new_conv_dic = {}
        # conv_tags is dic: {step: [list, of, cmd+val, strings]}
        for step, vals in conv_tags.items():
            for val_str in vals: 
                cmd = val_str.split()[0]
                val = ' '.join(val_str.split()[1:])
                if step not in new_conv_dic:
                    new_conv_dic[step] = {}
                if cmd in new_conv_dic[step]:
                    # multiple versions of same command (i.e. pdos)
                    if type(new_conv_dic[step][cmd]) == list:
                        new_conv_dic[step][cmd].append(val)
                    else:
                        new_conv_dic[step][cmd] = [new_conv_dic[step][cmd], val]
                else:
                    # single version of command
                    new_conv_dic[step][cmd] = val        
#        print(new_conv_dic)
        
        for step, vdic in new_conv_dic.items():
            for cmd, val in vdic.items():
                conv_dic[step][cmd] = val
        
        # remove steps if requested
        if remove != False:
            for r in remove: # list of steps to remove
                if r in conv_dic:
                    del conv_dic[r]
            
        h.write_convergence(root, conv_dic)

    def get_mol_loc(self, mol):
        if os.path.exists(os.path.join(molecule_folder, mol, 'POSCAR')):
            return os.path.join(molecule_folder, mol, 'POSCAR')
        elif os.path.exists(os.path.join(molecule_folder, mol, 'a', 'POSCAR')):
            return os.path.join(molecule_folder, mol, 'a', 'POSCAR')
        else:
            print('ERROR: Molecule not found: '+mol+'. Please add new folder with POSCAR to '+molecule_folder)
            return False
    
    def get_ref_mols(self, mol):
        refs = []
        for ref_mol in h.reference_molecules()[mol]['refs']:
            if h.reference_molecules()[ref_mol]['refs'] == [ref_mol]:
                refs += [ref_mol]
            else:
                refs += self.get_ref_mols(ref_mol)
        return refs

    def add_tags(self, root, tags):
        # inputs is a dictionary, tags is a list of lines to add
        inputs = h.read_inputs(root)
        inputs['restart'] = 'False'
        assert type(tags) == list, 'METAERROR: type for variable "tags" must be a list.'
        for tag in tags:
            tag_k = tag.split(' ')[0]
            tag_v = ' '.join(tag.split(' ')[1:])
            if tag_v in ['None']:
                if tag_k in inputs:
                    del inputs[tag_k]
            elif tag_v in ['pH','ph']:
                inputs['pH'] = tag_v
            else:
                inputs[tag_k] = tag_v
        h.write_inputs(inputs, root)
    
    def upgrade_calc(self, new_root, old_root, bias, tags, verbose = True, ):
        
        copy_electronic = True if self.args.copy_electronic == 'True' else False
        # upgrade from similar type of calc, 
        # do not need to update inputs of convergence aside from changing bias
        os.mkdir(new_root)
        self.run('cp ' + os.path.join(old_root, 'CONTCAR') + ' ' + os.path.join(new_root, 'POSCAR'))
        
        if copy_electronic:
            print('copying electronic data from '+old_root+' to new calc: '+new_root)
            for efile in ['wfns','fillings','eigenvals','fluidState']:
                if os.path.exists(os.path.join(old_root, efile)):
                    self.run('cp ' + os.path.join(old_root, efile) 
                             + ' ' + os.path.join(new_root, efile))
        
        # copy over convergence file
        if self.args.use_convergence == 'True' and os.path.exists(os.path.join(old_root, 'convergence')):
            self.run('cp ' + os.path.join(old_root, 'convergence') 
                     + ' ' + os.path.join(new_root, 'convergence'))
        
        inputs = h.read_inputs(old_root)
        if bias in ['No_bias']:
            if 'target-mu' in inputs:
                del inputs['target-mu']
        else:
            mu = self.get_mu(bias, inputs, tags)
            inputs['target-mu'] = '%.4f'%(mu)
        inputs['restart'] = 'False'
        h.write_inputs(inputs, new_root)
        if verbose: print('Upgraded '+old_root+' to '+new_root)

    def get_mu(self, bias, inputs, tags = []):
        if bias == 'None':
            return 'None'
        assert 'fluid' in inputs, 'ERROR: fluid tag must be in inputs files to run biases!'
        fluid = inputs['fluid'].replace(' ','')
        pcm_var = inputs['pcm-variant'].replace(' ','') if 'pcm-variant' in inputs else 'None'
        if fluid == 'LinearPCM' and pcm_var == 'CANDLE':
            Vref = 4.66
        elif fluid == 'LinearPCM' and pcm_var == 'GLSSA13':
            Vref = 4.68
        elif fluid == 'NonlinearPCM' and (pcm_var == 'GLSSA13'):
            Vref = 4.62
        elif fluid == 'SaLSA':
            Vref = 4.54
        elif fluid == 'ClassicalDFT':
            Vref = 4.44
        else:
            assert False, ('ERROR: Fluid model must be in [CANDLE, SaLSA, ClassicalDFT]. '+
                           'Other models not yet configured. ('+fluid+', '+pcm_var+')')
        rhe_shift = 0
        if self.args.rhe_zeroed == 'True':
            #JDFT uses SHE as zero point. 0V vs RHE === (-0.0591 * pH) V vs SHE
#            rhe_shift = -0.0591 * self.args.ph_rhe # input is RHE/V bias, output is SHE/JDFT/Hartree bias
            pH = 7.0
            for tag in tags:
                if 'pH' in tag or 'ph' in tag: 
                    pH = float(tag.split()[-1])
            rhe_shift = -0.0591 * pH #7 if 'pH' not in inputs else -0.0591 * float(inputs['pH'])
        return -(Vref + bias + rhe_shift)/27.2114 

    def read_manager_control(self, mc_text):
        '''
        Reads manager_control.txt file so that it can be used to setup and upgrade calculations
        '''
        managed = {'molecules': {}}
        ignore = True
        surf = None
        surf_bias = None
        mol = None
        error = False

        for line in mc_text.split('\n'):
            if '----- CALCULATIONS BELOW -----' in line:
                ignore = False
                continue
            if ignore or line == '':
                continue
            if line[0] == '#':
                continue
            
            if line[0] == '=':
                # new surface 
                surf = line[1:]
                managed[surf] = {}
                mol = None
                surf_bias = None
                mol_bias = None
                continue
            
            if line[0] == '-':
                # molecule for surface
                mol = line.split(':')[0][1:]
                # DONE: fix so this can read other types of inputs, even double inputs
#                sites = [int(x) for x in h.read_bias(line)]
                sites = h.get_sites(line)
                if surf is None:
                    print('Error in manager_control: surface must be listed before molecule '+mol)
                    error = True
                    continue
                managed[surf][mol] = {'sites': sites}
                managed['molecules'][mol] = {'biases': []}
                continue
            
            if line[0] == '+' and '[' not in line:
                # tag for inputs file
                if surf is None:
                    print('Error in manager_control: surface must be listed before adding "+" tags')
                    error = True
                    continue
                if mol is None:
                    if 'tags' not in managed[surf]:
                        managed[surf]['tags'] = []
                    managed[surf]['tags'].append(line[1:])
                else:
                    if 'tags' not in managed[surf][mol]:
                        managed[surf][mol]['tags'] = []
                    managed[surf][mol]['tags'].append(line[1:])
                continue
            
            if line[0] == '+' and '[' in line and ']' in line:
                # tags for convergence file
                if surf is None:
                    print('Error in manager_control: surface must be listed before adding "+" tags')
                    error = True
                    continue
                step_number = line.split('[')[1][0]
                val = line.split(']')[1]
                if val[0] == ' ':
                    val = val[1:]
                if mol is None:
                    if 'conv-tags' not in managed[surf]:
                        managed[surf]['conv-tags'] = {}
                    if step_number not in managed[surf]['conv-tags']:
                        managed[surf]['conv-tags'][step_number] = []
                    managed[surf]['conv-tags'][step_number].append(val)
                else:
                    if 'conv-tags' not in managed[surf][mol]:
                        managed[surf][mol]['conv-tags'] = {}
                    if step_number not in managed[surf][mol]['conv-tags']:
                            managed[surf][mol]['conv-tags'][step_number] = []
                    managed[surf][mol]['conv-tags'][step_number].append(val)
                continue
                
            if 'Biases:' in line:
                if surf is None:
                    print('Error in manager_control: surface must be listed before biases using "=".')
                    error = True
                    continue
                if mol is None:
                    # biases for surface
                    surf_bias = h.read_bias(line)
                    managed[surf]['biases'] = surf_bias
                else:
                    # biases for molecule
                    mol_bias = h.read_bias(line)
                    if surf_bias is None:
                        print("Error in manager_control: surface "+surf+" has no biases listed before mol "+mol)
                        error = True
                        continue
                    if any([b not in surf_bias for b in mol_bias]):
                        print("Error in manager_control: biases for mol "+mol+" not all in surface "+surf+" biases")
                        error = True
                        continue
                    managed[surf][mol]['biases'] = mol_bias
                    managed['molecules'][mol]['biases'] += mol_bias
                continue
            
            if 'Desorb:' in line:
                # use this command to setup desorbed calculations
                if surf is None:
                    print('Error in manager_control: surface must be listed before Desorb')
                    error = True
                    continue
                if mol is None:
                    print("Error in manager_control: molecule must be listed before Desorb")
                    error = True
                    continue
                if mol_bias is None:
                    print("Error in manager_control: biases for mol "+mol+" must be listed before desorb")
                    error = True
                    continue
                desorb_bias = h.read_bias(line)
                if any([b not in mol_bias for b in desorb_bias]):
                    print("Error in manager_control: desorb biases for mol "+mol+" must be in mol biases")
                    error = True
                    continue
                managed[surf][mol]['desorb'] = desorb_bias
                managed['molecules'][mol]['desorb'] = True
                continue
            
            if 'Dist:' in line:
                # use this command to assign adsorbate/desorbed distance from surf
                if surf is None:
                    print('Error in manager_control: surface must be listed before Dist command')
                    error = True
                    continue
                if mol is None:
                    print("Error in manager_control: molecule must be listed before Dist command")
                    error = True
                    continue
                mol_dist = float(line.split(':')[-1])
                if 'desorb' in managed[surf][mol]:
                    managed[surf][mol]['desorb_dist'] = mol_dist
                else:
                    managed[surf][mol]['ads_dist'] = mol_dist
                continue
            
            if 'NEB:' in line:
                # within surf, format: 
                #       NEB: path/to/init path/to/final nimages name [biases, to, run] 
                #       path should include 'BIAS' in place of biases 
                if surf is None: # or surf_bias is None or mol is None or mol_bias is None or desorb_bias is None:
                    print("Error in manager_control: NEB tag must be nested under a surface")
                    error = True
                    continue
                try:
                    init = line.split()[1]
                    final = line.split()[2]
                    images = int(line.split()[3])
                    path_name = line.split()[4]
                    biases = h.read_bias(line)
                except:
                    print("Error in manager_control: NEB line entered incorrectly: "+line)
                    error = True
                    continue
                if 'NEB' not in managed[surf]:
                    managed[surf]['NEB'] = []
                
                managed[surf]['NEB'].append({'init': init, 'final': final,
                       'biases': biases, 'images': images, 'path_name': path_name})
                continue
            
        if error:
            print('\nPlease fix errors in manager_control.txt and rerun\n')
            return False
        return managed

    def add_calc_inputs(self, folders):
        # setup calcs NOT IN MANAGER_CONTROL.TXT (cannot use values from this)
        properly_setup = []
        for root in folders:
            # get type of calculation
            calc_type = None
            for subf in self.calc_subfolders:
                tag = os.path.join(calc_folder, subf)
                if tag in root:
                    calc_type = subf
                    continue
            if calc_type is None:
                print('Error: No calc_type found for root: '+root+' Skipping.')
                continue
            inputs_file = calc_type + '_inputs'
            self.run('cp '+os.path.join(inputs_folder, inputs_file)+' '+os.path.join(root, 'inputs'))
            
            if self.args.use_convergence == 'True' and calc_type != 'molecules':
                if os.path.exists(os.path.join(inputs_folder, 'convergence')):
                    self.run('cp '+os.path.join(inputs_folder, 'convergence')+
                         ' '+os.path.join(root, 'convergence'))
                else:
                    print('WARNING: "convergence" file not found in '+inputs_folder)
            
            good_setup = self.set_input_system_params(root, calc_type)
            if not good_setup:
                continue
            
            # add bias tag to inputs if needed
            if calc_type in ['surfs','molecules','desorbed','neb']: # bias is last listed
                bias_tag = root.split(os.sep)[-1]
            elif calc_type in ['adsorbed']:
                bias_tag = root.split(os.sep)[-2]
            elif calc_type == 'bulks':
                properly_setup.append(root)
                continue
            if bias_tag == 'No_bias':
                properly_setup.append(root)
                continue
            bias = h.get_bias(bias_tag)
            tags = ['target-mu %.4f' % self.get_mu(bias, h.read_inputs(root))]
            self.add_tags(root, tags)
            properly_setup.append(root)
        return properly_setup
    
    def set_input_system_params(self, root, calc_type): # kpoint_density = 1000):
        kpoint_density = self.args.kpoint_density
        kpoint_density_bulk = self.args.kpoint_density_bulk
        # called when setting up new calcs from scratch or setting up calcs from manager_control
        st = Structure.from_file(opj(root, 'POSCAR'))
        tags = h.read_inputs(root)
        
        if tags['kpoint-folding'] == '*':
            # set kpoint-folding for bulk systems
            if calc_type in ['bulks',]: #'molecules'
                kpts = Kpoints.automatic_density(st, kpoint_density_bulk).as_dict()
                tags['kpoint-folding'] = ' '.join([str(k) for k in kpts['kpoints'][0]])
                
                if calc_type=='bulks' and self.args.use_convergence == 'True' and ope(opj(root,'convergence')):
                    # update bulk convergence to not use fmax tags and remove SP step 3
                    conv = h.read_convergence(root)
                    steps = list(conv.keys())
                    self.set_conv_tags(root, {s: ['fmax 0.00'] for s in steps[:-1]}, remove=[steps[-1]])
                    print('Convergence file updated for bulk: '+root)
            
            elif calc_type in ['molecules']:
                tags['kpoint-folding'] = '1 1 1'
            
            # set kpoints for surfs from bulk calcs
            elif calc_type in ['surfs']:
#                try: # try to get bulk inputs from __all_surfs folder (from bulk)
#                    bulk_tags = h.read_inputs(opj(os.sep.join(root.split(os.sep)[:-1]),
#                                                  '__all_surfs'),file='bulk_inputs')
#                    kpts = bulk_tags['kpoint-folding']
#                    tags['kpoint-folding'] = ' '.join(kpts.split()[0:2] + ['1'])
#                except:
                try:
                    kpts = Kpoints.automatic_density(st, kpoint_density).as_dict()
                    tags['kpoint-folding'] = ' '.join([str(k) for k in kpts['kpoints'][0]])
                    print('kpoints auto-generated for surface '+root.split(os.sep)[2]
                          +' with density '+str(kpoint_density))
                except:
                    print('ERROR: kpoint-folding cannot be set for: '+root)
                    return False
            
            elif calc_type in ['adsorbed','desorbed']:
                try: # copy kpoint-folding from surfs (no longer bulk from surfs)
#                    surf_subfolder = opj(calc_folder, 'surfs', root.split(os.sep)[2], '__all_surfs')
                    surf_subfolder = opj(calc_folder, 'surfs', root.split(os.sep)[2], root.split(os.sep)[4])
                    surf_tags = h.read_inputs(surf_subfolder, file='inputs') #bulk_inputs
                    kpts = surf_tags['kpoint-folding']
                    tags['kpoint-folding'] = kpts #' '.join(kpts.split()[0:2] + ['1'])
                except:
                    print('ERROR: kpoint-folding cannot be set for: '+root)
                    return False
            
            else:
                print('ERROR: kpoint-folding cannot be set for: '+root)
                return False
            
        # set elec-n-bands for all systems 
        tags['elec-n-bands'] = str(self.set_elec_n_bands(root))
        
        h.write_inputs(tags, root)
        return True
    
    def set_elec_n_bands(self, root, band_scaling = 1.2):
        st = Structure.from_file(opj(root, 'POSCAR'))
        tags = h.read_inputs(root)
        
        psdir = os.environ['JDFTx_pseudos']
        ps_type = tags['pseudos'] if 'pseudos' in tags else 'GBRV'
        ps_key = opj(psdir, self.pseudoMap[ps_type])
        els = [s.species_string for s in st.sites]
        el_dic = {el: els.count(el) for el in els}
        # get electrons from psd files
        nelec = 0
        for el, count in el_dic.items():
            file = ps_key.replace('$ID', el if ps_type in ['dojo'] else el.lower())
            with open(file, 'r') as f:
                ps_txt = f.read()
            zval = [line for line in ps_txt.split('\n') if 'Z valence' in line # GBRV
                    or 'z_valence' in line][0] # SG15
            electrons = int(float(zval.split()[0])) if 'Z' in zval else (
                        int(float(zval.split()[1].replace('"',''))))
            nelec += electrons * count
        nbands_add = int(nelec / 2) + 10
        nbands_mult = int((nelec / 2) * band_scaling)
        return max([nbands_add, nbands_mult])

    def get_running_jobs_dirs(self):
        p = subprocess.Popen(['squeue' ,'-o', '"%Z %T"'],   #can be user specific, add -u username 
                         stdout=subprocess.PIPE)
        jobs_running = []
        for i,line in enumerate(p.stdout):
            if i == 0:
                continue
            jobs_running.append(str(line, 'utf-8').replace('"', '').split()[0])
        return jobs_running

    def run_new_calcs(self, new_calcs):
        print('\n----- Running new calcs -----\n')
        for ii,root in enumerate(new_calcs):
            if ii%100 == 99:
                print('\n...\n')
                sleep(5) # pause for 5 seconds before submitting more jobs
            calc_type = root.split(os.sep)[1]
            os.chdir(root)
#            print('Rerunning: '+self.get_job_name(root))
            if self.args.smart_procs == 'True':
#                print('debug: smart_procs:',self.args.smart_procs, 'calc type:', calc_type)
                if calc_type in ['adsorbed','surfs','desorbed','neb']:
#                    print('-procs 2')
#                if calc_type in ['adsorbed','surf','desorbed','neb']:
                    self.run(self.run_cmd + ' -o '+self.get_job_name(root)+ ' -m 2') 
                else:
                    self.run(self.run_cmd + ' -o '+self.get_job_name(root)+ ' -m 8') 
            else:
                self.run(self.run_cmd + ' -o '+self.get_job_name(root))
            os.chdir(self.cwd)
            print('Calculation run: '+self.get_job_name(root))
    
    def get_parallel_running(self): # TODO: finish this so that -p jobs don't run over eachother
        shell_folder = 'tmp_parallel'
        if not os.path.exists(shell_folder):
            return []
        if 'running.txt' not in os.listdir(shell_folder):
            return []

    def run_all_parallel(self, roots, bundle = False):
        '''
        This function handles submitting clusters of calculations together on single nodes
        '''
        print('\n----- Running All Calcs in Parallel -----\n')
        
        max_per_node = self.args.parallel
        total_calcs = len(roots) 
        if total_calcs < max_per_node:
            max_per_node = total_calcs
        cores_per_node = core_architecture
        total_nodes = int(np.ceil(total_calcs / max_per_node))
        cores_per_calc = int(np.floor(cores_per_node / max_per_node))
        
        if bundle:
            total_nodes = total_calcs
            max_per_node = total_calcs
            cores_per_calc = cores_per_node
        
        shells = []
        shell_folder = 'tmp_parallel'
        if os.path.exists(shell_folder):
            for shell_root, dirs, files in os.walk(shell_folder):
                for name in files:
                    os.remove(os.path.join(shell_root, name))
        else:
            os.mkdir(shell_folder)
        
        if bundle:
#            out_file = 'submit_bundle'
#            write_parallel_bundle(roots, self.cwd, total_nodes, cores_per_node, 
#                                  self.args.run_time, out_file, shell_folder, 
#                                  qos = 'standard', gpu = self.args.gpu)
#            shells.append(out_file + '.sh')
            
            sub_parallel(roots, self.cwd, self.args.nodes, os.environ['CORES_PER_NODE'],
                         self.args.run_time)
            return
        
        else:
            # write calcs for each node to tmp_parallel folder
            for i in range(total_nodes):
                # TODO: set cores per calc to update based on number of calcs in sub_roots
                sub_roots = roots[i*max_per_node:(i+1)*max_per_node]
                out_file = 'submit_'+str(i)
                write_parallel(sub_roots, self.cwd, cores_per_node, cores_per_calc, self.args.run_time, out_file, 
                               shell_folder, self.args.qos)
                shells.append(out_file + '.sh')
        
        # submit shell scripts
        os.chdir(shell_folder)
        for shell in shells:
            os.system('sbatch '+shell)
        os.chdir(self.cwd)
    
    def backup_calcs(self, convert_contcar = False):
        if not os.path.exists(backup_folder):
            os.mkdir(backup_folder)
        # scan all sub_dirs in calcs and copy to backup_folder   
        for root, folders, files in os.walk(calc_folder):
            # copy folder over
            for i, sub in enumerate(root.split(os.sep)):
                if i == 0: continue
                sub_folder = os.sep.join(root.split(os.sep)[1:i+1])
                backup_f = os.path.join(backup_folder, sub_folder)
                if not os.path.exists(backup_f):
                    os.mkdir(backup_f)
            # copy over files
            for file in files_to_backup:
                if file in files:
                    if convert_contcar and file == 'CONTCAR':
                        self.run('cp '+os.path.join(root, file)+' '+os.path.join(backup_f, 'POSCAR'))
                        continue
                    if convert_contcar and file == 'POSCAR': continue
                    self.run('cp '+os.path.join(root, file)+' '+os.path.join(backup_f, file))
        print('\nCalculation files backed up successfully.')
    
    def remove_wfns(self, converged):
        sleep(5)
        cwd = os.getcwd()
        for root in converged:
            os.chdir(root)
            files = os.listdir()
            if 'wfns' in files:
                print('Removing wfns: '+root)
                self.run('rm wfns')
            os.chdir(cwd)

    def manager(self):
        '''
        Main class function for jdft_manager. Linearly runs the main sub-functions of jdft_manager 
        based on user inputs from command line. Main sub-functions also have descriptions.
        
        '''
        # ensure subfolders are correctly setup
        if self.args.setup == 'True':
            return
        assert self.proper_setup, ('ERROR: jdft_manager not yet setup! Please run with '+
                                   '-h to check input parameters or -s to setup folder.')
        
        # read through current results data 
        all_data = {}
        if os.path.isfile(self.data_file) and self.args.read_all != 'True':
            with open(self.data_file, 'r') as f:
                all_data = json.load(f)
                
        # get parallel tag
        parallel = self.args.parallel
        bundle = True if self.args.bundle_jobs == 'True' else False
        
        # scan through all subfolders to check for converged structures 
        add_inputs = []
        ncalcs = None
        if self.args.check_calcs == 'True':
            # scan through folders
            running_jobs_dirs = self.get_running_jobs_dirs()
            all_data, add_inputs, rerun, run_new, failed_calcs, ncalcs  = self.scan_calcs(all_data, 
                                                                                          running_jobs_dirs)
            # save all data 
            if self.args.save == 'True':
                # run analysis of converged calcs
                if self.args.analyze == 'True':
                    print('\n----- Running Calculation Analysis -----')
                    if len(all_data.keys()) < 2:
                        print('No data available yet!')
                    else: # TODO: make this work 
                        h.analyze_data(all_data)
                # save 
#                if len(all_data.keys()) < 2:
                print('\nSaving converged calculations.')
                with open(self.data_file, 'w') as f:
                    json.dump(all_data, f)
        
            # save and rerun unconverged (if requested)
            with open(os.path.join(results_folder, 'unconverged.txt'), 'w') as f:
                f.write('\n'.join(rerun))
            with open(os.path.join(results_folder, 'failed.txt'), 'w') as f:
                f.write('\n'.join(failed_calcs))
            if self.args.rerun_unconverged == 'True' and len(rerun) > 0: #self.args.check_calcs == 'True' and 
#                print('\nRerunning unconverged calculations')
                self.update_rerun(rerun)
                if parallel == 1 and not bundle:
                    self.rerun_calcs(rerun)
                    print('\n',len(rerun),'calcs rerun.')
        
        # make new surfaces based on manager_control.txt file, add new calcs to add_inputs
        new_folders = []
        inputs_added = []
        if self.args.make_new == 'True':
            new_folders = self.make_new_calcs(all_data['converged'])
            if new_folders == False:
                print('Exiting.\n\n')
                return
            
            # add inputs to any created file with a POSCAR and no 'inputs'. 
            # Does not need to be in manager_control.txt
            if len(add_inputs) > 0:
                inputs_added = self.add_calc_inputs(add_inputs)
        
        # run jobs with added 'inputs'
        # molecules should only be run if requested or in manager_control
        # neb calcs should be handled differently ? submission is the same, inputs is all that's changed.
        #   sub_folders should start with single point calcs after setup! 
        # bias calcs should pass structure forward from nomu -> 0V -> other biases
        if self.args.run_new == 'True' and len(new_folders + inputs_added + run_new) > 0:
            if len(run_new) > 0:
                h.update_run_new(run_new)
            if parallel == 1 and not bundle:
                self.run_new_calcs(new_folders + inputs_added + run_new)
                print('\n',len(new_folders + inputs_added + run_new),'new calcs run.')
        
        # run calcs in parallel 
        all_roots = []
        if self.args.run_new == 'True':
            all_roots += new_folders + inputs_added + run_new
        if self.args.rerun_unconverged == 'True':
            all_roots += rerun
        if parallel > 1 or bundle:
            self.run_all_parallel(all_roots, bundle = bundle)
        
        # backup calcs if requested
        if self.args.backup == 'True':
            print('\n\nBacking up main files (this may take awhile)')
            self.backup_calcs()
        
        if self.args.clean_wfns == 'True':
            print('\nRemoving wfns from converged directories (5s pause to cancel)')
            self.remove_wfns(all_data['converged'])
            
        print('----- Done -----')
        if ncalcs is not None:
            print('Number of Calcs Under Management: '+str(ncalcs)+'\n\n')



# set to False for debugging 
run_script = True    

# set computer defaults
#try:
#    home_dir = os.environ['JDFTx_home']
#except:
#    assert False, "Error: Environment variable 'JDFTx_home' must be set and point to jdftx location"
try:
    core_architecture = int(os.environ['CORES_PER_NODE'])
except:
    print("Warning: 'CORES_PER_NODE' not found, cores per node set to 36.")
    core_architecture = 36
    
#defaults_folder = os.path.join(home_dir, 'bin/JDFTx_Tools/manager/defaults/')
#run_command = 'python '+ os.path.join(home_dir, 'bin/JDFTx_Tools/sub_JDFTx.py')
manager_home = os.environ['JDFTx_manager_home']
defaults_folder = os.path.join(manager_home, 'defaults')
run_command = 'python '+ os.path.join(manager_home, 'sub_JDFTx.py')

# all files to include in backup
files_to_backup = ['CONTCAR','POSCAR','inputs','convergence','opt.log','neb.log','Ecomponents',
                   'tinyout', 
                   ]    
    
if __name__ == '__main__' and run_script:
    jm = jdft_manager()
    jm.manager()