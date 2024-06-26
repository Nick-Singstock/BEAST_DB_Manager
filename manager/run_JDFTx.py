#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Musgrave Group. Functions to run JDFTx via python.

@author: zaba1157, nisi6161
"""

from JDFTx import JDFTx
import os
from ase.io import read
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin, FIRE
from ase.constraints import FixBondLength
from ase.io.trajectory import Trajectory
from ase.neb import NEB
from ase import Atoms
import argparse
import subprocess
from ase.optimize.minimahopping import MinimaHopping
import numpy as np
from jdft_helper import helper 
from suggest_input_tags import set_elec_n_bands
import re
h = helper()

opj = os.path.join
ope = os.path.exists
hartree_to_ev = 27.2114

'''
TODO:
    auto kpoints (bulk + surfs)
    auto nbands
    remove 0V surf calc dependency
    remove any bias dependency for adsorbate calcs 
'''

try:
    comp=os.environ['JDFTx_Computer']
except:
    comp='Eagle'

def conv_logger(txt, out_file = 'calc.log'):
    if os.path.exists(out_file):
        with open(out_file,'r') as f:
            old = f.read()
    else:
        old = ''
    with open(out_file,'w') as f:
        f.write(old + txt + '\n')

def sp_conv_logger(txt, out_file = 'sp_calc.log'):
    if os.path.exists(out_file):
        with open(out_file,'r') as f:
            old = f.read()
    else:
        old = ''
    with open(out_file,'w') as f:
        f.write(old + txt + '\n')

# ensure forces don't get too large or stop job
def force_checker(max_force = 500):
    if 'opt.log' in os.listdir():
        try:
            with open('opt.log', 'r') as f:
                opt_text = f.read()
        except:
            print('Unable to read opt.log for force checker.')
    elif 'neb.log' in os.listdir():
        try:
            with open('neb.log', 'r') as f:
                opt_text = f.read()
        except:
            print('Unable to read neb.log for force checker.')
    else:
        return
    if len(opt_text) == 0: return 
    
    opt = [line.split() for line in opt_text.split('\n') if line != '' and '*Force-consistent' not in line]
    try:
        force = float(opt[-1][4])
    except:
        conv_logger('Forces not read correctly from line: '+str(opt))
        return
    assert force < max_force, 'ERROR: Calculation ended due to high forces. Edit Structure.'

# add element to CONTCAR to fix bug
def insert_el(filename):
    """
    Inserts elements line in correct position for Vasp 5? Good for
    nebmovie.pl script in VTST-tools package
    Args:
        filename: name of file to add elements line
    """

    with open(filename, 'r') as f:
        file = f.read()
    contents = file.split('\n')
    ele_line = contents[0]
    if contents[5].split() != ele_line.split():
        contents.insert(5, ele_line)
    with open(filename, 'w') as f:
        f.write('\n'.join(contents))

# adds inputs_dos file to cmds
def add_dos(cmds, script_cmds):
    from pymatgen.core.structure import Structure
    if not ope('./inputs_dos') and not ('pdos' in script_cmds):
        return cmds
    new_cmds = []
    for cmd in cmds:
        if 'density-of-states' in cmd:
            print('WARNING: command density-of-states in inputs file is being overwritten by inputs_dos.')
            continue
        new_cmds += [cmd]
        
    dos_line = '' 
    st = Structure.from_file('./POSCAR')
    
    if ope('./inputs_dos'):
        with open('./inputs_dos','r') as f:
            dos = f.read()
       
        for line in dos.split('\n'):
            if 'Orbital' in line and len(line.split()) >= 3:
                # Format: Orbital Atom_type orbital_type(s, spaced)
                # Adds DOS for ALL atoms of this type and all requested orbitals
                atom_type = line.split()[1]
                indices = [i for i,el in enumerate(st.species) if str(el) == atom_type]
                orbitals = line.split()[2:]
                assert all([orb in ['s','p','px','py','pz','d','dxy','dxz','dyz','dz2','dx2-y2','f'] 
                            for orb in orbitals]), ('ERROR: Not all orbital types allowed! ('+', '.join(orbitals)+')')
                for i in range(len(indices)):
                    for orbital in orbitals:
                        dos_line += ' \\\nOrbital ' +atom_type + ' ' + str(i+1) + ' ' + orbital 
            else:
                dos_line += ' \\\n' + line
    
    # allow pdos line in script_cmds
    if 'pdos' in script_cmds:
        conv_logger('pdos found in script cmds')
        if type(script_cmds['pdos']) == str:
            script_cmds['pdos'] = [script_cmds['pdos']]
        for pdos in script_cmds['pdos']:
            conv_logger('pdos '+str(pdos))
            if len(pdos.split()) > 1:
                # Format (list): Atom_type orbital_type(s, spaced)
                # Adds DOS for ALL atoms of this type and all requested orbitals
                atom_type = pdos.split(' ')[0]
                indices = [i for i,el in enumerate(st.species) if str(el) == atom_type]
                orbitals = pdos.split(' ')[1:]
                assert all([orb in ['s','p','px','py','pz','d','dxy','dxz','dyz','dz2','dx2-y2','f'] 
                            for orb in orbitals]), ('ERROR: Not all orbital types allowed! ('+', '.join(orbitals)+')')
                for i in range(len(indices)):
                    for orbital in orbitals:
                        dos_line += ' \\\nOrbital ' +atom_type + ' ' + str(i+1) + ' ' + orbital 
            else:
                dos_line += ' \\\n' + pdos
    
    new_cmds += [('density-of-states', dos_line)]
    return new_cmds

def autodos_sp(cmds, atoms):
    els = list(set(atoms.get_chemical_symbols()))
    doskeys = {'s': ['s'], 'p': ['p','px','py','pz'], 'd': ['d','dxy','dxz','dyz','dz2','dx2-y2']}
    alldos = {'s': ['H',  'Li','Be',   'B','C','N','O','F',
                    'Na','Mg',  'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',  'Al','Si','P','S','Cl',
                    'K','Ca',   'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',  'Ga','Ge','As','Se','Br',
                    'Rb','Sr',  'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg'        'In','Sn','Sb','Te','I'],
              'p': ['B','C','N','O','F',        'Al','Si','P','S','Cl',
                    'Ga','Ge','As','Se','Br',   'In','Sn','Sb','Te','I', 'Tl','Pb'],
              'd': ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
                    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg']}
    new_dos = []
    for el in els:
        el_dos = [el]
        for orb in ['s','p','d']:
            orb_els = alldos[orb]
            if el in orb_els:
                el_dos += doskeys[orb]
        if len(el_dos) < 2:
            continue # no dos orbitals for this atom
        dosstr = ' '.join(el_dos)
        new_dos.append(dosstr)
    new_dos.append('Total')
    dos_cmds = {'pdos': new_dos}
    newcmds = add_dos(cmds, dos_cmds)
    return newcmds

def clean_doscmds(cmds):
    new_cmds = []
    for cmd in cmds:
        # remove repeat pdos cmds 
        if cmd[0] == 'density-of-states':
            dosline = cmd[1]
            pdos = dosline.split(' \\\n')
            new_pdos = []
            for p in pdos:
                if p not in new_pdos:
                    new_pdos.append(p)
            conv_logger('clean pdos: ' + str(new_pdos))
            cmd = (cmd[0], ' \\\n'.join(new_pdos))
        
        if cmd not in new_cmds:
            new_cmds.append(cmd)
    conv_logger('clean cmds: '+str(new_cmds))
    return new_cmds

def keys_from_commands(cmds):
    # cmds is a list of tuples. The first item in each tuple is the command key
    # and the second item is the value associated with it.
    # this function returns a list of the command keys
    keys = []
    for cmd in cmds:
        keys.append(cmd[0])
    return keys

######## Functions for setting up ASE calculator ########
def open_inputs(inputs_file):
    with open(inputs_file,'r') as f:
        inputs = f.read()
    return inputs

def read_line(line, notinclude):
    line = line.strip()
    if len(line) == 0 or line[0] == '#': 
        return 0,0,'None'
    linelist = line.split()
    
    defaults = ['default','inputs']
    inputs = h.read_inputs('./')
    
    if linelist[0] in notinclude:
        # command, value, type
        cmd, val, typ = linelist[0], ' '.join(linelist[1:]), 'script'
        if val in defaults:
            val = inputs[cmd]
        return cmd, val, typ
    else:
        if len(linelist) > 1:
            # command, value, type
            cmd, val, typ = linelist[0], ' '.join(linelist[1:]), 'cmd'
            if val in defaults:
                val = inputs[cmd]
            return cmd, val, typ
        else:
            return line, '', 'cmd'
    

def read_commands(inputs, notinclude):
    cmds = []
    script_cmds = {}
#        with open(command_file,'r') as f:
    for line in inputs.split('\n'):
#            conv_logger('Line = ', line)
        cmd, val, typ = read_line(line, notinclude)
        if typ == 'None':
            continue
        elif typ == 'script':
            if cmd in ['pdos','pDOS']:
                if 'pdos' not in script_cmds:
                    script_cmds['pdos'] = []
                script_cmds['pdos'].append(val)
                continue
            script_cmds[cmd] = val
        elif typ == 'cmd':
            tpl = (cmd, val)
            cmds.append(tpl)

#        cmds += [('core-overlap-check', 'none')]
    cmds = add_dos(cmds, script_cmds)
    return cmds, script_cmds



def calc_type(cmds, script_cmds):
#        conv_logger('calc_type debug: '+str(cmds))
    if 'nimages' in script_cmds.keys():
        calc = 'neb'
    elif 'optimizer' in script_cmds and script_cmds['optimizer'] in ['MD','md']:
        calc = 'md'
    elif 'opt' in script_cmds and script_cmds['opt'] in ['MD','md']:
        calc = 'md'
    elif any([('lattice-minimize' == c[0] and 'nIterations' in c[1] and 
                int(c[1].split()[1]) > 0 ) for c in cmds]):
        calc = 'lattice'
    else:
        calc = 'opt'
    return calc

def get_exe_cmd(script_cmds, interactive=False, nprocs = False):
    if comp == 'Eagle':
        exe_cmd = 'mpirun --bind-to none '+jdftx_exe
    elif comp in ['Cori',]:
        exe_cmd = 'srun --cpu-bind=cores -c 8 '+jdftx_exe
        conv_logger('Running on Cori with srun.')
    elif comp in ['Perlmutter']:
        exe_cmd = 'srun '+jdftx_exe
        conv_logger("Running on Perl using exe cmd {exe_cmd}".format(exe_cmd=exe_cmd))
    else:
        if nprocs != False: # read np from n-kpts
            jdftx_num_procs = nprocs
        else:
            jdftx_num_procs = os.environ['JDFTx_NUM_PROCS']
        
        if 'np' in script_cmds:
            jdftx_num_procs = script_cmds['np']
        
        exe_cmd = 'mpirun -np '+str(jdftx_num_procs)+' '+jdftx_exe
        conv_logger('exe_cmd: ' + exe_cmd)
    
    if interactive:
        print('Running JDFTx on an interactive node via sub_JDFTx.py')
        exe_cmd = 'srun -n 2 -c 16 --hint=nomultithread '+jdftx_exe #+' -i in -o out'
        
    return exe_cmd

     
def read_atoms(restart, script_cmds):
    if restart:
        atoms = read('CONTCAR',format='vasp')
    else:
        atoms = read('POSCAR',format='vasp')
    
    if 'lattice-type' in script_cmds and script_cmds['lattice-type'] in ['bulk','periodic','Bulk','Periodic']:
        return atoms
    elif 'lattice-type' in script_cmds and script_cmds['lattice-type'] in ['slab','Slab','surf','Surf']:
        atoms.set_pbc([True, True, False])
    elif 'lattice-type' in script_cmds and script_cmds['lattice-type'] in ['mol','Mol','isolated']:
        atoms.set_pbc([False, False, False])
    
    # default: periodic
#        else: 
#            atoms.set_pbc([True, True, False])
    return atoms

def get_nprocs(cmds):
    for cmd in cmds:
        if cmd[0] == 'kpoint-folding':
            kpts = [int(kpt) for kpt in cmd[1].split()]
            return np.product(kpts) * 2 # this *2 assumes spin-polarized calcs. May break for spin-paired?
    return False # no kpts tag found

def set_calc(cmds, script_cmds, outfile = os.getcwd(), jdftx_ionic = False, interactive=False):
    psuedos = script_cmds['pseudos']
    nprocs = get_nprocs(cmds)
    conv_logger('nprocs: '+str(nprocs))
    
    ionic_data = [ # steps, econv, 
        int(script_cmds['jdftx_steps']) if 'jdftx_steps' in script_cmds else 5,
        (float(script_cmds['econv']) if 'econv' in script_cmds else 1E-4) / hartree_to_ev,
        ]
    
    return JDFTx(
        executable = get_exe_cmd(script_cmds, nprocs, interactive),
        pseudoSet=psuedos,
        commands=cmds,
        outfile = outfile,
        ionic_steps = False if jdftx_ionic is False else ionic_data  
        )

def optimizer(imag_atoms,script_cmds,logfile='opt.log'):
    """
    ASE Optimizers:
        BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin and FIRE.
    """
    use_hessian = True if ('hessian' in script_cmds and script_cmds['hessian'] == 'True') else False
    opt_alpha = 150 if 'opt-alpha' not in script_cmds else int(script_cmds['opt-alpha'])
    if 'optimizer' in script_cmds:
        opt = script_cmds['optimizer']
    elif 'opt' in script_cmds:
        opt = script_cmds['opt']
    else:
        opt='FIRE'
    
    opt_dict = {'BFGS':BFGS, 'BFGSLineSearch':BFGSLineSearch,
                'LBFGS':LBFGS, 'LBFGSLineSearch':LBFGSLineSearch,
                'GPMin':GPMin, 'MDMin':MDMin, 'FIRE':FIRE}
    if opt in ['BFGS','LBFGS']:
        if use_hessian:
            dyn = opt_dict[opt](imag_atoms,logfile=logfile,restart='hessian.pckl',alpha=opt_alpha)
        else:
            dyn = opt_dict[opt](imag_atoms,logfile=logfile,alpha=opt_alpha)
    elif opt == 'FIRE':
        if use_hessian:
            dyn = opt_dict[opt](imag_atoms,logfile=logfile,restart='hessian.pckl',a=(opt_alpha/70) * 0.1)
        else:
            dyn = opt_dict[opt](imag_atoms,logfile=logfile,a=(opt_alpha/70) * 0.1)
    else:
        if use_hessian:
            dyn = opt_dict[opt](imag_atoms,logfile=logfile,restart='hessian.pckl')
        else:
            dyn = opt_dict[opt](imag_atoms,logfile=logfile)
    return dyn

def write_singlepoint_converegence(cmds, script_cmds):
    inputs_dict = cmds.update(script_cmds)
    inputs_str = h.inputs_to_string(inputs_dict)
    with open("singlepoint_convergence", "w") as f:
        f.write(inputs_str)

def update_commands_from_inputs_dict(cmds, script_cmds, inputs_dict):
    for updated_tag, updated_val in inputs_dict.items():
        for i, (cmd, val) in enumerate(cmds):
            if cmd == updated_tag and cmd != 'dump':
                cmds[i] = (str(updated_tag), str(updated_val))
            else: continue
    return cmds

#######################################################


########################################################################################################################
################################## Single point run function ###########################################################

def run_singlepoint(jdftx_exe, interactive=False):
    # this function runs a JDFTx singlepoint. It will not update any ASE files. It just dumps JDFTx outputs only.

    notinclude = ['ion-species','ionic-minimize',
                #'latt-scale','latt-move-scale','coulomb-interaction','coords-type',
                'ion','climbing','pH','ph',  'autodos',
                'logfile','pseudos','nimages','max_steps','max-steps','fmax','optimizer','opt',
                'restart','parallel','safe-mode','hessian', 'step', 'Step',
                'opt-alpha', 'md-steps', 'econv', 'pdos', 'pDOS', 'lattice-type', 'np',
                'use_jdftx_ionic', 'jdftx_steps', #Tags below here were added by Cooper
                'bond-fix', 'bader']

    inputs = open_inputs('singlepoint_inputs')
    inputs_dict = h.read_inputs('./', 'singlepoint_inputs')
    cmds, script_cmds = read_commands(inputs, notinclude)
    pseudos = script_cmds['pseudos']
    nbands, kpt_str = set_elec_n_bands('.','CONTCAR', pseudos, band_scaling=1.2, kpoint_density=1000)
    sp_conv_logger('nbands: '+str(nbands)+'\n')
    path = os.getcwd()
    bias = h.bias_from_path(path)
    sp_conv_logger('bias: '+str(bias)+'\n')
    print(cmds)
    if bias != None:
        bias_float = h.bias_str_to_float(bias)
        inputs_dict['target_mu'] = str(h.bias_to_mu(bias_float))
        sp_conv_logger('target-mu: '+str(h.bias_to_mu(bias_float))+'\n')
        cmds.append(('target-mu', str(h.bias_to_mu(bias_float))))
    elif bias == None:
        if 'target_mu' in keys_from_commands(cmds):
            inputs_dict.pop('target_mu')
    inputs_dict['elec-n-bands'] = nbands
    inputs_dict['kpoint-folding'] = kpt_str
    # update inputs commands
    cmds = update_commands_from_inputs_dict(cmds, script_cmds, inputs_dict)
    sp_conv_logger('Beginning singlepoint calculation')
    sp_conv_logger('cmds: '+str(cmds))
    atoms = read_atoms(restart=True, script_cmds=script_cmds) # restart=True to read CONTCAR every time
    # JDFTx.py sets coulomb truncation. It uses the get_pbc() method on the ASE atoms structure to do so
    atoms.set_pbc([True, True, False]) # currently only does slab calculations
    calculator = set_calc(cmds, script_cmds, jdftx_ionic = False, interactive=interactive)
    atoms.set_calculator(calculator)
    max_steps = 0 # singlepoint, max_steps should always be zero
    fmax = 1000
    traj = Trajectory('opt.traj', 'w', atoms, properties=['energy', 'forces'])
    dyn = optimizer(atoms, script_cmds, logfile="singlepoint.log")
    dyn.run(fmax=fmax,steps=max_steps)
    traj.close()
    sp_conv_logger('Singlepoint calculation complete')
    subprocess.run("cp singlepoint_inputs singlepoint_convergence", shell=True)

    
################################## Single point run function ###########################################################
########################################################################################################################
################################## Main optimization calculation run function ##########################################

# main function for calculations
def run_calc(command_file, jdftx_exe, autodoscmd, interactive, killcmd):

    #These tags in notinclude signify tags that are associated with ASE but not JDFTx
    notinclude = ['ion-species','ionic-minimize',
                  #'latt-scale','latt-move-scale','coulomb-interaction','coords-type',
                  'ion','climbing','pH','ph',  'autodos',
                  'logfile','pseudos','nimages','max_steps','max-steps','fmax','optimizer','opt',
                  'restart','parallel','safe-mode','hessian', 'step', 'Step',
                  'opt-alpha', 'md-steps', 'econv', 'pdos', 'pDOS', 'lattice-type', 'np',
                  'use_jdftx_ionic', 'jdftx_steps', #Tags below here were added by Cooper
                  'bond-fix', 'bader']

    inputs = open_inputs(command_file) # inputs is a string read from an inputs file
    cmds, script_cmds = read_commands(inputs,notinclude) # cmds is a list of tuples, script_cmds is a dictionary
#    cmds = add_dos(cmds)

    ctype = calc_type(cmds, script_cmds)
    conv_logger('ctype: '+ctype)
    # traj = Trajectory('opt.traj', 'w', atoms, properties=['energy', 'forces'])
    # dyn.attach(traj.write, interval=1)
#    psuedos = script_cmds['pseudos']
#    max_steps = int(script_cmds['max_steps'])
#    fmax = float(script_cmds['fmax'])
#    restart = True if ('restart' in script_cmds and script_cmds['restart'] == 'True') else False
#    parallel_neb = True if ('parallel' in script_cmds and script_cmds['parallel'] == 'True') else False
#    climbing_neb = True if ('climbing' in script_cmds and script_cmds['climbing'] == 'True') else False
#    safe_mode = False if ('safe-mode' in script_cmds and script_cmds['safe-mode'] == 'False') else True
#    use_hessian = True if ('hessian' in script_cmds and script_cmds['hessian'] == 'True') else False
    


    


    
    def bond_constraint(atoms, script_cmds):
        '''
        Constrain bond lenghts according to 'bond-fix' tag in inputs

        returns:
            Atoms object with constraints specified in script_cmds
        '''
        # Constrain bond lenghts according to 'bond-fix' tag in inputs
        try:
            position_constraint = atoms.constraints[0] # get selective dynamics of POSCAR
        except:
            position_constraint = atoms.constraints # if no selective dynamics exist, this returns a blank array
        if 'bond-fix' in script_cmds.keys():
            indices = script_cmds["bond-fix"].split() # bonded atom indices
            constraints = []
            assert len(indices) % 2 == 0, "list of bond-fix atoms must have an even number of items"
            for i, index in enumerate(indices):
                if i % 2 == 0:
                    constraints.append(FixBondLength((int(indices[i]) - 1), int(indices[i+1]) - 1))
                    #input tags are indexed to one but ase indexes to zero
                    conv_logger("constraining bonds {bond_1} and {bond_2}".format(bond_1=int(indices[i]), bond_2=int(indices[i+1])))
            # bond_constraint = FixBondLength(int(indices[0])-1, int(indices[1])-1)
            constraints.append(position_constraint)
            atoms.set_constraint(constraints)
            return atoms
        else:
            conv_logger("Didn't constrain bond")
            return atoms
    def bader():
        try:
            subprocess.run("jbader.py -f tinyout", shell=True)
        except:
            conv_logger("jbader.py didn't run correctly")

    def read_convergence():
        '''
        convergence example:
        step 1
        kpoints 1 1 1
        
        step 2
        kpoints 3 3 3
        '''
        with open('convergence','r') as f:
            conv_txt = f.read()        
        step = '1'
        add_step = False
        for line in conv_txt.split('\n'):
            if any(x in line for x in ['step ','Step ']):
                step = line.split()[1]
                if step == '0':
                    add_step = True
                if add_step:
                    step  = str(int(step)+1) # update so steps are always indexed to 1
            else:
                cmd, val, typ = read_line(line, notinclude)
                if typ == 'None':
                    continue
                if step in conv:
                    # repeat convergence tags
                    if cmd in conv[step] and type(conv[step][cmd][0]) == list:
                        conv[step][cmd][0].append(val)
                    elif cmd in conv[step]:
                        conv[step][cmd] = [[conv[step][cmd][0]], conv[step][cmd][1]]
                        conv[step][cmd][0].append(val)
                    else:
                        # standard
                        conv[step][cmd] = [val,typ]
                else:
                    # add new step 
                    conv[step] = {cmd: [val,typ]}
        return conv, len(conv)
    
    def read_out_nbands():
        with open('out','r', errors='ignore') as f:
            out_txt = f.read()
        for line in out_txt.split('\n')[::-1]:
            if 'nBands' in line and 'nElectrons' in line and 'nStates' in line:
                return line.split()[3]
        
    def update_cmds(conv, step, cmds, script_cmds, update_wfns = False):
        updates = conv[str(step)]
        new_cmds = {}
        for cmd, val in updates.items():
            # updates: {'cmd': [v, type], 'OR', [[v1, v2], type]}
            # val = list
            if val[1] == 'None':
                continue
            elif val[1] == 'script':
                script_cmds[cmd] = val[0]
            elif val[1] == 'cmd':
                new_cmds[cmd] = val[0]
        
        if update_wfns:
            ecut = '0'
            nbands = '0'
            if 'elec-cutoff' in new_cmds and step > 1 and 'elec-cutoff' in conv[str(step-1)]:
                ecut = conv[str(step-1)]['elec-cutoff'][0]
            if 'kpoint-folding' in new_cmds and step > 1 and 'kpoint-folding' in conv[str(step-1)]:
                nbands = read_out_nbands()
            if ecut != '0' or nbands != '0':
                new_cmds['wavefunction'] = 'read wfns '+nbands+' '+ecut
                new_cmds['elec-initial-fillings'] = 'read fillings '+nbands
        
        formatted_cmds = [(cmd,val) for cmd,val in new_cmds.items()]
        formatted_cmds += [cmd for cmd in cmds if cmd[0] not in new_cmds]
        return formatted_cmds, script_cmds
    
    def read_prev_step(file):
        if not os.path.exists(file):
            return 0
        with open(file,'r') as f:
            log = f.read()
        step = 0
        for line in log.split('\n'):
            if 'Running Convergence Step:' in line:
                step = int(line.split()[-1])
        return step
    
    def clean_folder(conv, step, folder = './', delete = True):
        if not delete:
            return
        elec_tags = ['kpoint-folding', 'elec-cutoff', 'pseudos']
        diffs = False
        for tag in elec_tags:
            t1 = tag in conv[str(step)]
            t2 = tag in conv[str(step+1)]
            if t1 != t2: # if tag in only one
                diffs = True
            # if tag in both and value is different
            elif t1 == t2 and t1 and conv[str(step)][tag] != conv[str(step+1)][tag]:
                diffs = True
        
        if diffs:
            files_to_remove = ['wfns','fillings','eigenvals','fluidState']
            for file in files_to_remove:
                if ope(opj(folder, file)):
                    subprocess.call('rm '+opj(folder,file), shell=True)
    
    if os.path.exists('conv.log'):
        subprocess.call('rm conv.log', shell=True)
        
    steps = 1
    conv = {}
    previous_step = 0
    
    if ctype == 'md':
        restart = True if ('restart' in script_cmds and script_cmds['restart'] == 'True') else False
        atoms = read_atoms(restart, script_cmds)
        
        calculator = set_calc(cmds, script_cmds, interactive=False) #TODO change back to interactive
        atoms.set_calculator(calculator)
        
        opt = MinimaHopping(atoms=atoms)
        mdsteps = int(script_cmds['md-steps']) if 'md-steps' in script_cmds else 10
        opt(totalsteps=mdsteps)
        
        atoms.write('CONTCAR',format="vasp", direct=True)
        insert_el('CONTCAR')
        
        
    if ctype in ['opt','lattice']:
#        conv_logger('ctype: '+ctype)
        
        if interactive:
            print('Starting JDFTx calculation.')
        
        if os.path.exists('convergence'):
            # check if convergence file exists and setup convergence dictionary of inputs to update
            conv, steps = read_convergence()
            previous_step = read_prev_step('opt.log')
            conv_logger('convergence found: '+str(conv))
            
        assert len(conv) == 0 or set([int(x) for x in conv]) == set(i+1 for i in range(steps)), ('ERROR: '+
                  'steps in convergence file must be sequential!')
        
        conv_logger('starting opt calc with '+str(steps)+' steps.')
        for i in range(steps):
            conv_logger('\nStep '+str(i+1)+' starting\n')
            if i+1 < previous_step: #advance to next step if previous calculation finished on later step
                conv_logger('Step '+str(i+1)+' previously converged')
                continue
            
            if len(conv) > 0:
                # update all commands from convergence file
                cmds, script_cmds = update_cmds(conv, i+1, cmds, script_cmds)
                # update dos tags
                cmds = add_dos(cmds, script_cmds)
                
                conv_logger('Updated cmds and script cmds with convergence file')
                conv_logger('cmds: '+str(cmds))
                conv_logger('script cmds: '+str(script_cmds))
                conv_logger('\nRunning Convergence Step: '+str(i+1), 'opt.log')
            
            if i == 0:
                restart = True if ('restart' in script_cmds and script_cmds['restart'] == 'True') else False
            else:
                restart = True
            
            max_steps = int(script_cmds['max_steps']) if 'max_steps' in script_cmds else (
                        int(script_cmds['max-steps']) if 'max-steps' in script_cmds else 100) # 100 default
            
            # single point calculation consistent notation
            if max_steps == 0 and comp in ['Summit','Alpine']:
                max_steps = 1
            elif max_steps == 1 and comp in ['Eagle','Perlmutter']:
                max_steps = 0
            
            # set atoms object
            atoms = read_atoms(restart, script_cmds)
            atoms = bond_constraint(atoms, script_cmds) #if bond-fix 
            
            autodos_tag = True if ('autodos' in script_cmds and script_cmds['autodos'] == 'True') else False
            # auto add all pdos for single points and clean cmds
            if (max_steps in [0, 1] and autodoscmd) or autodos_tag:
                cmds = autodos_sp(cmds, atoms)
            # clean repeat dos cmds
            cmds = clean_doscmds(cmds)

            # need to update
            
            # bundle ionic steps within jdftx to increase speed, if requested
            jdftx_ionic = True if ('use_jdftx_ionic' in script_cmds and script_cmds['use_jdftx_ionic'] == 'True') else False
            
            # set calculator
            calculator = set_calc(cmds, script_cmds, jdftx_ionic = jdftx_ionic, interactive=False) #TODO change back to interactive
            atoms.set_calculator(calculator)
    
            #Structure optimization                
            dyn = optimizer(atoms, script_cmds)
    
            def write_contcar(a=atoms):
                #conv_logger('write_contcar')
                a.write('CONTCAR',format="vasp", direct=True)
                insert_el('CONTCAR')
            
            def contcar_from_out(a=atoms):
                #conv_logger('contcar_from_out')
                # for lattice optimizations, write contcar file from out file between steps
                st = h.read_out_struct('./')
                st.to(filename='CONTCAR',fmt='POSCAR')
                dyn.atoms = read_atoms(True, script_cmds)
                dyn.atoms.set_calculator(calculator)
                # Done: Test lattice opt
                
            e_conv = (float(script_cmds['econv']) if 'econv' in script_cmds else 0.0) 
            energy_log = []
            def energy_convergence(a=atoms):
                if e_conv > 0.0:
                    e = a.get_potential_energy(force_consistent=False)
                    if len(energy_log) > 1: # look at last two energies for consistency 
                        if np.abs(e - energy_log[-1]) < e_conv and np.abs(e - energy_log[-2]) < e_conv:
                            #dyn.max_steps = 0
                            conv_logger('Energy convergence satisfied.')
                            conv_logger(str(e_conv) +' '+ str(e) +' '+ str(energy_log))
                            assert False, 'Energy Converged (code xkcd)'
                    energy_log.append(e)
                # Done: add attachment to optimizer to stop running based on energy conv (assert False)
    
#            Only energy and forces seem to be implemented in JDFTx.py. A Trajectory
#            object must be attached so JDFTx does not error out trying to get stress
            traj = Trajectory('opt.traj', 'w', atoms, properties=['energy', 'forces'])
            dyn.attach(traj.write, interval=1)
            
            
            if ctype == 'opt' and jdftx_ionic:
                conv_logger('Using jdftx as ionic minimizer')
                dyn.attach(contcar_from_out,interval=1)
            
            if ctype == 'opt' and not jdftx_ionic:
                conv_logger('Added write_contcar')
                dyn.attach(write_contcar,interval=1)
            if ctype == 'lattice':
                conv_logger('Added contcar_from_out')
                dyn.attach(contcar_from_out,interval=1)
            
            dyn.attach(energy_convergence,interval=1) # stop calculation on energy convergence if requested
            
            safe_mode = False if ('safe-mode' in script_cmds and script_cmds['safe-mode'] == 'False') else True
            if safe_mode: 
                dyn.attach(force_checker,interval=1)
            
#            max_steps = int(script_cmds['max_steps']) if 'max_steps' in script_cmds else (
#                        int(script_cmds['max-steps']) if 'max-steps' in script_cmds else 100) # 100 default
#            
#            # single point calculation consistent notation
#            if max_steps == 0 and comp in ['Summit']:
#                max_steps = 1
#            elif max_steps == 1 and comp in ['Eagle']:
#                max_steps = 0
#            
#            if max_steps in [0, 1]:
#                autodos()
                
#            fmax = float(script_cmds['fmax'])
            fmax = float(script_cmds['fmax']) if 'fmax' in script_cmds else 0.01 # default 0.01
            
            conv_logger('Max steps: '+str(max_steps))
            conv_logger('fmax: '+str(fmax))
            
            # using try loop for energy convergence assertion
            try:
                dyn.run(fmax=fmax,steps=max_steps)
                traj.close()
            except Exception as e:
                conv_logger(str(e))
                if 'Energy Converged (code xkcd)' in str(e):
                    conv_logger('META: Energy Converged Exception.')
                else:
                    print(e) # Done: make sure this syntax will still print JDFT errors correctly
                    assert False, str(e)
            conv_logger('Step '+str(i+1)+' complete!')
            
            if i+1 < steps:
                clean_folder(conv, i+1) # clear out state files if not on final convergence step
            if script_cmds.get("bader", False): # checks if "bader" is a key in script commands. If it is, it returns the assoacited value.
                # If the key doesn't exist, it returns false. Then makes the tinyout and runs bader analysis
                h.make_tinyout(os.getcwd())
                bader()
            
    if ctype == 'neb':
        if os.path.exists('convergence'):
            # check if convergence file exists and setup convergence dictionary of inputs to update
            conv, steps = read_convergence()
            previous_step = read_prev_step('neb.log')
            
        assert len(conv) == 0 or set([int(x) for x in conv]) == set(i+1 for i in range(steps)), ('ERROR: '+
                  'steps in convergence file must be sequential!')
        
        nimages = int(script_cmds['nimages'])
        image_dirs = [str(i).zfill(2) for i in range(0,nimages+2)]
        
        if interactive:
            print('Starting GCNEB calculation with', nimages, 'images.')
        
        conv_logger('starting neb calc with '+str(steps)+' steps.')
        # setup steps
        for ii in range(steps):
            conv_logger('Step '+str(ii+1)+' starting')
            if ii+1 < previous_step:
                conv_logger('Step '+str(ii+1)+' previously converged')
                continue
            
            if len(conv) > 0:
                #cmds, script_cmds = update_cmds(conv[str(ii+1)], cmds, script_cmds)
                cmds, script_cmds = update_cmds(conv, ii+1, cmds, script_cmds)
                conv_logger('updated cmds and script cmds with convergence file')
                conv_logger('Running Convergence Step: '+str(ii+1), 'neb.log')
            
            # clean repeat dos cmds
            cmds = clean_doscmds(cmds)
            
            if ii == 0:
                restart = True if ('restart' in script_cmds and script_cmds['restart'] == 'True') else False
            else:
                restart = True
        
            # read in neb images
            try:
                initial = read('00/CONTCAR')
                final = read(opj(str(nimages+1).zfill(2),'CONTCAR'))
            except:
                assert False, 'ERROR: Add CONTCAR files to 00 and '+str(nimages+1).zfill(2)+' directories.'
            
            images = [initial]
            if restart:
                try:
                    images += [read(opj(im,'CONTCAR')) for im in image_dirs[1:-1]]
                except:
                    images += [read(opj(im,'POSCAR')) for im in image_dirs[1:-1]]
                    print('WARNING: CONTCAR files not found, starting from POSCARs')
            else:
                images += [read(opj(im,'POSCAR')) for im in image_dirs[1:-1]]
            images += [final]
        
            parallel_neb = True if ('parallel' in script_cmds and script_cmds['parallel'] == 'True') else False
            climbing_neb = True if ('climbing' in script_cmds and script_cmds['climbing'] == 'True') else False
    
            neb = NEB(images, parallel=parallel_neb, climb=climbing_neb) 
            for im, image in enumerate(images[1:-1]):
                image.calc = set_calc(cmds, script_cmds, outfile=opj(os.getcwd(),image_dirs[im+1]))
                
            dyn = optimizer(neb, script_cmds, logfile = 'neb.log')
    
            '''
            Only energy and forces seem to be implemented in JDFTx.py. A Trajectory
            object must be attached so JDFTx does not error out trying to get stress
            '''
            traj = Trajectory('neb.traj', 'w', neb, properties=['energy', 'forces'])
    
            def write_contcar(img_dir, image):
                image.write(opj(img_dir,'CONTCAR'),format="vasp", direct=True)
                insert_el(opj(img_dir,'CONTCAR'))
    
            for im,image in enumerate(images[1:-1]):
                img_dir = image_dirs[im+1]
                dyn.attach(Trajectory(opj(os.getcwd(),img_dir,'opt-'+img_dir+'.traj'), 'w', image,
                                      properties=['energy', 'forces']))
                dyn.attach(write_contcar, interval=1, img_dir=img_dir, image=image)
    
            safe_mode = False if ('safe-mode' in script_cmds and script_cmds['safe-mode'] == 'False') else True
            if safe_mode: 
                dyn.attach(force_checker,interval=1)
            
#            max_steps = int(script_cmds['max_steps'])
            max_steps = int(script_cmds['max_steps']) if 'max_steps' in script_cmds else (
                        int(script_cmds['max-steps']) if 'max-steps' in script_cmds else 100) # 100 default
            
            fmax = float(script_cmds['fmax']) if 'fmax' in script_cmds else 0.0
            dyn.run(fmax=fmax,steps=max_steps)
            traj.close()
            conv_logger('Step '+str(ii+1)+' complete!')
            
            if ii+1 < steps:
                for folder in image_dirs:
                    if folder in ['00', str(nimages+1).zfill(2)]:
                        continue
                    clean_folder(conv, ii+1, folder+'/')
    
    if killcmd:
        assert False, "\n\n### Calculation completed: Kill command called to stop job ###\n\n"

################################## Main optimization calculation run function ##########################################
########################################################################################################################

if __name__ == '__main__':
    
    jdftx_exe = os.environ['JDFTx']
    
    # optional, change to another directory (for parallel runs)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='Directory to run in and save files to.',
                        type=str, default='./')
    parser.add_argument('-g', '--gpu', help='If True, runs GPU install of JDFTx.',
                        type=str, default='False')
    parser.add_argument('-ad', '--autodos', help='If True (Default), adds dos tags to SP calcs.',
                        type=str, default='True')
    parser.add_argument('-int', '--interactive', help='If True, run on interactive queue',
                        type=str, default='False')
    parser.add_argument('-k', '--kill', help='If True, kill calculation on completion to prevent I/O issues.',
                        type=str, default='False')
    parser.add_argument('-r', '--regen', help='If True, check calc convergence and rerun if needed.',
                        type=str, default='False')
    parser.add_argument('--singlepoint', help=('If True, run in singlepoint mode. The script looks'
                                                       ' for a single_point_inputs file and ignores everything else but a CONTCAR'),
                        type=bool, default=False)
#    parser.add_argument('-p', '--parallel', help='If True, runs parallel sub-job with JDFTx.',
#                        type=str, default='False')


    args = parser.parse_args()
    if args.dir != './':
        os.chdir(args.dir)
    if args.gpu == 'True':
        try:
            jdftx_exe = os.environ['JDFTx_GPU']
        except:
            print('Environment variable "JDFTx_GPU" not found, running standard JDFTx.')
    
#    if args.parallel == 'True':
#        jdftx_exe = '-N 1 -n 4 '+jdftx_exe

    command_file = 'inputs'
    
    conv_logger('\n\n----- Entering run function -----')
    autodoscmd = True if args.autodos == 'True' else False
    killcmd = True if args.kill == 'True' else False
    if args.singlepoint == False:
        run_calc(command_file, jdftx_exe, autodoscmd, 
                True if args.interactive == 'True' else False,
                killcmd)
    elif args.singlepoint == True:
        run_singlepoint(jdftx_exe, interactive=True if args.interactive == 'True' else False)
     
    # if calc did not converge, try to restart up to 3 times
    regen = True if args.regen == 'True' else False
    if regen:
        for attempt in range(3):
            data = h.read_data(os.getcwd())
            if not data['converged']:
                run_calc(command_file, jdftx_exe, autodoscmd, 
                         True if args.interactive == 'True' else False,
                         killcmd)
            else:
                break
