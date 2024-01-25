# running native ionic optimizaiton
# written by Cooper

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
from run_JDFTx import *
from OutParser import OutParser
from CalcConv import CalcConv
from suggest_input_tags import set_elec_n_bands
import re
h = helper()

opj = os.path.join
ope = os.path.exists
hartree_to_ev = 27.2114

notinclude = ['ion-species','ionic-minimize',
            #'latt-scale','latt-move-scale','coulomb-interaction','coords-type',
            'ion','climbing','pH','ph',  'autodos',
            'logfile','pseudos','nimages','max_steps','max-steps','fmax','optimizer','opt',
            'restart','parallel','safe-mode','hessian', 'step', 'Step',
            'opt-alpha', 'md-steps', 'econv', 'pdos', 'pDOS', 'lattice-type', 'np',
            'use_jdftx_ionic', 'jdftx_steps', #Tags below here were added by Cooper
            'bond-fix', 'bader']

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


def read_convergence_with_type(conv):
        # This function is the same as read_convergence() in run_JDFTx.py.
        # It was renamed to avoid a name conflict with the function of the same name in jdft_helper
        # which returns a different data structure and breaks this script.
        # You also need to pass the current conv ditionary, but an empty {} is fine.
        '''
        convergence example:
        step 1
        kpoints 1 1 1
        
        step 2
        kpoints 3 3 3

        Returns a dictionary of the form: {'cmd': [val, type]}
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

################################## Main optimization calculation run function ##########################################

# main function for calculations
def run_calc(command_file, jdftx_exe, autodoscmd, interactive, killcmd):

    inputs = h.open_inputs(command_file) # inputs is a string read from an inputs file
    cmds, script_cmds = h.read_commands(inputs, notinclude) # cmds is a list of tuples, script_cmds is a dictionary

    ctype = h.calc_type(cmds, script_cmds)
    conv_logger('ctype: '+ctype)
    
    if os.path.exists('conv.log'):
        subprocess.call('rm conv.log', shell=True)
        
    steps = 1
    conv = {}
    previous_step = 0
        
    if ctype in ['opt','lattice']:
#        conv_logger('ctype: '+ctype)
        
        if interactive:
            print('Starting JDFTx calculation.')
        
        if os.path.exists('convergence'):
            # check if convergence file exists and setup convergence dictionary of inputs
            conv, steps = read_convergence_with_type(conv)
            # returns a number corresponding to the last step recorded in the opt.log
            previous_step = h.read_prev_step('opt.log')
            conv_logger('convergence found: '+str(conv))
        
        assert len(conv) == 0 or set([int(x) for x in conv]) == set(i+1 for i in range(steps)), ('ERROR: '+
                  'steps in convergence file must be sequential!')
        
        conv_logger('starting opt calc with '+str(steps)+' steps.')
        for i in range(steps): # steps are from convergence file
            convergence_step = i+1 # switching to one index to be consistent with opt.log
            conv_logger('\nStep '+str(i+1)+' starting\n')
            if i+1 < previous_step: #advance to next step if previous calculation finished on later step
                conv_logger('Step '+str(i+1)+' previously converged')
                continue
            
            if len(conv) > 0:
                print('Updating commands with convergence file')
                # update all commands from convergence file
                cmds, script_cmds = h.update_cmds(conv, i+1, cmds, script_cmds)
                # update dos tags
                cmds = add_dos(cmds, script_cmds)
                
                conv_logger('Updated cmds and script cmds with convergence file')
                conv_logger('cmds: '+str(cmds))
                conv_logger('script cmds: '+str(script_cmds))
                print("adding opt.log boilerplate")
                conv_logger('\nRunning Convergence Step: '+str(i+1), 'opt.log')
            
            if i == 0:
                restart = True if ('restart' in script_cmds and script_cmds['restart'] == 'True') else False
            else:
                restart = True
            
            # set up atoms object
            atoms = h.read_atoms(restart, script_cmds)

            # single point calculation consistent notation
            max_steps = int(script_cmds['max_steps']) if 'max_steps' in script_cmds else 0
            if max_steps == 0 and comp in ['Summit','Alpine']:
                max_steps = 1
            elif max_steps == 1 and comp in ['Eagle','Perlmutter']:
                max_steps = 0
            
            autodos_tag = True if ('autodos' in script_cmds and script_cmds['autodos'] == 'True') else False
            # auto add all pdos for single points and clean cmds
            if (max_steps in [0, 1] and autodoscmd) or autodos_tag:
                cmds = h.autodos_sp(cmds, atoms)
            # clean repeat dos cmds
            cmds = h.clean_doscmds(cmds)

            # bundle ionic steps within jdftx to increase speed, if requested
            jdftx_ionic = True # superfluous maybe

            e_conv = (float(script_cmds['econv']) if 'econv' in script_cmds else 0.0) 
            energy_log = []  
            
            if ctype == 'opt' and jdftx_ionic:
                conv_logger('Using jdftx as ionic minimizer')

            if ctype == 'lattice':
                conv_logger('Added contcar_from_out')

            fmax = float(script_cmds['fmax']) if 'fmax' in script_cmds else 0.01 # default 0.01
            
            conv_logger('Max steps: '+str(max_steps))
            conv_logger('fmax: '+str(fmax))
            ######### OLD LOGIC ######### 
            # using try loop for energy convergence assertion
            # try:
            #     dyn.run(fmax=fmax,steps=max_steps)
            #     traj.close()
            # except Exception as e:
            #     conv_logger('ERROR: '+str(e))
            #     print(e) # Done: make sure this syntax will still print JDFT errors correctly
            #     assert False, str(e)
            ######### OLD LOGIC ######### 

            # Running calculation with new logic
            jdftx_exe = "srun " + os.environ["JDFTx_GPU"] #TODO need to fix this
            print("got executable")
            calculator = JDFTx(jdftx_exe, commands=cmds, outfile=os.getcwd(), ionic_steps = [3, 0.0000])
            print("initialized calculator")
            inputFile = calculator.constructInput(atoms)
            print("Attempting JDFTx calculation")
            try: 
                calculator.runJDFTx(inputFile)
            except Exception as e:
                conv_logger('CALCULATION ERROR: '+str(e))
                print(e)

            # Parsing output file
            if ope(f'parse_{convergence_step}'):
                print("parse option 1")
                with open(f'parse_{convergence_step}', 'r') as f:
                    lines = f.readlines()
                # There is one number in the parse file which shows where the parser finished 
                # Reading the out file last time this calculation was run in the same convergence step
                last_parse_line = int(lines[0]) 
                conv_logger(f'parse_{convergence_step} found, current convergence step has already been partially parsed')
                conv_logger(f'Beginning parsing after line {last_parse_line} in out file')
                parser = OutParser('out', start_line=last_parse_line, ctype=ctype)
            if ope(f'parse_{convergence_step - 1}'):
                print("parse option 2")
                # Previous convergence step was successfully finished and parsed,
                # so we read that file and start after that line
                with open(f'parse_{convergence_step -1}', 'r') as f:
                    lines = f.readlines()
                # There is one number in the parse file which shows where the parser finished 
                # Reading the out file last time this calculation was run in the same convergence step
                parse_line_from_prev_step = int(lines[0])
                parser = OutParser('out', start_line=parse_line_from_prev_step, ctype=ctype)
                parser = parser.from_end_of_previous_step()
                new_start_line = parser.start_line
                print(f'new_start_line: {new_start_line}')
                conv_logger(f'parse_{convergence_step -1 } found, previous convergence step was successfully finished and parsed')
                conv_logger(f'Beginning parsing after line {new_start_line} in out file')
                parser = OutParser('out', start_line=new_start_line, ctype=ctype)
                with open(f"parse_{convergence_step}", "w") as f:
                    f.write(str(parser.last_line))
            else:
                print("parse option 3")
                last_parse_line = 0
                conv_logger(f'parse_{convergence_step} not found, beginning parsing from beginning of out file')
                parser = OutParser('out', start_line=last_parse_line, ctype=ctype)
                with open(f"parse_{convergence_step}", "w") as f:
                    f.write(str(parser.last_line))
            # now build opt.log, convergence file, and parse file
            optlog_text = parser.build_optlog()
            parser.write_trajectory()
            parser.write_contcar()
            with open('opt.log', 'a') as f:
                f.write(optlog_text)

            conv_logger('Step '+str(i+1)+' complete!')
            
            if i+1 < steps:
                h.clean_folder(conv, i+1) # clear out state files if not on final convergence step
            elif i+1 == steps:
                h.make_tinyout(os.getcwd())
            if script_cmds.get("bader", False): # checks if "bader" is a key in script commands. If it is, it returns the assoacited value.
                # If the key doesn't exist, it returns false. Then makes the tinyout and runs bader analysis
                bader_result = h.bader()
                if bader_result != 0:
                    conv_logger(f"Bader error: {bader_result} ")
                else:
                    conv_logger("Bader analysis successful")
            
    if ctype == 'neb':
        conv_logger("Cannot do NEB with ionic minimization in JDFTx")
    
    if killcmd:
        assert False, "\n\n### Calculation completed: Kill command called to stop job ###\n\n"

################################## Main optimization calculation run function ##########################################
########################################################################################################################






if __name__ == '__main__':
    
    jdftx_exe = os.environ['JDFTx'] #TODO enable variable parallelization
    
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
            jdftx_exe = os.environ['JDFTx_GPU'] #TODO enable variable parallelization
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
        # run_singlepoint(jdftx_exe, interactive=True if args.interactive == 'True' else False)
        pass
     
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
