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
from ase.units import Hartree, Bohr
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

notinclude = [
    'ion-species',
            #   'ionic-minimize',
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
        return conv

def native_exe_cmd(gpu=True, ):
    if comp == 'Eagle':
        exe_cmd = 'mpirun --bind-to none '+jdftx_exe
    elif comp in ['Cori',]:
        exe_cmd = 'srun --cpu-bind=cores -c 8 '+jdftx_exe
        conv_logger('Running on Cori with srun.')
    elif comp in ['Perlmutter']:
        exe_cmd = 'srun '+jdftx_exe
        conv_logger("Running on Perl using exe cmd {exe_cmd}".format(exe_cmd=exe_cmd))
        
    return exe_cmd

def convert_cmds_to_native(cmds, script_cmds):
    # convert the ase commands to be compatible with native JDFTx optimization
    ctype = h.calc_type(cmds, script_cmds)
    print(script_cmds["max_steps"], "max steps command")
    # build the ionic optimization tag
    ionic_tag = "maxThreshold yes "
    if ctype in ['opt', 'sp']:
        if "fmax" in script_cmds:
            jdft_fmax = float(script_cmds.pop("fmax")) / (Hartree / Bohr)
            ionic_tag += f"knormThreshold {jdft_fmax} "
        if "max_steps" in script_cmds:
            max_steps = int(script_cmds.pop("max_steps"))
            ionic_tag += f"nIterations {max_steps} "
        if "econv" in script_cmds:
            econv = float(script_cmds.pop("econv")) / Hartree
            ionic_tag += f"energyDiffThreshold {econv} "
        # if the ionic tag is not already in the cmds, add it
        # Don't add it twice if it's already there
        if check_cmd(cmds, "ionic-minimize") == False:
            cmds.append(("ionic-minimize", ionic_tag))
    elif ctype == 'lattice':
        if "max_steps" in script_cmds:
            max_steps = int(script_cmds.pop("max_steps"))
            ionic_tag += f"nIterations {max_steps} "
        if "econv" in script_cmds:
            econv = float(script_cmds.pop("econv")) * Hartree
            ionic_tag += f"energyDiffThreshold {econv} "
        cmds.append(("lattice-minimize ", ionic_tag))
        if check_cmd(cmds, "lattice-minimize") == False:
            cmds.append(("lattice-minimize", ionic_tag))
    return cmds, script_cmds    

def check_cmd(cmds, cmd):
    for c in cmds:
        if cmd == c[0]:
            return True
    return False

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
            conv= read_convergence_with_type(conv)
            steps = len(conv)
            # returns a number corresponding to the last step recorded in the opt.log
            previous_step = h.read_prev_step('opt.log')
            conv_logger('convergence found: '+str(conv))
        
        assert len(conv) == 0 or set([int(x) for x in conv]) == set(i+1 for i in range(steps)), ('ERROR: '+
                  'steps in convergence file must be sequential!')
        
        # This file is written after the JDFTx calculation runs. It must be removed 
        # before the start of the convergence loop to tell the convergence
        # logic that the calculation is not running.
        if ope('calc_running'):
            os.remove('calc_running')

        conv_logger('starting opt calc with '+str(steps)+' steps.')
        for i in range(steps): # steps are from convergence file
            convergence_step = i+1 # switching to 1 index to be consistent with opt.log
            convergence_manager = CalcConv(convergence_step, ctype)
            # if step is converged according to parse file, continue to next step
            if convergence_manager.check_step_convergence(convergence_step):
                continue
            if len(conv) > 0:
                print('Updating commands with convergence file')
                # update all commands from convergence file
                print(script_cmds, '\n cmds before updating with convergence file')
                inputs = h.open_inputs(command_file) # inputs is a string read from an inputs file
                cmds, script_cmds = h.read_commands(inputs, notinclude)
                cmds, script_cmds = h.update_cmds(conv, convergence_step, cmds, script_cmds)
                # update dos tags
                # cmds = add_dos(cmds, script_cmds)
                
                conv_logger('Updated cmds and script cmds with convergence file')
                conv_logger('cmds: '+str(cmds))
                conv_logger('script cmds: '+str(script_cmds))
                print("adding opt.log boilerplate")
                conv_logger('\nRunning Convergence Step: '+str(convergence_step), 'opt.log')
            
            # convert the ase commands to be compatible with native JDFTx optimization
            cmds, script_cmds = convert_cmds_to_native(cmds, script_cmds)
            conv_logger('Commands after converting to native format')
            conv_logger(str(cmds))
            conv_logger(str(script_cmds))
            if i == 0:
                restart = True if ('restart' in script_cmds and script_cmds['restart'] == 'True') else False
            else: 
                restart = True

            
            if not ope(f'parse_{convergence_step}'):
                if ope('out'):
                    # two options. Advancing to next convergence step in the calculation loop or
                    # the calculation timed out last time.

                    if convergence_manager.calc_running():
                        # the JDFTx calculation just finished and we are on the next convergence step.
                        # don't need to parse any outputs, just continue onto the next JDFTx run.
                        conv_logger('Advancing to JDFTx calculation.')
                        pass

                    elif convergence_manager.calc_running() == False:
                        # calculation timed out on the current convergence step.
                        # Need to parse output, write CONTCAR, and start from there.
                        # Also need to change restart to True after writing a CONTCAR
                        conv_logger(f"Calculation timed out on step {convergence_step}. Parsing outputs and writing CONTCAR.")
                        parser = convergence_manager.get_parser(conv_logger, ctype)
                        if parser.check_calc_took_step() == False:
                            # last time it timed out, it didn't take a step. Need to start from last saved point.
                            # Sometimes this will mean starting from scratch.
                            conv_logger('Last calculation did not take a step. Starting from last saved point.')
                            if ope('CONTCAR'):
                                conv_logger('Found CONTCAR, setting restart to True.')
                                restart = True
                        elif parser.check_calc_took_step():
                            parser.write_parse_file(convergence_step)
                            optlog_text = parser.build_optlog()
                            parser.write_trajectory()
                            parser.write_contcar()
                            with open('opt.log', 'a') as f:
                                f.write(optlog_text)
                            restart = True
                else:
                    if convergence_step == 1:
                        conv_logger('No out file found. Starting calculation from scratch')
                    else:
                        conv_logger(f'No out file found on steps {convergence_step}. Awaiting manual fix')
                        break
            
            elif ope(f'parse_{convergence_step}'):
                converged = convergence_manager.check_parse_file_convergence(convergence_step)
                if converged:
                    # If the calculation converged, we don't need to do anything. Just continue to the next step
                    conv_logger('Step '+str(convergence_step)+' converged')
                    continue
                else:
                    timeout = convergence_manager.check_parse_file_timeout(convergence_step)
                    if timeout:
                        conv_logger('Step '+str(convergence_step)+' timed out. Running calculation from timeout point')
                        # Need to parse outputs from the timed out calculation
                        parser = convergence_manager.get_parser(conv_logger, ctype)
                        calc_started = parser.check_calc_started()
                        if calc_started:
                            parser.write_parse_file(convergence_step)
                            optlog_text = parser.build_optlog()
                            parser.write_trajectory()
                            parser.write_contcar()
                            with open('opt.log', 'a') as f:
                                f.write(optlog_text)
                        elif calc_started == False:
                            # If the next calculation hadn't started when the previous one timed out, was parsed,
                            # and saved, then there's nothing more in the out file to parse. Instead, continue on to the
                            # calculation loop.
                            pass
                        restart = True
                    else:
                        conv_logger('Step '+str(convergence_step)+' did not converge nor timeout. Awaiting manual fix')
                        break
            
            
            print(restart, 'restart before initializing atoms')
            # set up atoms object
            atoms = h.read_atoms(restart, script_cmds)
            print(atoms.constraints, 'atoms.constraints in run_JDFTx_native.py')

            # single point calculation consistent notation
            max_steps = int(script_cmds['max_steps']) if 'max_steps' in script_cmds else 0
            if max_steps == 0 and comp in ['Summit','Alpine']:
                max_steps = 1
            elif max_steps == 1 and comp in ['Eagle','Perlmutter']:
                max_steps = 0
            

            # No longer adding DOS tags to JDFTx runs. All the DOS information is recoverable
            # from the bandProjections.
            # autodos_tag = True if ('autodos' in script_cmds and script_cmds['autodos'] == 'True') else False
            # auto add all pdos for single points and clean cmds
            # if (max_steps in [0, 1] and autodoscmd) or autodos_tag:
            #     cmds = h.autodos_sp(cmds, atoms)
            # clean repeat dos cmds
            # cmds = h.clean_doscmds(cmds)

            if ctype == 'lattice':
                conv_logger('Added contcar_from_out')

            fmax = float(script_cmds['fmax']) if 'fmax' in script_cmds else 0.01 # default 0.01
            
            conv_logger('Max steps: '+str(max_steps))
            conv_logger('fmax: '+str(fmax))

            # Running calculation with new logic
            run_exe = "srun " + jdftx_exe
            print("got executable ", run_exe)
            calculator = JDFTx(run_exe, commands=cmds, outfile=os.getcwd(), ionic_steps = [3, 0.0000], pseudoSet=script_cmds['pseudos'])
            print("initialized calculator")
            inputFile = calculator.constructInput(atoms)
            print("Attempting JDFTx calculation")

            try: 
                # Run calculation in JDFTx
                calculator.runJDFTx(inputFile)
            except Exception as e:
                conv_logger('CALCULATION ERROR: '+str(e))
                print(e)
                break

            # file gets written after JDFTx calculation is finished
            # and tells convergence logic that the calculation
            # is still running so that the out file is not parsed
            with open('calc_running', 'w') as f:
                f.write('True')

            # Parsing output file
            if convergence_step == steps:
                # If on last step, feed singlepoint calc type into the outparser
                #NOTE: This means the last step can only be a single point.
                ctype_sp = 'sp'
                parser = convergence_manager.get_parser(conv_logger, ctype_sp)
                print(parser.index_string, 'parser.index_string')
                print(list(parser.step_indices()), "step indices")
            else:
                print("got parser after calculation")
                parser = convergence_manager.get_parser(conv_logger, ctype)
            parser.write_parse_file(convergence_step)
            optlog_text = parser.build_optlog()
            print('built optlog text')
            parser.write_trajectory()
            parser.write_contcar()
            with open('opt.log', 'a') as f:
                f.write(optlog_text)

            # check if calculation converged
            converged = parser.check_convergence()
            if converged:
                conv_logger('Step '+str(i+1)+' converged')
            else:
                conv_logger('Step '+str(i+1)+' did not converge')
                timeout = parser.check_timeout()
                if timeout:
                    conv_logger('Step '+str(i+1)+' timed out')
                else:
                    conv_logger('Step '+str(i+1)+' did not time out')
                    conv_logger('Breaking calculation loop and finishing for manual check')
                    break
            
            print(convergence_step, steps, 'convergence_step, steps')
            if convergence_step < steps:
                # clear out state files for early convergence step. State files are preserved for penultimate step
                # NOTE: This only works if the singlepoint does not change electronic parameters like cutoff energy or kpoints
                # compared to the penultimate convergence step
                print("cleaning state files")
                h.clean_folder(conv, convergence_step) 

            elif convergence_step == steps:
                # calculation is fully finished
                h.make_tinyout(os.getcwd())
                conv_logger('Calculation fully converged')
                with open('converged', 'w') as f:
                    f.write('True')
                
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
    parser.add_argument('--executable', help=('Specify an exectable string. For gpus with 2 gpus per job, the '
                                              'string shou ld be "-N 1 -n 2 /path/to/jdftx_gpu"'),
                        type=str, default=None)
#    parser.add_argument('-p', '--parallel', help='If True, runs parallel sub-job with JDFTx.',
#                        type=str, default='False')


    args = parser.parse_args()
    if args.dir != './':
        os.chdir(args.dir)
    if args.executable != None:
        jdftx_exe = args.executable
    else:
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
