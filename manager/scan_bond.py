import os
from ase.io import read, write
import subprocess
from ase.io.trajectory import Trajectory
from ase.constraints import FixBondLength
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin, FIRE
from JDFTx import JDFTx
import numpy as np
import shutil


""" HOW TO USE ME:
- Be on perlmutter
- Go to the directory of the optimized geometry you want to perform a bond scan on
- Create a file called "scan_input" in that directory (see the read_scan_inputs function below for how to make that)
- Copied below is an example submit.sh to run it
- Make sure JDFTx.py's "constructInput" function is edited so that the "if" statement (around line 234) is given an
else statement that sets "vc = v + "\n""
- This script will read "CONTCAR" in the directory, and SAVES ANY ATOM FREEZING INFO (if you don't want this, make sure
either there is no "F F F" after each atom position in your CONTCAR or make sure there is only "T T T" after each atom
position (pro tip: sed -i 's/F F F/T T T/g' CONTCAR)
- This script checks first for your "inputs" file for info on how to run this calculation - if you want to change this
to a lower level (ie change kpoint-folding to 1 1 1), make sure you delete all the State output files from the directory
(wfns, fillings, etc)
"""


"""
#!/bin/bash
#SBATCH -J scanny
#SBATCH --time=12:00:00
#SBATCH -o scanny.out
#SBATCH -e scanny.err
#SBATCH -q regular_ss11
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH -C gpu
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -A m4025_g

module use --append /global/cfs/cdirs/m4025/Software/Perlmutter/modules
module load jdftx/gpu

export JDFTx_NUM_PROCS=1

export SLURM_CPU_BIND="cores"
export JDFTX_MEMPOOL_SIZE=36000
export MPICH_GPU_SUPPORT_ENABLED=1

python /global/homes/b/beri9208/BEAST_DB_Manager/manager/scan_bond.py > scan.out
exit 0
"""


gbrv_15_ref = [
    "sn f ca ta sc cd sb mg b se ga os ir li si co cr pt cu i pd br k as h mn cs rb ge bi ag fe tc hf ba ru al hg mo y re s tl te ti be p zn sr n rh au hf nb c w ni cl la in v pb zr o ",
    "14. 7. 10. 13. 11. 12. 15. 10. 3. 6. 19. 16. 15. 3. 4. 17. 14. 16. 19. 7. 16. 7. 9. 5. 1. 15. 9. 9. 14. 15. 19. 16. 15. 12. 10. 16. 3. 12. 14. 11. 15. 6. 13. 6. 12. 4. 5. 20. 10. 5. 15. 11. 12. 13. 4. 14. 18. 7. 11. 13. 13. 14. 12. 6. "
]

# This is horribly incorrect (karma for asking chat gpt to write this for me)
# BUT it only gets called if it cant find the ion name in gbrv_15_ref parallel arrays, which is only true for Na which actually is filled correctly
valence_electrons = {
        'h': 1, 'he': 2,
        'li': 1, 'be': 2, 'b': 3, 'c': 4, 'n': 5, 'o': 6, 'f': 7, 'ne': 8,
        'na': 1, 'mg': 2, 'al': 3, 'si': 4, 'p': 5, 's': 6, 'cl': 7, 'ar': 8,
        'k': 1, 'ca': 2, 'sc': 2, 'ti': 2, 'v': 2, 'cr': 1, 'mn': 2, 'fe': 2, 'co': 2, 'ni': 2, 'cu': 1, 'zn': 2,
        'ga': 3, 'ge': 4, 'as': 5, 'se': 6, 'br': 7, 'kr': 8,
        'rb': 1, 'sr': 2, 'y': 2, 'zr': 2, 'nb': 1, 'mo': 1, 'tc': 2, 'ru': 2, 'rh': 1, 'pd': 0, 'ag': 1, 'cd': 2,
        'in': 3, 'sn': 4, 'sb': 5, 'te': 6, 'i': 7, 'xe': 8,
        'cs': 1, 'ba': 2, 'la': 2, 'ce': 2, 'pr': 2, 'nd': 2, 'pm': 2, 'sm': 2, 'eu': 2, 'gd': 3, 'tb': 3, 'dy': 3,
        'ho': 3, 'er': 3, 'tm': 2, 'yb': 2, 'lu': 2, 'hf': 2, 'ta': 2, 'w': 2, 're': 2, 'os': 2, 'ir': 2, 'pt': 2,
        'au': 1, 'hg': 2, 'tl': 3, 'pb': 4, 'bi': 5, 'po': 6, 'at': 7, 'rn': 8,
    }

def read_scan_inputs():
    """ Example:
    Scan: 1, 4, 10, -.2
    restart_at: 0
    work: /pscratch/sd/b/beri9208/1nPt1H_NEB/calcs/surfs/H2_H2O_start/No_bias/scan_bond_test/

    notes:
    Scan counts your atoms starting from 0, so the numbering in vesta will be 1 higher than what you need to put in here
    The first two numbers in scan are the two atom indices involved in the bond you want to scan
    The third number is the number of steps
    The fourth number is the step size (in angstroms) for each step
    """
    lookline = None
    restart_idx = 0
    work_dir = None
    with open("scan_input", "r") as f:
        for line in f:
            if "scan" in line.lower().split(":")[0]:
                lookline = line.rstrip("\n").split(":")[1].split(",")
            if "restart" in line.lower().split(":")[0]:
                restart_idx = int(line.rstrip("\n").split(":")[1])
            if "work" in line.lower().split(":")[0]:
                work_dir = line.rstrip("\n").split(":")[1]
    atom_pair = [int(lookline[0]), int(lookline[1])]
    scan_steps = int(lookline[2])
    step_length = float(lookline[3])
    return atom_pair, scan_steps, step_length, restart_idx, work_dir

def get_nbands(poscar_fname):
    atoms = read(poscar_fname)
    count_dict = {}
    for a in atoms.get_chemical_symbols():
        if a.lower() not in count_dict.keys():
            count_dict[a.lower()] = 0
        count_dict[a.lower()] += 1
    nval = 0
    for a in count_dict.keys():
        if a in gbrv_15_ref[0].split(" "):
            idx = gbrv_15_ref[0].split(" ").index(a)
            val = (gbrv_15_ref[1].split(". "))[idx]
            count = count_dict[a]
            nval += int(val) * int(count)
        else:
            nval += int(valence_electrons[a]) * int(count_dict[a])
    return max([int(nval / 2) + 10, int((nval / 2) * 1.2)])



def finished(dirname):
    with open(os.path.join(dirname, "finished.txt"), 'w') as f:
        f.write("Done")


def optimizer(atoms, opt="FIRE", opt_alpha=150, logfile='opt.log'):
    """
    ASE Optimizers:
        BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, GPMin, MDMin and FIRE.
    """
    opt_dict = {'BFGS': BFGS, 'BFGSLineSearch': BFGSLineSearch,
                'LBFGS': LBFGS, 'LBFGSLineSearch': LBFGSLineSearch,
                'GPMin': GPMin, 'MDMin': MDMin, 'FIRE': FIRE}
    if opt in ['BFGS', 'LBFGS']:
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='hessian.pckl', alpha=opt_alpha)
    elif opt == 'FIRE':
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='hessian.pckl', a=(opt_alpha / 70) * 0.1)
    else:
        dyn = opt_dict[opt](atoms, logfile=logfile, restart='hessian.pckl')
    return dyn

def bond_constraint(atoms, indices):
    atoms.set_constraint(FixBondLength(indices[0], indices[1]))
    return atoms


def dup_cmds(infile):
    lattice_line = None
    infile_cmds = {}
    infile_cmds["dump"] = "End State"
    ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump-name", "initial-state",
              "coulomb-interaction", "coulomb-truncation-embed"]
    with open(infile) as f:
        for i, line in enumerate(f):
            if "lattice " in line:
                lattice_line = i
            if not lattice_line is None:
                if i > lattice_line + 3:
                    if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                        skip = False
                        for ig in ignore:
                            if ig in line:
                                skip = True
                            elif line[:4] == "ion ":
                                skip = True
                        if not skip:
                            cmd = line[:line.index(" ")]
                            rest = line.rstrip("\n")[line.index(" ") + 1:]
                            if not cmd in ignore:
                                if not cmd == "dump":
                                    infile_cmds[cmd] = rest
                                # else:
                                #     infile_cmds["dump"].append(rest)
    return infile_cmds


def set_calc(exe_cmd, step_dir, inputs_cmds):
    outfile = os.getcwd()
    if inputs_cmds is None:
        cmds = dup_cmds(os.path.join(outfile, "in"))
    else:
        print(dup_cmds(os.path.join(outfile, "in")))
        print(inputs_cmds)
        cmds = inputs_cmds
    return JDFTx(
        executable=exe_cmd,
        pseudoSet="GBRV_v1.5",
        commands=cmds,
        outfile=step_dir,
        ionic_steps=False
    )

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

def read_inputs(inpfname):
    ignore = ["Orbital", "coords-type", "ion-species ", "density-of-states ", "dump", "initial-state",
              "coulomb-interaction", "coulomb-truncation-embed", "lattice-type", "opt", "max_steps", "fmax",
              "optimizer", "pseudos", "logfile", "restart", "econv", "safe-mode"]
    input_cmds = {"dump": "End State"}
    with open(inpfname) as f:
        for i, line in enumerate(f):
            if (len(line.split(" ")) > 1) and (len(line.strip()) > 0):
                skip = False
                for ig in ignore:
                    if ig in line:
                        skip = True
                if not skip:
                    cmd = line[:line.index(" ")]
                    rest = line.rstrip("\n")[line.index(" ") + 1:]
                    if not cmd in ignore:
                        input_cmds[cmd] = rest
    do_n_bands = False
    if "elec-n-bands" in input_cmds.keys():
        if input_cmds["elec-n-bands"] == "*":
            do_n_bands = True
    else:
        do_n_bands = True
    if do_n_bands:
        if os.path.exists("CONTCAR"):
            input_cmds["elec-n-bands"] = str(get_nbands("CONTCAR"))
        else:
            input_cmds["elec-n-bands"] = str(get_nbands("POSCAR"))
    return input_cmds

def prep_input(step_idx, atom_pair, step_length):
    atoms = read(str(step_idx) + "/POSCAR", format="vasp")
    dir_vec = atoms.positions[atom_pair[1]] - atoms.positions[atom_pair[0]]
    dir_vec *= step_length/np.linalg.norm(dir_vec)
    atoms.positions[atom_pair[1]] += dir_vec
    write(str(step_idx) + "/POSCAR", atoms, format="vasp")


def copy_files(src_dir, tgt_dir):
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, tgt_dir)

def run_step(step_dir, fix_pair, exe_cmd, inputs_cmds, fmax=0.1, max_steps=50):
    atoms = read(os.path.join(step_dir, "POSCAR"), format="vasp")
    atoms.pbc = [True, True, False]
    bond_constraint(atoms, fix_pair)
    print("creating calculator")
    calculator = set_calc(exe_cmd, step_dir, inputs_cmds)
    print("setting calculator")
    atoms.set_calculator(calculator)
    print("printing atoms")
    print(atoms)
    print("setting optimizer")
    dyn = optimizer(atoms)
    traj = Trajectory(step_dir +'opt.traj', 'w', atoms, properties=['energy', 'forces'])
    print("attaching trajectory")
    dyn.attach(traj.write, interval=1)
    def write_contcar(a=atoms):
        a.write(step_dir +'CONTCAR', format="vasp", direct=True)
        insert_el(step_dir +'CONTCAR')
    dyn.attach(write_contcar, interval=1)
    try:
        dyn.run(fmax=fmax, steps=max_steps)
        finished(step_dir)
    except Exception as e:
        print("couldnt run??")
        print(e)  # Done: make sure this syntax will still print JDFT errors correctly
        assert False, str(e)


if __name__ == '__main__':

    jdftx_exe = os.environ['JDFTx_GPU']
    exe_cmd = 'srun ' + jdftx_exe
    atom_pair, scan_steps, step_length, restart_idx, work_dir = read_scan_inputs()
    if work_dir[-1] != "/":
        work_dir += "/"
    os.chdir(work_dir)
    inputs_cmds = None
    if os.path.exists("inputs"):
        inputs_cmds = read_inputs("inputs")
    if (not os.path.exists("./0")) or (not os.path.isdir("./0")):
        os.mkdir("./0")
    copy_files("./", "./0")
    for i in list(range(1, scan_steps + 1))[restart_idx:]:
        if (not os.path.exists(f"./{str(i)}")) or (not os.path.isdir(f"./{str(i)}")):
            os.mkdir(f"./{str(i)}")
        copy_files(f"./{str(i-1)}", f"./{str(i)}")
        prep_input(i, atom_pair, step_length)
        run_step(work_dir + str(i) +"/", atom_pair, exe_cmd, inputs_cmds, fmax=0.1, max_steps=50)
