# testing OutParser class

import os
from OutParser import OutParser
from ase import Atoms
from ase.visualize import view
from ase.io.trajectory import TrajectoryWriter, TrajectoryReader
from ase.io import read, write
from ase.io.vasp import read_vasp

manager_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(manager_path)

# traj_write = TrajectoryWriter('sample.traj')

parser = OutParser('out', 'opt')
# parser.write_trajectory()
parser.write_contcar()

# for step_index in index_generator:
#     print('step_index:', step_index)
#     forces = parser.forces(step_index)
#     positions, species = parser.positions(step_index)
#     energy = parser.energy(step_index)
#     exc = parser.energy(step_index, energy_str='Exc_core')  
#     print('forces:', forces)
#     print('positions:', positions)
#     print('species:', species)
#     print('energy:', energy)
#     print('exc:', exc)
#     atoms = Atoms(species, positions)
#     print('atoms:', atoms)
#     traj_write.write(atoms, energy=energy, forces=forces)

# traj_write.close()
atoms = read("CONTCAR", format="vasp")
constraints = atoms.constraints
print('constraints:', constraints)

# traj = TrajectoryReader('sample.traj')
# atom = traj[0]

# view(traj)
view(atoms)

text = parser.build_optlog_boilerplate(1)
text += parser.build_optlog()
with open("optlog_test", "w") as f:
    f.write(text)

