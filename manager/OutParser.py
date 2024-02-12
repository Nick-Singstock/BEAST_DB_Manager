# class to parse out file from JDFTx

from ase.units import Hartree, Bohr, eV, Angstrom
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory, read, write
from ase.io.trajectory import TrajectoryWriter, TrajectoryReader
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from datetime import datetime, timedelta

class OutParser:
    # could consider adding a start line here that would clear any text
    # before the desired start line
    def __init__(self, file_name, ctype, start_line=None):
        indexing_strings = {"lattice": "LatticeMinimize: Iter:",
                          "opt": "IonicMinimize: Iter:",
                          "sp": "IonicMinimize: Iter:"}
        converged_strings = {"lattice": "LatticeMinimize: Converged",
                            "opt": "IonicMinimize: Converged",
                            "sp": "IonicMinimize: Converged"}
        failed_strings = {"lattice": "LatticeMinimize: None",
                        "opt": "IonicMinimize: None",
                        "sp": "IonicMinimize: None"}
        self.file_name = file_name
        self.ctype = ctype
        self.index_string = indexing_strings[ctype]
        self.converged_string = converged_strings[ctype]
        self.failed_string = failed_strings[ctype]
        with open(self.file_name, 'r') as f:
            text = f.read()
        self.text = text
        def get_start_line():
            # returns the line number of the first line of the out file
            # that is in the ionic minimization
            for i, line in enumerate(self.text.split('\n')):
                if line.startswith("Initialization completed"):
                    return i

        def trimmed_text(start_line):
            # trims the text to start at the first line of the ionic minimization
            # or the start line passed to the class constructor
            trimmed_text = '\n'.join(self.text.split('\n')[start_line:])
            return trimmed_text

        def read_start_time(start_line):
            time_fmt = "%H:%M:%S"
            for line in self.text.split('\n'):
                if line.startswith("Start date and time:"):
                    time_line = line
            time_str = time_line.split()[7]
            start_time = datetime.strptime(time_str, time_fmt)
            return start_time

        if start_line is None:
            self.start_line = get_start_line()
            self.start_time = read_start_time(0)
        else:
            self.start_line = start_line
            self.start_time = read_start_time(self.start_line)
        self.trimmed_text = trimmed_text(start_line)

    def step_indices(self):
        # returns a generator of indices
        # of the out lines that begin with the index_string class
        # attribute set in the __init__ method based on the type of calculation
        # can call list() on the generator to get a list of indices
        for i, line in enumerate(self.trimmed_text.split('\n')):
            if line.startswith(self.index_string):
                yield i

    def forces(self, index):
        text_list = self.trimmed_text.split('\n')
        force_list = []
        for i in range(index, 0, -1):
            line = text_list[i]
            if line.startswith('force'):
                force_list.append(self.parse_force_line(line))
            if line.startswith('# Forces'):
                break

        force_array = np.array(force_list)
        force_array = np.flip(force_array, axis=0)
        return force_array

    def positions(self, index):
        # returns an array of coordinates and a list of atomic symbols
        text_list = self.trimmed_text.split('\n')
        position_list = []
        species_list = []
        selective_dynamics = []
        for i in range(index, 0, -1):
            line = text_list[i]
            if line.startswith('ion'):
                positions, atomic_symbol, freeze = self.parse_position_line(line)
                position_list.append(positions)
                species_list.append(atomic_symbol)
                selective_dynamics.append(int(freeze))
            if line.startswith('# Ionic positions'):
                break

        position_array = np.array(position_list)
        position_array = np.flip(position_array, axis=0)
        species_list.reverse()
        selective_dynamics.reverse()
        return position_array, species_list, selective_dynamics

    def energy(self, index):
        text_list = self.trimmed_text.split('\n')
        for i in range(index, 0, -1):
            line = text_list[i]
            if line.strip().startswith("# Energy components:"):
                start_line = i
                break
        for i, line in enumerate(text_list[start_line:]):
            if not line.strip():
                energy_line = text_list[start_line + i - 1]
                energy = self.parse_energy_line(energy_line)
                break
        return energy

    def time(self, index):
        # time line comes after index
        time_line = self.trimmed_text.split('\n')[index]
        time, unit = self.parse_time_line(time_line)
        return time, unit

    def stress(self, index):
        text_list = self.trimmed_text.split('\n')
        for i in range(index, 0, -1):
            line = text_list[i]
            if line == "# Stress tensor in Cartesian coordinates [Eh/a0^3]:":
                start_index = i
                break
        vec1 = self.parse_cell_line(text_list[start_index+1])
        vec2 = self.parse_cell_line(text_list[start_index+2])
        vec3 = self.parse_cell_line(text_list[start_index+3])
        stress = np.array([vec1, vec2, vec3])
        return stress

    def parse_force_line(self, line):
        # return list of forces in H/Bohr
        line_list = line.split()
        forces = [float(line_list[i]) for i in range(2, 5)]
        return forces

    def parse_position_line(self, line):
        # return list of positions in Bohr
        line_list = line.split()
        positions = [float(line_list[i]) for i in range(2, 5)]
        atomic_symbol = line_list[1]
        freeze = line_list[-1]
        return positions, atomic_symbol, freeze

    def parse_energy_line(self, line):
        # return energy in Hartree
        line_list = line.split()
        energy = float(line_list[2])
        return energy

    def parse_time_line(self, line):
        # return time in seconds
        line_list = line.split()
        time = float(line_list[-1])
        unit_str = line_list[-2]
        unit = unit_str.split("[")[1].split("]")[0]
        return time, unit

    def parse_cell_line(self, line):
        vec = [float(x) for x in line.split()[1:4]]
        return vec

    def build_atoms(self):
        # returns a list of ASE Atoms objects
        atoms_list = []
        if self.ctype == 'opt':
            cell = self.get_start_cell()
            for step_index in self.step_indices():
                atoms = self.get_atoms(step_index)
                atoms.set_cell(cell)
                calc = self.set_calc(atoms, step_index)
                atoms.calc = calc
                atoms_list.append(atoms)
        elif self.ctype == 'lattice':
            for step_index in self.step_indices():
                atoms = self.get_atoms(step_index)
                cell = self.get_cell(step_index)
                atoms.set_cell(cell)
                calc = self.set_calc(atoms, step_index)
                atoms.calc = calc
                atoms_list.append(atoms)
        return atoms_list

    def get_atoms(self, index):
        positions, symbols, selective_dynamics = self.positions(index)
        positions = positions * Bohr

        # in JDFTx a 0 means the atom is frozen, in ASE a 0 means the atom is free
        # so we need to invert the selective dynamics list
        selective_dynamics = [not i for i in selective_dynamics]
        constraint = FixAtoms(mask=selective_dynamics)
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.set_constraint(constraint)
        return atoms

    def set_calc(self, atoms, index):
        # returns a calculator object
        forces = self.forces(index) * Hartree/Bohr
        energy = self.energy(index) * Hartree
        stress = None
        magmoms = None
        if self.ctype == "lattice":
            stress = self.stress(index) * Hartree / Bohr**3
        calc = SinglePointCalculator(atoms=atoms, forces=forces, energy=energy, stress=stress, magmoms=magmoms)
        return calc

    def get_cell(self, index):
        # Returns the cell for a given ionic minimization step
        # This is only to be used in a lattice optimization
        # returns a numpy 3x3 array of cell vectors in Bohr
        lines = self.trimmed_text.split('\n')
        for i in range(index, 0, -1):
            line = lines[i]
            if line.startswith("R = "):
                start_line = i
                break
        vec1 = self.parse_cell_line(lines[start_line+1])
        vec2 = self.parse_cell_line(lines[start_line+2])
        vec3 = self.parse_cell_line(lines[start_line+3])
        cell = np.array([vec1, vec2, vec3])
        return cell * Bohr

    def get_start_cell(self):
        read = False
        i = 0
        cell = np.zeros((3, 3))
        for line in self.text.split('\n'):
            if i == 3: # after reading 3 vectors, break loop
                break
            if line.startswith("lattice "):
                read = True # start reading cell vectors
                continue
            if read:
                cell[i] = [float(x) for x in line.split()[0:3]]
                i += 1
        return cell.T * Bohr

    def write_trajectory(self):
        atoms_list = self.build_atoms()
        writer = TrajectoryWriter('sample.traj', mode='a')
        for atoms in atoms_list:
            writer.write(atoms=atoms)
        writer.close()

    def calc_fmax(self, forces, selective_dynamics):
        # fmax calculation comes from: https://wiki.fysik.dtu.dk/ase/_modules/ase/mep/neb.html#NEBTools.get_fmax
        # note this is different than JDFTx's force convergence criteria: https://github.com/shankar1729/jdftx/issues/244

        # Don't calculate forces on frozen atoms, use the selective dynamics mask
        mask = np.tile(selective_dynamics, (3, 1)).T
        masked_forces = np.ma.masked_array(forces, mask=mask)
        return np.sqrt(((masked_forces * Hartree / Bohr) ** 2).sum(axis=1).max())

    def build_optlog(self, optimizer="LBFGS"):
        text = ""
        step_num = 0
        for step_index in self.step_indices():
            posisiton_array, species_list, selective_dynamics = self.positions(step_index)
            # need to invert selective dynamic list to match ASE convention
            selective_dynamics = [not i for i in selective_dynamics]
            forces = self.forces(step_index)
            fmax = self.calc_fmax(forces, selective_dynamics)
            energy = self.energy(step_index) * Hartree
            time, unit = self.time(step_index)
            if unit == 's':
                time = self.start_time + timedelta(seconds=time)
            elif unit == 'm':
                time = self.start_time + timedelta(minutes=time)
            time_str = time.strftime("%H:%M:%S")
            text += f"{optimizer}: {step_num:>3} {time_str:>1}   {energy:>2.6f}* {fmax:>10.4f}\n"
            step_num += 1
        return text

    def build_optlog_boilerplate(self, convergence_step):
        text = f"Running Convergence Step: {convergence_step}\n"
        text += "      Step     Time          Energy         fmax\n"
        text += "*Force-consistent energies used in optimization.\n"
        return text

    def write_contcar(self):
        # reads the structure for the last step in the ionic minimization
        # and writes it to a CONTCAR file
        last_line = list(self.step_indices())[-1]
        atoms = self.get_atoms(last_line)
        if self.ctype == 'lattice':
            cell = self.get_cell(last_line)
            atoms.set_cell(cell)
        elif self.ctype in ['opt', 'sp']:
            cell = self.get_start_cell()
            atoms.set_cell(cell)
        write("CONTCAR", atoms, format="vasp")

    def check_convergence(self):
        # returns True if the calculation converged, False if it failed
        # returns None if the did not finish or failed
        text_list = self.trimmed_text.split('\n')
        converged_string = self.converged_string
        failed_string = self.failed_string
        converged = False
        for i in range(self.end_of_file_line, 0, -1):
            line = text_list[i]
            if line.startswith(converged_string):
                converged =  True
            elif line.startswith(failed_string):
                converged =  False
        return converged # Assumes the calculation is not converged if it couldn't find either line

    def check_timeout(self):
        # returns True if the calculation timed out, False if it did not
        text_list = self.trimmed_text.split('\n')
        timeout = True
        for i in range(self.end_of_file_line, 0, -1):
            line = text_list[i]
            if line.strip() ==  "Done!":
                timeout = False
        return timeout

    def from_end_of_previous_step(self):
        '''
        Return a new OutParser object from the end of the last written to the parse_i file

        This method will begin reading from that line until it finds the start of the new
        JDFTx run. It returns the new OutParser object with the start_line set to the
        line number of the start of the new JDFTx run.
        '''
        for i, line in enumerate(self.trimmed_text.split('\n')):
            if line.startswith("Initialization completed successfully"):
                # need to add self.start_line to get the line number in the full out file
                # not just the trimmed text
                calculation_start_line = i + self.start_line
        return OutParser(self.file_name, self.ctype, start_line=calculation_start_line)

    def check_calc_started(self):
        # This method checks to see if the JDFTx calculation has started
        # If the JDFTx banner is found, the calculation has started and True is returned
        # If the JDFTx banner is not found, the calculation has not started and False is returned
        text_list = self.trimmed_text.split('\n')
        for line in text_list:
            if line.startswith("************"):
                return True
        return False

    def check_calc_took_step(self):
        # This method checks to see if the JDFTx calculation has taken an ionic/lattice step
        if len(list(self.step_indices())) > 1:
            print(list(self.step_indices()), "step indices")
            return True
        else:
            return False

    @property
    def last_line(self):
        return list(self.step_indices())[-1] + self.start_line

    @property
    def trimmed_last_line(self):
        return list(self.step_indices())[-1]

    @property
    def trimmed_text_lines(self):
        return self.trimmed_text.split('\n')

    @property
    def end_of_file_line(self):
        # returns the line number of the last line in the out file
        return len(self.trimmed_text.split('\n')) - 1

    def write_optlog(self, filename):
        # LBFGS is the correct syntax for the opt.log file
        pass

    def write_parse_file(self, convergence_step):
        # writes the last line of the out file that was parsed
        last_line = self.last_line
        converged = self.check_convergence()
        timeout = self.check_timeout()
        write_str = (f"{last_line} \n"
                     f"converged: {converged} \n"
                    f"timeout: {timeout} ")
        with open(f"parse_{convergence_step}", "w") as f:
                    f.write(write_str)
