#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Script to cut surfaces from bulk materials using bond conservation logic

@author: Cooper_Tezak
"""

from ase.io import read, write
from ase.data import atomic_numbers, covalent_radii, covalent_radii
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
import statistics
import numpy as np
import os


##### Sun's Imports #####
from ScreenSurf import ScreenSurf
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#########################


def calculate_surf_stats(surface_atoms):
    '''
    Function to calculate surface properties for evaluating surface stability. The algorithm converts the structure to a cubic
    lattice and then calculates the z coordinate statistics from that lattice. The area is calculated from the original lattice.

    Inputs: ASE atoms object
    Returns:
        A - Surface area in angstroms
        surf_dens - Surface atomic density in atoms/A^2
        top_var - z coordinate variance of surface atoms from z coordinate of top surface atom 

    '''
    A = np.linalg.norm(np.cross(surface_atoms.cell[0], surface_atoms.cell[1])) # surface area
    S = np.eye(3)
    L = np.array(surface_atoms.get_cell())
    np.fill_diagonal(S, np.linalg.norm(L, axis=1))
    positions = surface_atoms.get_positions()
    new_positions = []
    for i, vec in enumerate(positions):
        new_positions.append(np.linalg.inv(L.T).dot(vec)) # convert to fractional coordinates
    new_positions = np.array(new_positions) 
    new_positions = S.dot(new_positions.T).T
    radius_multiplier = 1
    highest_atom_index = np.argmax(new_positions[:,2])
    highest_atom_radius = covalent_radii[atomic_numbers[surface_atoms.get_chemical_symbols()[highest_atom_index]]]
    surf_count = 0
    surf_indices  = []
    for i, vec in enumerate(new_positions):
        if vec[2] + covalent_radii[atomic_numbers[surface_atoms.get_chemical_symbols()[i]]] > new_positions[highest_atom_index, 2] - radius_multiplier*highest_atom_radius:
            surf_count += 1
            surf_indices.append(i)
            continue
    surf_z_coords = new_positions[:,2][surf_indices]
    top_var = statistics.variance(surf_z_coords, new_positions[highest_atom_index, 2])
    surf_dens = surf_count/A
    return A, surf_dens, top_var # returns area, atomic surface density, variance from top atom

def calculate_area(surface_atoms):
    lattice = surface_atoms.

def write_managed_surface(surface_atoms, bulk, index, manager_root):
    index = list(index)
    index = map(str, index)
    index = ''.join(index)
    surface_path = os.path.join(manager_root, "calcs/surfs")
    if not os.path.exists(os.path.join(surface_path, bulk + "_" + index)):
        os.mkdir(os.path.join(surface_path, bulk + "_" + index))
        write(os.path.join(surface_path, bulk + "_" + index, "POSCAR"), surface_atoms)
    else:
        write(os.path.join(surface_path, bulk + "_" + index, "POSCAR"), surface_atoms)


manager_root = os.getcwd()
if "manager_control.txt" not in os.listdir(manager_root):
    raise Exception("manager_control.txt not found in root directory. Make sure this script is run from a gc_manager directory.")





slab_height = 10
bulk_path = os.path.join(manager_root, "calcs/bulks")

for bulk_directory in os.listdir(bulk_path):
    bulk_atoms = read(os.path.join(bulk_path, bulk_directory, "POSCAR"))
    bulk_struct = Structure.from_file(os.path.join(bulk_path, bulk_directory, "POSCAR"))
    SS=ScreenSurf(bulk_struct)
    indices, percentages =SS.ScreenSurfaces(natomsinsphere=50,keep=0.8,samespeconly=False,ignore=[], verbose=False)
    successful_surfaces = 0 # Add to this count after a surface meets all critera. Stop generating surfaces when this count hits a threshold
    for miller_index in indices: # each miller_index is a unique miller index from Sun's algorithm
        slabgen = SlabGenerator(bulk_struct, miller_index, slab_height, min_vacuum_size=15, center_slab=True)
        slabs = slabgen.get_slabs()
        dens = 0 # initializing dens and flatness variables for comparing terminations
        flatness = 1000
        for slab in slabs: # each slab is a unique termination
            slab_atoms = AseAtomsAdaptor().get_atoms(slab) # slab atoms denotes terminations that have not yet been accepted
            A, new_dens, new_flatness = calculate_surf_stats(slab_atoms)
            if new_dens > dens: # Find the slab with highest atomic density
                dens = new_dens
                flatness = new_flatness
                surface_atoms = slab_atoms # surface_atoms is the accepted termination within the index
            elif dens - new_dens < 0.001: # check if floats are equal within tolerance
                if new_flatness < flatness: # if surface atomic densities are equal, make decision based on surface flatness
                    flatness = new_flatness
                    surface_atoms = slab_atoms

        # now the surface must also pass the number of atoms and unit cell size test
        A = 
            write_managed_surface(surface_atoms, bulk_directory, miller_index, manager_root)

    # print(index, percentages)

    






