#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Script to cut surfaces from bulk materials using bond conservation logic

@author: Cooper_Tezak
"""

from ase.io import read, write
from ase.data import atomic_numbers, covalent_radii, covalent_radii
from ase.build import make_supercell
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
import statistics
import numpy as np
import os
import argparse
import json


##### Sun's Imports #####
from surface.ScreenSurf import ScreenSurf
#########################

def evaluate_list(lst, threshold=5):
    return [1 if item > threshold else 2 for item in lst]

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
    if len(surf_z_coords) == 1: # If there is only one surface atom, the variance is meaningless so the script will just return an unreasonably high number
        top_var = 1000
    else:
        top_var = statistics.variance(surf_z_coords, new_positions[highest_atom_index, 2])
    surf_dens = surf_count/A
    return A, surf_dens, top_var # returns area, atomic surface density, variance from top atom

def calculate_a_b_lenghts(surface_atoms):
    lattice = surface_atoms.get_cell()
    a = np.linalg.norm(lattice[0,:])
    b = np.linalg.norm(lattice[1,:]) # get a and b lengths from unit cell
    return a, b

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

def write_surfaces(surfaces, bulk, index, manager_root):
    index = list(index)
    index = map(str, index)
    index = ''.join(index)
    surface_path = os.path.join(manager_root, "calcs/surfs")
    if not os.path.exists(os.path.join(surface_path, bulk + "_" + index)):
        os.mkdir(os.path.join(surface_path, bulk + "_" + index))
        os.mkdir(os.path.join(surface_path, bulk + "_" + index, "__all_surfs"))
    for i, surf in enumerate(surfaces):
        surf = AseAtomsAdaptor.get_atoms(surf)
        write(os.path.join(surface_path, bulk + "_" + index, "__all_surfs", "POSCAR_" + str(i).zfill(2)), surf)

def write_stats(stats:dict, manager_root):
    with open(os.path.join(manager_root, 'surface_manager_stats.json'), 'w') as f:
        json.dump(stats, f)

def get_bulks(manager_root, rerun):
    bulks = [i for i in os.listdir(os.path.join(manager_root, "calcs/bulks")) if not i.startswith("__")] # ignore files starting with "__"
    made_bulks = [i.split("_")[0] for i in os.listdir(os.path.join(manager_root, "calcs/surfs"))] # finds bulks that have already been made into surfaces
    if rerun:
        return bulks
    else: # if rerun is false, only return bulks that have not been made into surfaces
        return [i for i in bulks if i not in made_bulks]
 
def generate_surfaces(bulk, slab_width, slab_height, num_atoms, num_facets, selection_stats, manager_root):
    '''
    Main function for finding stable surface facets and choosing stable terminations from those facets.

    ==== Inputs ====
    slab_width: float - minimum slab width
    slab_height: int - slab height passed to pymatgen SlabGenerator function
    num_atoms: int - maximum number of atoms in slab unit cell
    num_facets: int - target number of accepted facets. It may not find this many but it definitely won't find more
    selection_stats: dict - dictionary that keeps data on how the algorithm is deciding on surfaces

    ==== Returns ====
    selection_stats: dict - dictionary that keeps data on how the algorithm is deciding on surfaces
    successful_surfaces: int - number of surfaces that the algorithm finds for a given bulk


    '''
    
    bulk_path = os.path.join(manager_root,"calcs/bulks/")
    bulk_atoms = read(os.path.join(bulk_path, bulk, "CONTCAR"))
    bulk_struct = Structure.from_file(os.path.join(bulk_path, bulk, "CONTCAR"))
    SS=ScreenSurf(bulk_struct)
    indices, percentages =SS.ScreenSurfaces(natomsinsphere=50,keep=0.8,samespeconly=False,ignore=[], print_errors=False)
    successful_surfaces = 0 # Add to this count after a surface meets all critera. Stop generating surfaces when this count hits a threshold
    selection_stats[bulk] = {}
    for miller_index in indices: # each miller_index is a unique miller index from Sun's algorithm
        miller_string = ''.join(map(str,list(miller_index)))
        selection_stats[bulk][miller_string] = {}
        slabgen = SlabGenerator(bulk_struct, miller_index, slab_height, min_vacuum_size=15, center_slab=True)
        slabs = slabgen.get_slabs()
        write_surfaces(slabs, bulk, miller_index, manager_root)
        dens = 0 # initializing dens and flatness variables for comparing terminations
        flatness = 1000
        for islab, slab in enumerate(slabs): # each slab is a unique termination
            slab_atoms = AseAtomsAdaptor().get_atoms(slab) # slab atoms denotes terminations that have not yet been accepted
            A, new_dens, new_flatness = calculate_surf_stats(slab_atoms)
            selection_stats[bulk][miller_string].update({str(islab).zfill(2): {'surf_dens': new_dens, 'flatness': new_flatness}})
            if new_dens > dens: # Find the slab with highest atomic density
                dens = new_dens
                flatness = new_flatness
                surface_atoms = slab_atoms # surface_atoms is the accepted termination within the index
                kept_islab = islab
            elif dens - new_dens < 0.001: # check if floats are equal within tolerance
                if new_flatness < flatness: # if surface atomic densities are equal, make decision based on surface flatness
                    flatness = new_flatness
                    surface_atoms = slab_atoms
                    kept_islab = islab

        # now the surface must also pass unit cell size test.
        a, b = calculate_a_b_lenghts(surface_atoms)
        if min([a,b]) < slab_width: # need to make supercell if either dimension is below 5
            supercell_dimensions = evaluate_list([a,b], threshold=slab_width) # returns a list of 1's or 2's depending on size of a and b 
            P = np.array([[supercell_dimensions[0], 0, 0], # transformation matrix to scale surface unit cell.
                        [0, supercell_dimensions[1], 0],
                        [0, 0, 1]])
            surface_atoms = make_supercell(surface_atoms, P)
        
        # finally, the surface must not have too many atoms
        if len(surface_atoms.numbers) < num_atoms:
            # surface passed the final check and is written to the surfaces directory
            successful_surfaces += 1
            selection_stats[bulk][miller_string].update({"selected_termination": str(kept_islab).zfill(2)})
            write_managed_surface(surface_atoms, bulk, miller_index, manager_root)
        if successful_surfaces >= num_facets: # go to next bulk material
            break

    # write_stats(selection_stats, manager_root) # write summary of surface search as json
    return selection_stats, successful_surfaces

def main(slab_width, slab_height, num_atoms, num_facets, rerun):
    manager_root = os.getcwd()
    if "manager_control.txt" not in os.listdir(manager_root):
        raise Exception("manager_control.txt not found in root directory. Make sure this script is run from a gc_manager directory.")

    slab_height = 10
    bulk_path = os.path.join(manager_root, "calcs/bulks")
    selection_stats = {} # dictionary to keep track of which surfaces the algorithm chooses
    selection_stats["converged"] = {}
    bulks_to_go = get_bulks(manager_root, rerun) # keep track of which bulks the algorithm has yet to converge for
    print(bulks_to_go)
    for ih, height in enumerate(range(slab_height, 7, -2)):
        # this height loop will incrementally lower the surface height until the algorithm is able to make surfaces for all the bulks
        print("\n ==================== \n Screening surfaces with height {height} \n ==================== \n".format(height=height))
        for ibulk, bulk in enumerate(bulks_to_go):
            print("\n Screening surfaces for {bulk}".format(bulk=bulk))
            selection_stats, succesful_surfaces = generate_surfaces(bulk, slab_width, height, num_atoms, num_facets, selection_stats, manager_root)
            if succesful_surfaces == num_facets:
                selection_stats["converged"].update({bulk: height})
                bulks_to_go.pop(ibulk) # If the algorithm finds the target number of surfaces, keep the bulk in this list.
    
    selection_stats.update({"unconverged":bulks_to_go}) # write unssuccessful surfaces to selection stats
    write_stats(selection_stats, manager_root) # dump final json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sw', '--slab_width', help='Threshold for determining if a slab is too narrow. defaults to 5 angstroms',
                            type=float, default=5.0)
    parser.add_argument('-sh', '--slab_height', help='Unit cell maximum slab height. defaults to 10 angstroms',
                        type=float, default=10.0)
    parser.add_argument('-na', '--num_atoms', help='Maximum number of atoms per unit cell. Defaults to 40',
                        type=int, default=40)
    parser.add_argument('-nf', '--num_facets', help='Algorithm will generate facets until it hits this number. defaults to 2',
                        type=int, default=2)
    parser.add_argument('-r', '--rerun', help='Whether the algorithm will rerun for all bulks or just do the bulks that are not in the surfs/ directory. defaults to True',
                        type=str, default='True')
    
    args = parser.parse_args()

    # run everything in the main() function
    main(slab_width = args.slab_width, slab_height = args.slab_height, num_atoms = args.num_atoms, num_facets = args.num_facets, rerun = args.rerun)