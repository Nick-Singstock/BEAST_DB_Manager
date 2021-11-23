#!/usr/bin/env python3

#Created on Wed Oct  2 10:14:24 2019
#@author: Nick

from pymatgen.core.surface import SlabGenerator, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Lattice
import os
import argparse
import subprocess

def make_surface(file, folder, index, slab_height, vac_space, center, stoich, primitive, order = True,
                 repeat = 'False'):
    st = Structure.from_file(file)
    mindex = tuple([int(x) for x in index])
    
    # get primitive
    if primitive == 'True':
        sga = SpacegroupAnalyzer(st)
        st = sga.find_primitive()
    
    repeat = False if repeat == 'False' else int(repeat)
    if repeat == False:
        # standard pymatgen surface generation
        slabgen = SlabGenerator(st, mindex, slab_height, vac_space, center_slab=center)
        all_slabs = slabgen.get_slabs(symmetrize = True)
        print("The slab has %s termination." %(len(all_slabs)))
        
        if stoich:
            keep_surfs = []
            bulk_els = [s.species_string for s in st.sites]
            bulk_dic = {el: bulk_els.count(el) for el in bulk_els}
            for surf in all_slabs:
                surf_els = [s.species_string for s in surf.sites]
                surf_dic = {el: surf_els.count(el) for el in surf_els}
                
                stoich_compare = [surf_dic[el] / v if el in surf_dic else 0 for el,v in bulk_dic.items()]
                if all([x == stoich_compare[0] for x in stoich_compare]):
                    keep_surfs.append(surf)
            all_slabs = keep_surfs
            print('Kept '+str(len(all_slabs))+' surfaces with preserved stoichiometry.')
    else:
        all_slabs = [get_repeat_slab(st, repeat, vac_space, center)]
    
    if order:
        new_slabs = []
        for slab in all_slabs:
            new_slabs.append(order_st(slab))
        all_slabs = new_slabs
    
    if not os.path.exists('../../surfs/'+folder+'_'+index):
        os.mkdir('../../surfs/'+folder+'_'+index)
    if not os.path.exists('../../surfs/'+folder+'_'+index+'/__all_surfs/'):
        os.mkdir('../../surfs/'+folder+'_'+index+'/__all_surfs/')

    if len(slab) == 0:
        print('ERROR: No surfaces generated with specified settings.')
        
    for i, slab in enumerate(all_slabs):
        slab.to('POSCAR', '../../surfs/'+folder+'_'+index+'/__all_surfs/' + 'POSCAR_'+str(i).zfill(2))
    
    # copy inputs for bulk for surf kpoints
    subprocess.call('cp inputs '+ '../../surfs/'+
                    folder+'_'+index+'/__all_surfs/bulk_inputs', shell=True)
    
    print(str(i+1)+' slabs for '+folder+' ('+index+') generated at: '+'surfs/'+folder+'_'+index+'/__all_surfs/')
    print('Review surfaces and move best POSCAR into '+'surfs/'+folder+'_'+index+' to be managed by gc_manager.py')

def order_st(st):
    els = [s.species_string for s in st.sites]
    dic = {el: els.count(el) for el in els}
    new_sites = []
    for el in dic:
        for s in st.sites:
            sel = s.species_string
            if sel != el:
                continue
            new_sites.append(s)
    new_st = Structure.from_sites(new_sites)
    return new_st
    
def get_repeat_slab(st, units, vac, center = True):
    sts = st.copy()
    sts.make_supercell([1, 1, units])
    
    lattice_lens = [sts.lattice.a, sts.lattice.b, sts.lattice.c + vac]
    lattice_angles = sts.lattice.angles
    lattice = Lattice.from_lengths_and_angles(lattice_lens, lattice_angles)

    species = [s.species_string for s in sts.sites]
    coords = [s.frac_coords * [1, 1, sts.lattice.c/(sts.lattice.c + vac)] for s in sts.sites]
    
    if center:
        coords = [c + [0, 0, 0.25] for c in coords]
    
    slab = Structure(lattice, species, coords, coords_are_cartesian=False)
    return slab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='File to read (default CONTCAR)',
                        type=str, default="CONTCAR")
    parser.add_argument('-i', '--index', help='Surface Miller Index (default 100)',
                        type=str, default="100")
    parser.add_argument('-sh', '--slab_height', help='Height of the slab (default 10)',
                        type=int, default=10)
    parser.add_argument('-vh', '--vac_space', help='Height of vacuum space (default 15)',
                        type=int, default=15)
    parser.add_argument('-c', '--center', help='Whether to center slab in unit cell (default True)',
                        type=str, default='True')
    parser.add_argument('-ps', '--preserve_stoich', help='Whether to only keep surfaces with the same '+
                        'stoichiometry as the bulk, useful for multinary systems (default False)',
                        type=str, default='False')
    parser.add_argument('-r', '--repeat_bulk', help='False (default) or int. Generate surface by repeating bulk '+
                        'in z-dir for (int) units. Useful when Pymatgen surface creation fails. '+
                        'Folder is named using -i tag.',
                        type=str, default='False')
    parser.add_argument('-p', '--primitive', help='Whether to convert to primitive unit cell first '+
                        '(default False)', type=str, default='False')

    args = parser.parse_args()
	
#    mindex = tuple([int(x) for x in args.index])
    
    folder_name = os.getcwd().split(os.sep)[-1]
    center = True if args.center == 'True' else False
    stoich = True if args.preserve_stoich == 'True' else False
	
    make_surface(args.file, folder_name, args.index, args.slab_height, 
                 args.vac_space, args.center, stoich, args.primitive, repeat = args.repeat_bulk)
