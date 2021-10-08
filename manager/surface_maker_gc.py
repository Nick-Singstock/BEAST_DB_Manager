#!/usr/bin/env python3

#Created on Wed Oct  2 10:14:24 2019
#@author: Nick

from pymatgen.core.surface import SlabGenerator, Structure
import os
import argparse
import subprocess

def make_surface(file, folder, index, slab_height, vac_space, center):
    st = Structure.from_file(file)
    mindex = tuple([int(x) for x in index])
    
    slabgen = SlabGenerator(st, mindex, slab_height, vac_space, center_slab=center)
    all_slabs = slabgen.get_slabs(symmetrize = True)
    print("The slab has %s termination." %(len(all_slabs)))

    
    if not os.path.exists('../../surfs/'+folder+'_'+index):
        os.mkdir('../../surfs/'+folder+'_'+index)
    if not os.path.exists('../../surfs/'+folder+'_'+index+'/__all_surfs/'):
        os.mkdir('../../surfs/'+folder+'_'+index+'/__all_surfs/')

    for i, slab in enumerate(all_slabs):
        slab.to('POSCAR', '../../surfs/'+folder+'_'+index+'/__all_surfs/' + 'POSCAR_'+str(i).zfill(2))
    
    # copy inputs for bulk for surf kpoints
    subprocess.call('cp inputs '+ '../../surfs/'+
                    folder+'_'+index+'/__all_surfs/bulk_inputs', shell=True)
    
    print(str(i+1)+' slabs for '+folder+' ('+index+') generated at: '+'surfs/'+folder+'_'+index+'/__all_surfs/')
    print('Review surfaces and move best POSCAR into '+'surfs/'+folder+'_'+index+' to be managed by gc_manager.py')


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

    args = parser.parse_args()
	
#    mindex = tuple([int(x) for x in args.index])
    
    folder_name = os.getcwd().split(os.sep)[-1]
    center = True if args.center == 'True' else False
	
    make_surface(args.file, folder_name, args.index, args.slab_height, args.vac_space, args.center)
