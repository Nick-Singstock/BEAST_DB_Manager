This is the main control file for using gc_manager.py in a managed directory.
Please read this header section if you are unfamiliar with the manager script.

jdft manager was built to manage a large number of JDFTx surface calculations.
	This script will run unbiased bulk and surface calculations, add requested biases to surfaces,
	add adsorbate molecules to requested surface sites, create desorbed calculations and
	NEB pathways. It will also rerun unconverged jobs and compile and analyze converged
	calculations. All data is saved in the file results/all_data.json and can be used
	for further analysis with python. 

To begin, either:
	1) Add bulk POSCAR to 'calcs/bulks/name' and run manager
	   Once bulk is converged, use 'surface_maker_gc.py' to generate surfaces in 'calcs/surfs/name_facet/__all_surfs'
	   Select chosen surface amongst all generated and place 1 folder up in 'calcs/surfs/name_facet/POSCAR'

	2) OR, Add a surface POSCAR directly to a new subfolder: 'calcs/surfs/surf_name'


To manage this calculation, please add calculation parameters in 'manager_control.txt'. 

	An example of how to manage a surface is included below for a surface named SURF_01 
	which is to be converged first with no bias, then at 0V, -0.1V and -0.5V.  
	Note: biases should be listed in Volts vs SHE (0V). "target-mu" will be updated based on "pcm-variant".
	Note: default system parameters for each calc_type can be changed in 'inputs/' folder
	
	"=" tag indicates a new surface
	"+" tags refer to specific changes made to "inputs" file for respective surface.
		All nested calculations for the surface inherit the same tag changes. 
		Molecule specific changes can also be made separate from surface. pH can be set as a tag.
	"-" adds a molecule to the surface above the listed atoms. Can be 1) name of atom on surface, 
		2) specific atom index, or 3) 'All' positions on surface
	
	Keywords -- for molecules/adsorbates these include: 
		"Biases:" - List of all biases for adsorbate calcs, must be subset of surface biases. 
					Also needed for all surfaces. Must be separated by ", ". Biases in SHE by default.
		"Desorb:" - Creates desorbed state for NEB calcs at listed biases
		"NEB:" - Creates an NEB calc

	Note: Inclusion of a molecule to be an adsorbate will run the corresponding molecule in the 
	molecules/ folder at all requested biases in order to get binding energies.	


----- Syntax Example -----

# setup managed surface, requires 'POSCAR' in folder with name 'calcs/surfs/SURF_01/BIASES'
=SURF_01

# set biases for this surface
Biases: [None, 0, -0.1, -0.5]

# set system-specific inputs tags (applies to surface and adsorbates)
+elec-initial-magnetization 0 no 

# set system-specific convergence tags at corresponding step
+[2] max_steps 0
+[2] pdos element or bit als 
# Note: runs corresponding orbitals for all elements in system of type 'element'

# define molecule to add to surface for adsorbate calc 
# Folder: calcs/adsorbed/SURF_01/MOL_01/BIASES/SITES/
# MOL_01 must be in 'molecules/' folder
# list defines adsorbtion sites, options are: 
   [atom #, 'ontop', 'hollow', 'bridge', 'all', el symbol, (x,y,z) position, 
    '{site_1,site_2,site_2}' triangulation (with apostrophes)]
-MOL_01: [10, 11] # places MOL_01 on atoms 10 and 11

# Note: Molecules are also run in 'calcs/molecules/MOL/BIASES'

# set non-standard distance for adsorbate above sites, default is 2.0
Dist: 1.5 

# set biases for adsorbate calc, must be within set of surface biases
Biases: [0, -0.1]

# set biases for desorbed SP calcs (used for NEB)
Folder: calcs/desorbed/SURF_01/MOL_01/BIASES/SITES/
Desorb: [0, -0.1]

# create NEB pathway between two images
# Folder: calcs/neb/SURF_01/path_name/BIASES
# 'BIAS' should be used in paths instead of specific biases
NEB: path/to/init path/to/final nimages path_name [biases, to, run]



